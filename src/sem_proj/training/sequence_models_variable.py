"""
Training script for variable-length sequence models.

Fine-tunes a GRU classifier on variable-length sequences extracted from
continuous sleep recordings, using naturally-bounded segments.
"""

from typing import Optional
import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
from functools import partial

# Set seeds
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from sem_proj.data.datasets_variable import BoasVariableLengthSequenceDataset, variable_length_collate_fn
from sem_proj.data.preprocessing import PreprocessingConfig, get_expected_seq_length
from sem_proj.data.splits import get_train_subjects, get_val_subjects
from sem_proj.models.model_factory import SSLEpochTransformerConv1D_v2
from sem_proj.models.model_factory_variable import SequenceGRUClassifier_Variable, SequenceGRUClassifier_VariableWithAttention

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"


def make_variable_dataloaders(
    batch_size: int = 4,
    min_seq_len: int = 5,
    mode: str = "headband",
    preprocess_config: Optional[PreprocessingConfig] = None,
    use_cache: bool = True,
    use_augmentation: bool = False,
    train_subjects: Optional[list] = None,
    val_subjects: Optional[list] = None,
):
    """
    Create dataloaders for variable-length sequences.
    
    Parameters
    ----------
    batch_size : int
        Batch size. Keep small due to variable-length sequences.
    min_seq_len : int
        Minimum sequence length to include.
    mode : str
        "headband", "psg", or "cross".
    preprocess_config : PreprocessingConfig, optional
        Preprocessing configuration.
    use_cache : bool
        Whether to use cached data.
    use_augmentation : bool
        Whether to apply augmentation (typically False for variable-length).
    train_subjects : list, optional
        List of training subject IDs. If None, uses get_train_subjects().
    val_subjects : list, optional
        List of validation subject IDs. If None, uses get_val_subjects().
    
    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    # Use provided subjects or default to standard splits
    tr_subs = train_subjects if train_subjects is not None else get_train_subjects()
    val_subs = val_subjects if val_subjects is not None else get_val_subjects()
    
    # Augmentation not recommended for variable-length (disrupts natural boundaries)
    train_transform = None
    
    train_ds = BoasVariableLengthSequenceDataset(
        subjects=tr_subs,
        mode=mode,
        min_seq_len=min_seq_len,
        transform_hb=train_transform,
        transform_psg=None,
        target_transform=None,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
    )
    
    val_ds = BoasVariableLengthSequenceDataset(
        subjects=val_subs,
        mode=mode,
        min_seq_len=min_seq_len,
        transform_hb=None,
        transform_psg=None,
        target_transform=None,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
    )
    
    # Create collate function with mode
    collate_fn = partial(variable_length_collate_fn, mode=mode)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


def evaluate_variable_model(model, dataloader, criterion, device, mode="headband"):
    """
    Evaluate variable-length sequence model.
    
    Properly handles padding by masking loss and metrics.
    
    Returns
    -------
    avg_loss : float
    accuracy : float
    macro_f1 : float
    per_class_f1 : np.ndarray
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if mode in {"headband", "psg"}:
                x_batch, y_batch, lengths = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                lengths = lengths.to(device)
                
                # Forward pass
                logits = model(x_batch, lengths)  # (B, L_max, num_classes)

                ### use mixed precision evaluation ###
                with autocast():
                    logits = model(x_batch, lengths)  # (B, L_max, num_classes)
                ########
                
            else:  # cross mode
                x_hb, x_psg, y_batch, lengths = batch
                # For cross mode, you'd need to modify model to handle both inputs
                raise NotImplementedError("Cross mode not yet implemented for variable-length")
            
            # # Flatten for loss computation
            # logits_flat = logits.view(-1, logits.size(-1))  # (B*L_max, num_classes)
            # y_flat = y_batch.view(-1)  # (B*L_max,)
            
            # # Compute loss (CrossEntropyLoss automatically ignores -100)
            # loss = criterion(logits_flat, y_flat)


            ### use mixed precision evaluation instead ###
            with autocast():
                # Flatten for loss computation
                logits_flat = logits.view(-1, logits.size(-1))  # (B*L_max, num_classes)
                y_flat = y_batch.view(-1)  # (B*L_max,)
                # Compute loss (CrossEntropyLoss automatically ignores -100)
                loss = criterion(logits_flat, y_flat)
            ########


            total_loss += loss.item() * x_batch.size(0)
            
            # Get predictions
            preds_flat = torch.argmax(logits_flat, dim=1)  # (B*L_max,)
            
            # Mask out padding for metrics
            valid_mask = (y_flat != -100)
            valid_preds = preds_flat[valid_mask].cpu().numpy()
            valid_labels = y_flat[valid_mask].cpu().numpy()
            
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)
            
            # Count correct predictions
            total_correct += (valid_preds == valid_labels).sum()
            total_samples += valid_mask.sum().item()
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    return avg_loss, accuracy, macro_f1, per_class_f1


def train_variable_gru(
    num_epochs: int = 50,
    batch_size: int = 4,
    min_seq_len: int = 5,
    lr_encoder: float = 1e-5,
    lr_gru: float = 1e-4,
    mode: str = "headband",
    experiment_name: str = "variable_gru_v1",
    preprocess_config: Optional[PreprocessingConfig] = None,
    use_cache: bool = True,
    # Epoch encoder params
    encoder_checkpoint: Optional[Path] = None,
    freeze_encoder: bool = False,
    fully_supervised: bool = True,
    d_model: int = 128,
    nhead: int = 4,
    num_layers_encoder: int = 3,
    dim_feedforward: int = 512,
    dropout_encoder: float = 0.2,
    target_tokens: int = 240,
    # GRU params
    gru_hidden: int = 128,
    gru_layers: int = 2,
    gru_bidirectional: bool = False,
    gru_dropout: float = 0.2,
    use_attention: bool = False,
    # Training params
    class_weighted_loss: bool = False,
    gradient_clip: float = 5.0,
    early_stopping_patience: int = 10,
    num_classes: int = 5,
    # Data split params
    train_subjects: Optional[list] = None,
    val_subjects: Optional[list] = None,
):
    """
    Train variable-length GRU sequence classifier.
    
    This is the main training function:
    - Extract sequences until artifact/disconnection
    - Use GRU to model variable-length sequences
    - Fine-tune pretrained epoch encoder (optional)
    
    Parameters
    ----------
    num_epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size (keep small for variable-length, e.g., 4-8).
    min_seq_len : int
        Minimum sequence length to include.
    lr_encoder : float
        Learning rate for epoch encoder (if unfrozen).
    lr_gru : float
        Learning rate for GRU layers.
    mode : str
        "headband" or "psg".
    experiment_name : str
        Experiment name for checkpoints/logs.
    preprocess_config : PreprocessingConfig, optional
        Preprocessing configuration.
    use_cache : bool
        Use cached data.
    encoder_checkpoint : Path, optional
        Path to pretrained epoch encoder checkpoint.
    freeze_encoder : bool
        Whether to freeze epoch encoder during training.
    d_model, nhead, num_layers_encoder, etc.
        Epoch encoder architecture params.
    gru_hidden, gru_layers, gru_bidirectional
        GRU architecture params.
    use_attention : bool
        Use attention-augmented GRU.
    class_weighted_loss : bool
        Use class-weighted loss.
    gradient_clip : float
        Gradient clipping max norm.
    early_stopping_patience : int
        Early stopping patience.
    train_subjects : list, optional
        List of training subject IDs. If None, uses get_train_subjects().
    val_subjects : list, optional
        List of validation subject IDs. If None, uses get_val_subjects().
    """
    checkpoint_path = CHECKPOINT_DIR / experiment_name
    log_path = LOG_DIR / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(log_path))
    
    if preprocess_config is None:
        preprocess_config = PreprocessingConfig.from_yaml(
            PROJECT_ROOT / "configs" / "preprocess" / "notch_bandpass_resample_znorm.yaml"
        )
    
    seq_length = get_expected_seq_length(preprocess_config)
    
    # Print configuration
    print(f"\n{'='*70}")
    print(f"Variable-Length GRU Training Configuration")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Min sequence length: {min_seq_len}")
    print(f"Mode: {mode}")
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Learning rates: encoder={lr_encoder}, gru={lr_gru}")
    print(f"\nEpoch Encoder:")
    print(f"  d_model={d_model}, nhead={nhead}, layers={num_layers_encoder}")
    print(f"  target_tokens={target_tokens}, dropout={dropout_encoder}")
    print(f"  Status: {'pretrained from ' + str(encoder_checkpoint) if encoder_checkpoint else 'random init (fully supervised)'}")
    print(f"\nGRU:")
    print(f"  hidden={gru_hidden}, layers={gru_layers}, bidirectional={gru_bidirectional}")
    print(f"  dropout={gru_dropout}, attention={use_attention}")
    print(f"  Learning rates: encoder={lr_encoder}, gru={lr_gru}")
    if fully_supervised and encoder_checkpoint is None:
        print(f"  (Using uniform LR={lr_gru} for fully supervised training)")
    print(f"\nPreprocessing: {preprocess_config.to_dict()}")
    print(f"{'='*70}\n")
    
    # Create dataloaders
    train_loader, val_loader = make_variable_dataloaders(
        batch_size=batch_size,
        min_seq_len=min_seq_len,
        mode=mode,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
    )
    
    print(f"Training sequences: {len(train_loader.dataset)}")
    print(f"Validation sequences: {len(val_loader.dataset)}\n")
    
    # Build epoch encoder
    input_channels = 2 if mode == "headband" else 6
    
    epoch_model = SSLEpochTransformerConv1D_v2(
        input_channels=input_channels,
        seq_length=seq_length,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers_encoder,
        dim_feedforward=dim_feedforward,
        dropout=dropout_encoder,
        num_classes=num_classes,
        target_tokens=target_tokens,
    )
    
    # Load pretrained encoder if provided
    if encoder_checkpoint is not None:
        print(f"Loading pretrained encoder from {encoder_checkpoint}")
        checkpoint = torch.load(encoder_checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            epoch_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            epoch_model.load_state_dict(checkpoint)
        
        print("Pretrained encoder loaded successfully!\n")
    
    # Build sequence model
    if use_attention:
        model = SequenceGRUClassifier_VariableWithAttention(
            epoch_model=epoch_model,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            num_classes=num_classes,
            bidirectional=gru_bidirectional,
            dropout=gru_dropout,
            use_attention=True,
        )
    else:
        model = SequenceGRUClassifier_Variable(
            epoch_model=epoch_model,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            num_classes=num_classes,
            bidirectional=gru_bidirectional,
            dropout=gru_dropout,
        )
    
    model = model.to(device)
    
    # Freeze encoder if requested
    if freeze_encoder:
        model.freeze_epoch_encoder()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.epoch_model.parameters() if p.requires_grad)
    gru_params = total_params - encoder_params
    
    print(f"Model parameters:")
    print(f"  Epoch encoder: {encoder_params:,} ({'frozen' if freeze_encoder else 'trainable'})")
    print(f"  GRU + Classifier: {gru_params:,}")
    print(f"  Total trainable: {total_params:,}\n")
    
    # Loss and optimizer
    if class_weighted_loss:
        # Compute class weights from training data
        print("Computing class weights from training data...")
        class_counts = np.zeros(num_classes)
        
        for batch in tqdm(train_loader, desc="Computing weights"):
            if mode in {"headband", "psg"}:
                _, y_batch, _ = batch
            else:
                _, _, y_batch, _ = batch
            
            valid_labels = y_batch[y_batch != -100].numpy()
            for cls in range(num_classes):
                class_counts[cls] += (valid_labels == cls).sum()
        
        total = class_counts.sum()
        # Apply square root smoothing to reduce extreme weight disparities
        class_weights = total / (num_classes * class_counts)
        class_weights = np.sqrt(class_weights)  # Square root smoothing
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        
        print(f"Class weights (with sqrt smoothing): {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Optimizer setup
    if fully_supervised and encoder_checkpoint is None:
        # Fully supervised from scratch: same LR everywhere
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_gru,  # Use GRU LR as baseline
            weight_decay=1e-2,
        )
    elif freeze_encoder:
        # Fine-tune with frozen encoder: only train GRU
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_gru,
            weight_decay=1e-2,
        )
    elif fully_supervised and encoder_checkpoint is not None:
        raise ValueError("fully_supervised=True with encoder_checkpoint not None is not a valid configuration.")
    else:
        # Fine-tune with unfrozen encoder: different LRs
        optimizer = torch.optim.AdamW([
            {'params': model.epoch_model.parameters(), 'lr': lr_encoder},
            {'params': model.gru.parameters(), 'lr': lr_gru},
            {'params': model.classifier.parameters(), 'lr': lr_gru},
            {'params': model.dropout.parameters(), 'lr': lr_gru},
        ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, verbose=True
    )
    
    ### use mixed precision training ###
    scaler = GradScaler()   # for mixed precision training
    ######

    # Training loop
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            if mode in {"headband", "psg"}:
                x_batch, y_batch, lengths = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                lengths = lengths.to(device)
            else:
                raise NotImplementedError("Cross mode not implemented")
            
            # optimizer.zero_grad()
            
            # # Forward
            # logits = model(x_batch, lengths)  # (B, L_max, num_classes)
            
            # # Compute loss
            # logits_flat = logits.view(-1, logits.size(-1))
            # y_flat = y_batch.view(-1)
            # loss = criterion(logits_flat, y_flat)
            
            # # Backward
            # loss.backward()
            
            # # Gradient clipping
            # if gradient_clip > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # optimizer.step()


            ### use mixed precision training instead ###
            optimizer.zero_grad()
            # Forward with mixed precision
            with autocast():
                logits = model(x_batch, lengths)  # (B, L_max, num_classes)
                
                # Compute loss
                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y_batch.view(-1)
                loss = criterion(logits_flat, y_flat)
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            # Gradient clipping (unscale first)
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            ########

            
            # Metrics
            train_loss += loss.item() * x_batch.size(0)
            
            preds_flat = torch.argmax(logits_flat, dim=1)
            valid_mask = (y_flat != -100)
            train_correct += (preds_flat[valid_mask] == y_flat[valid_mask]).sum().item()
            train_total += valid_mask.sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / max(train_total, 1)})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        val_loss, val_acc, val_macro_f1, val_per_class_f1 = evaluate_variable_model(
            model, val_loader, criterion, device, mode=mode
        )
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val_macro', val_macro_f1, epoch)
        
        for i, f1 in enumerate(val_per_class_f1):
            writer.add_scalar(f'F1/val_{class_names[i]}', f1, epoch)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f}")
        print(f"Per-class F1: {' | '.join([f'{name}: {f1:.3f}' for name, f1 in zip(class_names, val_per_class_f1)])}")
        
        # Learning rate scheduling
        scheduler.step(val_macro_f1)
        
        # Save best model
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': val_macro_f1,
                'val_per_class_f1': val_per_class_f1,
                'config': {
                    'experiment_name': experiment_name,
                    'batch_size': batch_size,
                    'min_seq_len': min_seq_len,
                    'lr_encoder': lr_encoder,
                    'lr_gru': lr_gru,
                    'freeze_encoder': freeze_encoder,
                    'd_model': d_model,
                    'gru_hidden': gru_hidden,
                    'gru_layers': gru_layers,
                    'gru_bidirectional': gru_bidirectional,
                    'use_attention': use_attention,
                    'preprocess_config': preprocess_config.to_dict(),
                }
            }
            
            torch.save(checkpoint, checkpoint_path / "best_model.pt")
            print(f"✓ Saved best model (F1: {val_macro_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation macro F1: {best_val_f1:.4f}")
    print(f"Model saved to: {checkpoint_path / 'best_model.pt'}")
    print(f"{'='*70}\n")
    
    return best_val_f1


if __name__ == "__main__":
    # Example training run
    from sem_proj.data.preprocessing import PreprocessingConfig
    
    config = PreprocessingConfig.from_yaml(
        PROJECT_ROOT / "configs" / "preprocess" / "notch_bandpass_resample_znorm.yaml"
    )
    
    train_variable_gru(
        num_epochs=100,
        batch_size=4,
        min_seq_len=5,
        lr_encoder=1e-5,
        lr_gru=1e-4,
        mode="headband",
        experiment_name="variable_gru_headband_v1",
        preprocess_config=config,
        use_cache=True,
        encoder_checkpoint=None,  # Set to pretrained SSL checkpoint if available
        freeze_encoder=False,
        fully_supervised=True,
        d_model=96,
        gru_hidden=128,
        gru_layers=2,
        gru_bidirectional=True,
        use_attention=False,
        class_weighted_loss=True,
    )
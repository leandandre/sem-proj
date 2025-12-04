from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np
import random
import os

# Set seeds
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from sem_proj.data.datasets import BoasSequenceDataset
from sem_proj.data.preprocessing import PreprocessingConfig, get_expected_seq_length
from sem_proj.data.splits import get_train_subjects, get_val_subjects
from sem_proj.data.transforms import RandomTimeShift, RandomAmplitudeScale, RandomGaussianNoise, Compose
from sem_proj.models.model_factory import EpochTransformerConv1D_v2, SequenceGRUClassifier, SequenceTransformerClassifier

# Project root = .../sem-proj
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs" / "preprocess"


def compute_class_weights(dataloader, num_classes=5):
    """
    Compute inverse frequency class weights from sequence dataset.
    
    Note: For BoasSequenceDataset, we need to flatten sequences.
    """
    from collections import Counter
    
    all_labels = []
    for batch in dataloader:
        # Handle both (x_seq, y_seq) and (x_hb_seq, x_psg_seq, y_seq) returns
        y_seq = batch[-1]  # Last element is always y_seq
        # y_seq shape: (B, seq_len)
        all_labels.extend(y_seq.flatten().numpy())
    
    counter = Counter(all_labels)
    total = len(all_labels)
    
    # Inverse frequency weights
    weights = []
    for class_idx in range(num_classes):
        count = counter.get(class_idx, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def make_seq_dataloaders(
    batch_size: int = 16,
    seq_len: int = 20,
    stride: int = 5,
    mode: str = "headband",
    preprocess_config: Optional[PreprocessingConfig] = None,
    use_cache: bool = True,
    use_augmentation: bool = True,
    add_channel_dim: bool = False,  # Set to False since our model expects (B, L, C, T)
):
    """
    Create sequence dataloaders for SequenceGRUClassifier.
    
    Parameters
    ----------
    batch_size : int
        Batch size.
    seq_len : int
        Number of consecutive epochs per sequence.
    stride : int
        Stride between sequence start positions.
    mode : str
        "headband", "psg", or "cross".
    preprocess_config : PreprocessingConfig, optional
        Preprocessing configuration.
    use_cache : bool
        Whether to use cached preprocessed data.
    use_augmentation : bool
        Whether to apply data augmentation (time-shift + amplitude scaling).
    add_channel_dim : bool
        If True, add singleton channel dimension. Set False for (B, L, C, T) format.
    
    Returns
    -------
    train_loader, val_loader : DataLoader
        Training and validation dataloaders.
    """
    # Load fixed splits
    tr_subs = get_train_subjects()
    val_subs = get_val_subjects()

    # Define augmentation transforms (only for training)
    train_transform = None
    if use_augmentation:
        train_transform = Compose([
            RandomTimeShift(max_shift_ratio=0.15),           # +-15% time shift, maybe increase?
            RandomAmplitudeScale(scale_range=(0.8, 1.2)),    # +-20% amplitude, maybe increase?
            RandomGaussianNoise(noise_scale=(0.01, 0.05)),   # Add Gaussian noise
        ])

    train_ds = BoasSequenceDataset(
        subjects=tr_subs,
        mode=mode,
        seq_len=seq_len,
        stride=stride,
        transform_hb=train_transform,   # Apply augmentation
        transform_psg=None,
        target_transform=None,
        add_channel_dim=add_channel_dim,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
    )
    
    val_ds = BoasSequenceDataset(
        subjects=val_subs,
        mode=mode,
        seq_len=seq_len,
        stride=stride,
        transform_hb=None,              # No augmentation for validation
        transform_psg=None,
        target_transform=None,
        add_channel_dim=add_channel_dim,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,          # local: 2, cluster: 7
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,          # local: 2, cluster: 7
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return train_loader, val_loader


def evaluate_sequence_model(model, dataloader, criterion, device):
    """
    Evaluate sequence model on a dataset.
    
    Returns
    -------
    avg_loss : float
        Average loss over all batches
    accuracy : float
        Overall accuracy (correct predictions / total samples)
    macro_f1 : float
        Macro-averaged F1 score
    per_class_f1 : np.ndarray
        F1 score for each class
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle both single and cross modality
            if len(batch) == 2:  # (x_seq, y_seq)
                x_seq, y_seq = batch
                x_seq = x_seq.to(device, dtype=torch.float32)
            else:  # (x_hb_seq, x_psg_seq, y_seq)
                x_hb_seq, x_psg_seq, y_seq = batch
                x_seq = x_hb_seq.to(device, dtype=torch.float32)  # Use headband for now
            
            y_seq = y_seq.to(device, dtype=torch.long)
            
            # Forward pass
            logits = model(x_seq)  # (B, L, num_classes)
            B, L, K = logits.shape
            
            # Compute loss
            loss = criterion(
                logits.view(B * L, K),
                y_seq.view(B * L),
            )
            
            # Accumulate loss
            total_loss += loss.item() * B
            
            # Compute accuracy
            preds = logits.argmax(dim=-1)  # (B, L)
            total_correct += (preds == y_seq).sum().item()
            total_samples += y_seq.numel()
            
            # Store predictions and labels for F1 computation
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(y_seq.cpu().numpy().flatten())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    
    # Macro F1 (average across classes)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    return avg_loss, accuracy, macro_f1, per_class_f1


def train_sequence_model(
    num_epochs: int = 50,
    batch_size: int = 8,
    seq_len: int = 20,
    stride: int = 5,
    lr: float = 1e-4,
    mode: str = "headband",
    experiment_name: str = "sequence_gru_v1",
    preprocess_config: Optional[PreprocessingConfig] = None,
    use_cache: bool = True,
    class_weighted_loss: bool = False,
    use_augmentation: bool = True,
    # EpochTransformer hyperparameters
    d_model: int = 96,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 384,
    dropout: float = 0.2,
    target_tokens: int = 240,
    # Sequence model selection
    use_gru: bool = True,
    use_transformer: bool = False,
    # GRU hyperparameters
    gru_hidden: int = 128,
    gru_layers: int = 1,
    gru_bidirectional: bool = False,
    # Transformer hyperparameters (only used if use_transformer=True)
    d_model_seq: int = 96,
    nhead_seq: int = 4,
    num_layers_seq: int = 2,
    dim_feedforward_seq: int = 384,
    num_classes: int = 5,
):
    """
    Train SequenceGRUClassifier model.
    
    This model:
    1. Encodes each epoch with EpochTransformerConv1D_v2 (frozen or trainable)
    2. Feeds epoch embeddings through GRU to capture temporal dependencies
    3. Classifies each epoch in the sequence (many-to-many)
    
    Parameters
    ----------
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size (smaller than epoch models due to sequences).
    seq_len : int
        Number of consecutive epochs per sequence.
    stride : int
        Stride between sequence start positions.
    lr : float
        Learning rate.
    mode : str
        "headband", "psg", or "cross".
    experiment_name : str
        Name for this experiment (used for checkpoints and logs).
    preprocess_config : PreprocessingConfig, optional
        Preprocessing configuration.
    use_cache : bool
        Whether to use cached preprocessed data.
    class_weighted_loss : bool
        Whether to use class-balanced loss.
    use_augmentation : bool
        Whether to apply data augmentation.
    d_model : int
        Embedding dimension for epoch transformer.
    nhead : int
        Number of attention heads in epoch transformer.
    num_layers : int
        Number of transformer layers in epoch model.
    dim_feedforward : int
        Feedforward dimension in epoch transformer.
    dropout : float
        Dropout rate.
    target_tokens : int
        Number of tokens after convolution (240 or 480).
    gru_hidden : int
        GRU hidden size.
    gru_layers : int
        Number of GRU layers.
    gru_bidirectional: bool
        Whether to use bidirectional GRU.
    num_classes : int
        Number of output classes (sleep stages).
    """
    checkpoint_path = CHECKPOINT_DIR / experiment_name
    log_path = LOG_DIR / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(log_path))

    # Determine sequence length from preprocessing config
    if preprocess_config is None:
        preprocess_config = PreprocessingConfig.no_preprocessing()
    
    epoch_seq_length = get_expected_seq_length(preprocess_config)
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Epoch window length: {epoch_seq_length} timepoints")
    print(f"Sequence length: {seq_len} epochs")
    print(f"Stride: {stride} epochs")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Using cache: {use_cache}")
    print(f"Using augmentation: {use_augmentation}")
    print(f"\nPreprocessing config:")
    for key, value in preprocess_config.to_dict().items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # Create dataloaders
    train_loader, val_loader = make_seq_dataloaders(
        batch_size=batch_size,
        seq_len=seq_len,
        stride=stride,
        mode=mode,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        use_augmentation=use_augmentation,
        add_channel_dim=False,  # Model expects (B, L, C, T)
    )
    
    print(f"Training sequences: {len(train_loader.dataset)}")
    print(f"Validation sequences: {len(val_loader.dataset)}")
    print(f"Underlying training epochs: {len(train_loader.dataset.epoch_dataset)}")
    print(f"Underlying validation epochs: {len(val_loader.dataset.epoch_dataset)}\n")

    # Infer number of channels and samples from data
    example_batch = next(iter(train_loader))
    example_x = example_batch[0] if len(example_batch) == 2 else example_batch[0]
    # example_x: (B, L, C, T)
    B, L, C, T_samples = example_x.shape
    print(f"Example batch shape: B={B}, L={L}, C={C}, T={T_samples}")
    
    # Validate sequence length matches preprocessing
    assert T_samples == epoch_seq_length, (
        f"Expected {epoch_seq_length} samples from preprocessing, got {T_samples}"
    )
    print(f"✓ Sequence length matches preprocessing config\n")

    # Build epoch encoder (EpochTransformerConv1D_v2)
    epoch_model = EpochTransformerConv1D_v2(
        input_channels=C,
        seq_length=epoch_seq_length,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_classes=num_classes,  # Not used, we bypass classifier
        target_tokens=target_tokens,
    ).to(device)

    # Build sequence classifier (GRU or Transformer)
    if use_gru:
        model = SequenceGRUClassifier(
            epoch_model=epoch_model,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            num_classes=num_classes,
            bidirectional=gru_bidirectional,
        ).to(device)
    elif use_transformer:
        model = SequenceTransformerClassifier(
            epoch_model=epoch_model,
            d_model_seq=d_model_seq,
            nhead=nhead_seq,
            num_layers=num_layers_seq,
            dim_feedforward=dim_feedforward_seq,
            dropout=dropout,
            num_classes=num_classes,
        ).to(device)
    else:
        raise ValueError("Either use_gru=True or use_transformer=True must be set")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters\n")

    # Loss function and optimizer
    if class_weighted_loss:
        class_weights = compute_class_weights(train_loader, num_classes=num_classes).to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}\n")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        manual_weights = torch.tensor([0.7, 2.5, 0.35, 2.0, 0.7], dtype=torch.float32, device=device) # change for stronger/weaker weighting
        manual_weights = manual_weights * (manual_weights.numel() / manual_weights.sum())
        criterion = nn.CrossEntropyLoss(weight=manual_weights, label_smoothing=0.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2) # evlt 1e-4?
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6
    )

    # Training config
    training_config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'stride': stride,
        'learning_rate': lr,
        'use_cache': use_cache,
        'class_weighted_loss': class_weighted_loss,
        'use_augmentation': use_augmentation,
        'model_type': 'SequenceGRUClassifier' if use_gru else 'SequenceTransformerClassifier',
        'optimizer': 'AdamW',
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_patience': 6,
        'scheduler_factor': 0.5,
        'early_stop_patience': 12,
        'num_trainable_params': num_params,
        'train_sequences': len(train_loader.dataset),
        'val_sequences': len(val_loader.dataset),
        'd_model': d_model,
        'nhead': nhead,
        'transformer_layers': num_layers,
        'target_tokens': target_tokens,
        'gru_hidden': gru_hidden,
        'gru_layers': gru_layers,
        'gru_bidirectional': gru_bidirectional,
    }

    # Early stopping
    early_stop_patience = 12
    epochs_since_improvement = 0

    # Training loop
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    global_step = 0
    best_val_acc = 0.0
    best_macro_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # === TRAINING ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training")
        for batch in train_pbar:
            # Handle both single and cross modality
            if len(batch) == 2:  # (x_seq, y_seq)
                x_seq, y_seq = batch
                x_seq = x_seq.to(device, dtype=torch.float32)
            else:  # (x_hb_seq, x_psg_seq, y_seq)
                x_hb_seq, x_psg_seq, y_seq = batch
                x_seq = x_hb_seq.to(device, dtype=torch.float32)
            
            y_seq = y_seq.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            logits = model(x_seq)  # (B, L, num_classes)
            B_cur, L_cur, K = logits.shape
            
            loss = criterion(
                logits.view(B_cur * L_cur, K),
                y_seq.view(B_cur * L_cur),
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)    # try max_norm=1.0?
            optimizer.step()
            
            batch_loss = loss.item()
            preds = logits.argmax(dim=-1)
            batch_correct = (preds == y_seq).sum().item()
            batch_samples = y_seq.numel()
            
            train_loss += batch_loss * B_cur
            train_correct += batch_correct
            train_samples += batch_samples
            
            writer.add_scalar('Loss/train_batch', batch_loss, global_step)
            batch_acc = batch_correct / batch_samples
            writer.add_scalar('Accuracy/train_batch', batch_acc, global_step)
            
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.3f}'
            })
            
            global_step += 1
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_samples
        
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', train_accuracy, epoch)
        
        # === VALIDATION ===
        print("Evaluating on validation set...")
        val_loss, val_accuracy, val_macro_f1, val_per_class_f1 = evaluate_sequence_model(
            model, val_loader, criterion, device
        )
        
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Macro F1: {val_macro_f1:.4f}")
        print(f"Per-class F1 scores:")
        for i, (cls_name, f1) in enumerate(zip(class_names, val_per_class_f1)):
            print(f"  {cls_name}: {f1:.4f}")
            writer.add_scalar(f'F1/val_{cls_name}', f1, epoch)
        
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Accuracy/val_epoch', val_accuracy, epoch)
        writer.add_scalar('F1/val_macro', val_macro_f1, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Scheduler step
        scheduler.step(val_macro_f1)
        
        # Save best model
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_val_acc = val_accuracy
            epochs_since_improvement = 0
            checkpoint_file = checkpoint_path / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'val_macro_f1': val_macro_f1,
                'val_per_class_f1': val_per_class_f1.tolist(),
                'hyperparameters': {
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dim_feedforward': dim_feedforward,
                    'dropout': dropout,
                    'target_tokens': target_tokens,
                    'gru_hidden': gru_hidden,
                    'gru_layers': gru_layers,
                    'gru_bidirectional': gru_bidirectional,
                    'num_classes': num_classes,
                },
                'preprocessing': preprocess_config.to_dict(),
                'training_config': training_config,
                'class_weights': class_weights.cpu().tolist() if class_weighted_loss else None,
            }, checkpoint_file)
            print(f"✓ Saved best model (macro F1: {val_macro_f1:.4f})")
        else:
            epochs_since_improvement += 1
            print(f"No improvement in macro F1 for {epochs_since_improvement} epoch(s).")
        
        # Early stopping
        if epochs_since_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered (no improvement for {early_stop_patience} epochs).")
            break
        
        # Save latest checkpoint
        latest_checkpoint = checkpoint_path / "latest_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'val_macro_f1': val_macro_f1,
            'val_per_class_f1': val_per_class_f1.tolist(),
            'hyperparameters': {
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dim_feedforward': dim_feedforward,
                'dropout': dropout,
                'target_tokens': target_tokens,
                'gru_hidden': gru_hidden,
                'gru_layers': gru_layers,
                'gru_bidirectional': gru_bidirectional,
                'num_classes': num_classes,
            },
            'preprocessing': preprocess_config.to_dict(),
            'training_config': training_config,
        }, latest_checkpoint)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Macro F1 Score: {best_macro_f1:.4f}")
    print(f"Checkpoints saved to: {checkpoint_path}")
    print(f"TensorBoard logs saved to: {log_path}")
    print(f"{'='*60}")
    
    writer.close()
    
    return model


if __name__ == "__main__":
    # Configuration
    CONFIG_NAME = "notch_bandpass_resample_znorm"  # Full preprocessing
    
    # Training hyperparameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 8          # Smaller because sequences use more memory
    SEQ_LEN = 20            # 20 consecutive epochs = 10 minutes
    STRIDE = 1              # determines overlap between sequences
    LEARNING_RATE = 1e-4    # Lower LR for sequence models
    USE_CACHE = True
    WEIGHTED_LOSS = False
    USE_AUGMENTATION = True
    
    # EpochTransformer hyperparameters (for epoch encoding)
    D_MODEL = 96
    N_HEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = D_MODEL * 4
    DROPOUT = 0.3
    TARGET_TOKENS = 240     # 240 or 480
    
    # Model type selection
    USE_GRU = True
    USE_TRANSFORMER = False
    
    # GRU hyperparameters (for sequence modeling)
    GRU_HIDDEN = 128
    GRU_LAYERS = 1
    GRU_BIDIRECTIONAL = True
    
    # Transformer hyperparameters (only used if USE_TRANSFORMER=True)
    D_MODEL_SEQ = 96
    NHEAD_SEQ = 4
    NUM_LAYERS_SEQ = 2
    DIM_FEEDFORWARD_SEQ = D_MODEL_SEQ * 4
    
    # Load preprocessing config
    config_file = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    preproc_cfg = PreprocessingConfig.from_yaml(config_file)
    
    # Generate experiment name based on model type
    VERSION = 1
    model_type_str = "gru" if USE_GRU else "transformer"
    experiment_name = f"sequence_{model_type_str}_{CONFIG_NAME}_seq{SEQ_LEN}_stride{STRIDE}_v{VERSION}"
    
    print(f"\nStarting Sequence{model_type_str.upper()}Classifier training")
    print(f"Config file: {config_file}")
    
    model = train_sequence_model(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        lr=LEARNING_RATE,
        mode="headband",
        experiment_name=experiment_name,
        preprocess_config=preproc_cfg,
        use_cache=USE_CACHE,
        class_weighted_loss=WEIGHTED_LOSS,
        use_augmentation=USE_AUGMENTATION,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        target_tokens=TARGET_TOKENS,
        use_gru=USE_GRU,
        use_transformer=USE_TRANSFORMER,
        gru_hidden=GRU_HIDDEN,
        gru_layers=GRU_LAYERS,
        gru_bidirectional=GRU_BIDIRECTIONAL,
        d_model_seq=D_MODEL_SEQ,
        nhead_seq=NHEAD_SEQ,
        num_layers_seq=NUM_LAYERS_SEQ,
        dim_feedforward_seq=DIM_FEEDFORWARD_SEQ,
        num_classes=5,
    )
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np
import random, os
random.seed(42); os.environ["PYTHONHASHSEED"]="42"; np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

from sem_proj.data.datasets import BoasDataset
from sem_proj.data.preprocessing import PreprocessingConfig, get_expected_seq_length
from sem_proj.models.model_factory import EpochTransformer, EpochTransformerConv1D, EpochTransformerConv1D_v2, MultiChannelSleepNet
from sem_proj.data.splits import load_splits, get_train_subjects, get_val_subjects
from sem_proj.data.transforms import RandomTimeShift, RandomAmplitudeScale, RandomGaussianNoise, Compose

# Project root = .../sem-proj
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs" / "preprocess"

# Utility function to compute class weights to handle class imbalance
def compute_class_weights(dataloader, num_classes=5):
    """Compute inverse frequency class weights."""
    from collections import Counter
    
    all_labels = []
    for _, y in dataloader:
        all_labels.extend(y.numpy())
    
    counter = Counter(all_labels)
    total = len(all_labels)
    
    # Inverse frequency weights
    weights = []
    for class_idx in range(num_classes):
        count = counter.get(class_idx, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)



def make_dataloaders(batch_size: int = 16,
                     preprocess_config: Optional[PreprocessingConfig] = None,
                     use_cache: bool = True,
                     use_augmentation: bool = True
):
    # Load fixed splits
    tr_subs = get_train_subjects()
    val_subs = get_val_subjects()

    # Define augmentation transforms (only for training)
    train_transform = None
    if use_augmentation:
        train_transform = Compose([
            RandomTimeShift(max_shift_ratio=0.10),      # +-% time shift
            RandomAmplitudeScale(scale_range=(0.9, 1.1)),  # +-% amplitude
            # RandomGaussianNoise(noise_scale=(0.01, 0.05)),   # Add Gaussian noise
        ])

    train_ds = BoasDataset(
        subjects=tr_subs, 
        mode="headband",
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        transform_hb=train_transform    # augment training samples
    )
    val_ds = BoasDataset(
        subjects=val_subs, 
        mode="headband",
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        transform_hb=None   # no augmentation for validation
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,      # local: 2, cluster: 7
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,    # optional
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,    # local: 2, cluster: 7
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,    # optional
    )

    return train_loader, val_loader


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset.
    
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
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            # Accumulate loss
            total_loss += loss.item() * x.size(0)
            
            # Compute accuracy
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            
            # Store predictions and labels for F1 computation
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # Macro F1 (average across classes, treating each class equally)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    return avg_loss, accuracy, macro_f1, per_class_f1


def train_epochtransformer(
    num_epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    experiment_name: str = "epoch_transformer_v1",
    model_kwargs: dict | None = None,
    preprocess_config: Optional[PreprocessingConfig] = None,
    use_cache: bool = True,
    class_weighted_loss: bool = False,
    method: str = "conv_transformer",
    use_augmentation: bool = True
):
    """
    Train EpochTransformer model.

    Parameters
    ----------
    num_epochs : int
    batch_size : int
    lr : float
    experiment_name : str
        Name for this experiment (used for checkpoints and logs)
    model_kwargs : dict | None
        Overrides for model constructor.
    preprocess_config : PreprocessingConfig | None
        Preprocessing configuration.
    use_cache : bool
        Whether to use cached preprocessed data.
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
    
    seq_length = get_expected_seq_length(preprocess_config)
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Expected window length: {seq_length} timepoints")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Using cache: {use_cache}")
    print(f"\nPreprocessing config:")
    for key, value in preprocess_config.to_dict().items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # Default model config
    default_model_cfg = dict(
        input_channels=2,
        seq_length=seq_length,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=5,
    )
    if model_kwargs:
        default_model_cfg.update(model_kwargs)
        if 'seq_length' in model_kwargs and model_kwargs['seq_length'] != seq_length:
            print(f"WARNING: Ignoring model_kwargs['seq_length']={model_kwargs['seq_length']}. "
                  f"Using seq_length={seq_length} from preprocessing config.")
            default_model_cfg['seq_length'] = seq_length

    # Choose model architecture
    if method == "conv_transformer":
        model = EpochTransformerConv1D_v2(**default_model_cfg).to(device)
        print("Using Conv1D patch embedding")
    elif method == "mean_pool_transformer":
        model = EpochTransformer(**default_model_cfg).to(device)
        print("Using mean-pooling patch embedding")
    elif method == "multichannel_sleepnet":
        model = MultiChannelSleepNet().to(device)   # use default params of the model
        print("Using MultiChannelSleepNet model")
    else: 
        raise ValueError(f"Unknown method: {method}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters\n")
    
    # Create dataloaders
    train_loader, val_loader = make_dataloaders(
        batch_size=batch_size, 
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        use_augmentation=use_augmentation
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")

    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    # Loss function and optimizer
    if class_weighted_loss:     # "balanced class weighting"
        class_weights = compute_class_weights(train_loader, num_classes=5).to(device)
        print(f"\nClass weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else: # manual weights or unweighted
        manual_weights = torch.tensor([1.0, 1.5, 0.7, 1.5, 1.0], dtype=torch.float32, device=device)
        manual_weights = manual_weights * (manual_weights.numel() / manual_weights.sum())  # normalize to mean = 1.0
        criterion = nn.CrossEntropyLoss(weight=manual_weights, label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6
    )

    # Create complete training config dictionary
    training_config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'use_cache': use_cache,
        'class_weighted_loss': class_weighted_loss,
        'method': method,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_patience': 6,
        'scheduler_factor': 0.5,
        'early_stop_patience': 12,
        'num_trainable_params': num_params,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        # 'target_tokens': model.target_tokens,
        # 'patch_size': model.patch_size,
        # 'final_seq_length': model.final_seq_length
    }

    # Early stopping state
    early_stop_patience = 12
    epochs_since_improvement = 0

    # Training loop
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
        for x, y in train_pbar:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            batch_loss = loss.item()
            preds = logits.argmax(dim=1)
            batch_correct = (preds == y).sum().item()
            batch_size_actual = y.size(0)
            
            train_loss += batch_loss * batch_size_actual
            train_correct += batch_correct
            train_samples += batch_size_actual
            
            writer.add_scalar('Loss/train_batch', batch_loss, global_step)
            batch_acc = batch_correct / batch_size_actual
            writer.add_scalar('Accuracy/train_batch', batch_acc, global_step)
            
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.3f}'
            })
            
            global_step += 1
        
        avg_train_loss = train_loss / train_samples
        train_accuracy = train_correct / train_samples
        
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', train_accuracy, epoch)
        
        # === VALIDATION ===
        print("Evaluating on validation set...")
        val_loss, val_accuracy, val_macro_f1, val_per_class_f1 = evaluate_model(
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

        # Scheduler step (after val metrics)
        scheduler.step(val_macro_f1)
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_val_acc = val_accuracy
            epochs_since_improvement = 0    # reset counter
            checkpoint_file = checkpoint_path / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'val_macro_f1': val_macro_f1,
                'val_per_class_f1': val_per_class_f1.tolist(),
                'hyperparameters': default_model_cfg,
                'preprocessing': preprocess_config.to_dict(),
                'training_config': training_config,
                'class_weights': class_weights.cpu().tolist() if class_weighted_loss else None,  # ADD THIS
            }, checkpoint_file)
            print(f"✓ Saved best model (macro F1: {val_macro_f1:.4f})")
        else:
            epochs_since_improvement += 1
            print(f"No improvement in macro F1 for {epochs_since_improvement} epoch(s).")
        
        # Early stopping condition
        if epochs_since_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered (no macro F1 improvement for {early_stop_patience} epochs).")
            break

        
        latest_checkpoint = checkpoint_path / "latest_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'val_macro_f1': val_macro_f1,
            'val_per_class_f1': val_per_class_f1.tolist(),
            'hyperparameters': default_model_cfg,
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
    # CONFIG_NAME = "no_preprocess"                  # No preprocessing
    # CONFIG_NAME = "notch_bandpass"                 # Just filters
    # CONFIG_NAME = "notch_bandpass_znorm"           # Filters + znorm
    # CONFIG_NAME = "notch_bandpass_resample"        # Filters + resample
    # CONFIG_NAME = "only_znorm"                     # Just normalization
    
    # Training hyperparameters
    NUM_EPOCHS = 120
    BATCH_SIZE = 128     # look at GPU memory and choose in {32, 64, 128, 256}
    LEARNING_RATE = 2e-3    # try 1e-3, 2e-3, 5e-4, depending on model size
    USE_CACHE = True    # Set to False to disable caching

    WEIGHTED_LOSS = False  # Set to True to use class-balanced loss
    METHOD = 'multichannel_sleepnet'  # select {'conv_transformer', 'mean_pool_transformer', 'multichannel_sleepnet'}
    USE_AUGMENTATION = True  # Set to True to use data augmentation
    
    D_MODEL = 96  # Reduced model size for testing, change to 64 later
    N_HEAD = 4     # 4 or 8 heads
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = D_MODEL * 4   # always d_model * 4
    DROPOUT = 0.2
    TARGET_TOKENS = 240   # SHOULD BE in {240, 480}
    
    
    # Load preprocessing config
    config_file = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}\nAvailable configs: {list(CONFIG_DIR.glob('*.yaml'))}")
    
    preproc_cfg = PreprocessingConfig.from_yaml(config_file)
    
    # Model config (seq_length is automatically determined from preprocessing)
    model_cfg = {
        'input_channels': 2,   # fixed for headband data
        'd_model': D_MODEL,
        'nhead': N_HEAD,
        'num_layers': NUM_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'num_classes': 5,       # fixed for 5 sleep stages
        'target_tokens': TARGET_TOKENS
    }
    
    # Generate experiment name based on config
    VERSION = 1     # CHANGE for each new run
    model_type = METHOD
    experiment_name = f"epochlevel_{model_type}_{CONFIG_NAME}_v{VERSION}"
    
    print(f"\n Starting training with {model_type} model")
    print(f" Config file: {config_file}")
    
    model = train_epochtransformer(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        experiment_name=experiment_name,
        model_kwargs=model_cfg,
        preprocess_config=preproc_cfg,
        use_cache=USE_CACHE,
        class_weighted_loss=WEIGHTED_LOSS,
        method=METHOD,
        use_augmentation=USE_AUGMENTATION
    )
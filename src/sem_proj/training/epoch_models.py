from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np

from sem_proj.data.datasets import BoasDataset
from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.models.model_factory import EpochTransformer
from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.data.splits import load_splits, get_train_subjects, get_val_subjects

# Project root = .../sem-proj
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"


def make_dataloaders(batch_size: int = 16, preprocess_config: Optional[PreprocessingConfig] = None):
    # Load fixed splits, call by seperate functions for clarity
    tr_subs = get_train_subjects()
    val_subs = get_val_subjects()

    train_ds = BoasDataset(
        subjects=tr_subs, 
        mode="headband",
        preprocess_config=preprocess_config 
    )
    val_ds = BoasDataset(
        subjects=val_subs, 
        mode="headband",
        preprocess_config=preprocess_config 
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
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
    preprocess_config: Optional[PreprocessingConfig] = None  
):
    """
    Train EpochTransformer model.

    Parameters
    ----------
    num_epochs : int
    batch_size : int
    lr : float
    experiment_name : str
    model_kwargs : dict | None
        Overrides for model constructor, e.g. {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.2,
            'input_channels': 2,
            'seq_length': 7680,
            'num_classes': 5
        }.
    """
    checkpoint_path = CHECKPOINT_DIR / experiment_name
    log_path = LOG_DIR / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(log_path))

    # Default model config
    default_model_cfg = dict(
        input_channels=2,
        seq_length=7680,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=5,
    )
    if model_kwargs:
        default_model_cfg.update(model_kwargs)

    model = EpochTransformer(**default_model_cfg).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Log preprocessing config
    if preprocess_config is not None:
        print(f"Preprocessing config: {preprocess_config.to_dict()}")
    
    # Create dataloaders
    train_loader, val_loader = make_dataloaders(batch_size=batch_size, preprocess_config=preprocess_config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
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
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            batch_loss = loss.item()
            preds = logits.argmax(dim=1)
            batch_correct = (preds == y).sum().item()
            batch_size_actual = y.size(0)
            
            train_loss += batch_loss * batch_size_actual
            train_correct += batch_correct
            train_samples += batch_size_actual
            
            # Log batch metrics
            writer.add_scalar('Loss/train_batch', batch_loss, global_step)
            batch_acc = batch_correct / batch_size_actual
            writer.add_scalar('Accuracy/train_batch', batch_acc, global_step)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.3f}'
            })
            
            global_step += 1
        
        # Compute epoch training metrics
        avg_train_loss = train_loss / train_samples
        train_accuracy = train_correct / train_samples
        
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        
        # Log epoch training metrics
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
        
        # Log validation metrics
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Accuracy/val_epoch', val_accuracy, epoch)
        writer.add_scalar('F1/val_macro', val_macro_f1, epoch)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save checkpoint if best validation metric (use macro F1 as primary metric)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_val_acc = val_accuracy
            checkpoint_file = checkpoint_path / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'val_macro_f1': val_macro_f1,
                'val_per_class_f1': val_per_class_f1.tolist(),
                'hyperparameters': default_model_cfg | {'lr': lr, 'batch_size': batch_size},
            }, checkpoint_file)
            print(f"✓ Saved best model (macro F1: {val_macro_f1:.4f})")
        
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
        }, latest_checkpoint)
    
    # Training complete
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
    model_cfg = {
        'input_channels': 2,
        'seq_length': 7680,
        'd_model': 64,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 64*4,
        'dropout': 0.2,
        'num_classes': 5,
    }
    # Option 1: Train WITH preprocessing
    preproc_cfg = PreprocessingConfig(
        notch_freqs=[50.0, 100.0],  # Remove power line noise
        bandpass_l_freq=0.5,
        bandpass_h_freq=40.0,
        resample_freq=128.0,  # Downsample from 256 Hz to 128 Hz
        apply_preprocessing=True,
    )
    ### important: adjust seq_length based on resample_freq
    model_cfg['seq_length'] = int(preproc_cfg.to_dict()['resample_freq'] * 30)  # 30 seconds epochs  = 3840 samples
    # Train the model
    model = train_epochtransformer(
        num_epochs=10,
        batch_size=4,
        lr=1e-3,
        experiment_name="epoch_transformer_v1",
        model_kwargs=model_cfg,
        preprocess_config=preproc_cfg
    )

    # Option 2: Train WITHOUT preprocessing (for comparison)
    # no_preproc_cfg = PreprocessingConfig.no_preprocessing()
    # model_cfg_raw = model_cfg.copy()
    # model_cfg_raw['seq_length'] = 7680  # 256 Hz * 30 sec
    # 
    # model_raw = train_epochtransformer(
    #     num_epochs=20,
    #     batch_size=4,
    #     lr=1e-3,
    #     experiment_name="transformer_raw_v1",
    #     model_kwargs=model_cfg_raw,
    #     preprocess_config=no_preproc_cfg,
    # )

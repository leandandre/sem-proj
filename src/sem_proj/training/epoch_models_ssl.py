# REUSE: All imports
from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np
import random, os

from sem_proj.data.preprocessing import PreprocessingConfig, get_expected_seq_length
from sem_proj.data.datasets import BoasDataset
from sem_proj.data.splits import get_train_subjects, get_val_subjects
from sem_proj.models.model_factory import SSLEpochTransformerConv1D, SSLEpochTransformerConv1D_v2,SSLClassifierHead
from sem_proj.data.transforms import RandomTimeShift, RandomAmplitudeScale, RandomGaussianNoise, Compose

random.seed(42); os.environ["PYTHONHASHSEED"]="42"; np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs" / "preprocess"



def make_dataloaders_ssl(batch_size: int = 16, preprocess_config: Optional[PreprocessingConfig] = None, use_cache: bool = True):
    """Create dataloaders for SSL (both headband and PSG needed)."""
    tr_subs = get_train_subjects()
    val_subs = get_val_subjects()

    # Use mode="cross" to get both modalities
    train_ds = BoasDataset(
        subjects=tr_subs, 
        mode="cross",
        preprocess_config=preprocess_config,
        use_cache=use_cache
    )
    val_ds = BoasDataset(
        subjects=val_subs, 
        mode="cross",
        preprocess_config=preprocess_config,
        use_cache=use_cache
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # Important for contrastive learning
        pin_memory=True,
        persistent_workers=False,   # if num_workers > 0, set to True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        persistent_workers=False,   # if num_workers > 0, set to True
    )

    return train_loader, val_loader

def token_contrastive_loss(hb_tokens: torch.Tensor, psg_tokens: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Token-level symmetric InfoNCE between headband and PSG.

    Args:
        hb_tokens:  (B, L, D) tensor tokens from headband encoder.
        psg_tokens: (B, L, D) tensor tokens from PSG encoder.
        temperature: scalar float for scaling the logits.

    Returns:
        Scalar loss (torch.Tensor of shape []).
    """
    assert hb_tokens.shape == psg_tokens.shape, "Shapes of hb_tokens and psg_tokens must match"
    B, L, D = hb_tokens.shape

    hb = hb_tokens.reshape(B * L, D)
    psg = psg_tokens.reshape(B * L, D)

    hb = F.normalize(hb, dim=-1)   # (BL, D)
    psg = F.normalize(psg, dim=-1) # (BL, D)

    logits = hb @ psg.T            # (BL, BL)
    logits = logits / temperature

    labels = torch.arange(B * L, device=hb.device)

    # hb to psg
    loss_hb2psg = F.cross_entropy(logits, labels)

    # psg to hb
    loss_psg2hb = F.cross_entropy(logits.T, labels)

    # final symmetric loss
    loss = 0.5 * (loss_hb2psg + loss_psg2hb)
    return loss

def global_contrastive_loss(z_a, z_b, temperature=0.07):
    # z_a, z_b: (B, D)
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.T) / temperature  # (B, B)
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_ab + loss_ba)



def train_ssl_epochtransformer(
    num_epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-4,
    experiment_name: str = "ssl_transformer_v1",
    model_kwargs: dict | None = None,
    preprocess_config: Optional[PreprocessingConfig] = None,
    use_cache: bool = True,
    temperature: float = 0.07,          
    lambda_token: float = 1.0,          
    lambda_global: float = 1.0,       
):
    """
    Train SSL encoder with cross-modal contrastive learning.
    
    Parameters
    ----------
    temperature : float
        Temperature scaling for contrastive loss.
    lambda_token : float
        Weight for token-level contrastive loss.
    lambda_global : float
        Weight for global contrastive loss.
    """
    checkpoint_path = CHECKPOINT_DIR / experiment_name
    log_path = LOG_DIR / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(log_path))

    if preprocess_config is None:
        preprocess_config = PreprocessingConfig.no_preprocessing()
    seq_length = get_expected_seq_length(preprocess_config)

    # Print config summary
    print(f"\n{'='*60}")
    print(f"SSL Training Configuration")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Temperature: {temperature}")
    print(f"Lambda token: {lambda_token}")
    print(f"Lambda global: {lambda_global}")
    print(f"Using cache: {use_cache}")
    print(f"\nPreprocessing config:")
    for key, value in preprocess_config.to_dict().items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    default_model_cfg = dict(
        input_channels=2,   # code will later create separate models for headband (2) and PSG (6)
        seq_length=seq_length,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.2,
        num_classes=5,  # ignored in SSL
        target_tokens=240,     
    )

    if model_kwargs:
        default_model_cfg.update(model_kwargs)

    # create two models: one for headband (2 channels), one for PSG (6 channels)
    model_hb_cfg = default_model_cfg.copy()
    model_hb_cfg['input_channels'] = 2  # Headband
    model_hb = SSLEpochTransformerConv1D_v2(**model_hb_cfg).to(device)
    
    model_psg_cfg = default_model_cfg.copy()
    model_psg_cfg['input_channels'] = 6  # PSG
    model_psg = SSLEpochTransformerConv1D_v2(**model_psg_cfg).to(device)
    num_params_hb = sum(p.numel() for p in model_hb.parameters() if p.requires_grad)
    num_params_psg = sum(p.numel() for p in model_psg.parameters() if p.requires_grad)
    print(f"Headband encoder: {num_params_hb:,} parameters")
    print(f"PSG encoder: {num_params_psg:,} parameters")
    print(f"Total: {num_params_hb + num_params_psg:,} parameters\n")

    train_loader, val_loader = make_dataloaders_ssl(
        batch_size=batch_size, 
        preprocess_config=preprocess_config,
        use_cache=use_cache
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")

    # Optimize both models, AdamW with weight decay
    optimizer = torch.optim.AdamW(
        list(model_hb.parameters()) + list(model_psg.parameters()), 
        lr=lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6
    )

    # Training config for checkpoint
    training_config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'temperature': temperature,
        'lambda_token': lambda_token,
        'lambda_global': lambda_global,
        'use_cache': use_cache,
        'optimizer': 'AdamW',
        'weight_decay': 0.01,
        'scheduler': 'ReduceLROnPlateau',
        'early_stop_patience': 10,
        'num_params_hb': num_params_hb,
        'num_params_psg': num_params_psg,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
    }

    early_stop_patience = 10
    epochs_since_improvement = 0
    best_val_loss = float('inf')  # for SSL track loss, not F1
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # === TRAINING ===
        model_hb.train()
        model_psg.train()
        train_loss = 0.0
        train_loss_token = 0.0
        train_loss_global = 0.0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training SSL")
        for x_hb, x_psg, y in train_pbar:  # gets x_hb, x_psg, y (y is ignored)
            x_hb = x_hb.to(device)  # (batch, 2, seq_length)
            x_psg = x_psg.to(device)  # (batch, 6, seq_length)

            optimizer.zero_grad()
            
            # Forward pass through both models
            tokens_hb = model_hb(x_hb)   # (batch, num_tokens, d_model)
            tokens_psg = model_psg(x_psg)  # (batch, num_tokens, d_model)
            
            z_hb = tokens_hb.mean(dim=1)    # (batch, d_model)
            z_psg = tokens_psg.mean(dim=1)  # (batch, d_model)
            
            # Compute contrastive loss
            loss_token = token_contrastive_loss(tokens_hb, tokens_psg, temperature)
            loss_global = global_contrastive_loss(z_hb, z_psg, temperature)
            loss = lambda_token * loss_token + lambda_global * loss_global

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model_hb.parameters()) + list(model_psg.parameters()), 
                max_norm=1.0
            )
            optimizer.step()

            batch_size_actual = x_hb.size(0)
            train_loss += loss.item() * batch_size_actual
            train_loss_token += loss_token.item() * batch_size_actual
            train_loss_global += loss_global.item() * batch_size_actual
            train_samples += batch_size_actual
            
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            writer.add_scalar('Loss/train_token', loss_token.item(), global_step)
            writer.add_scalar('Loss/train_global', loss_global.item(), global_step)
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            global_step += 1
        
        avg_train_loss = train_loss / train_samples
        avg_train_loss_token = train_loss_token / train_samples
        avg_train_loss_global = train_loss_global / train_samples
        
        print(f"Train Loss: {avg_train_loss:.4f} "
              f"(token: {avg_train_loss_token:.4f}, global: {avg_train_loss_global:.4f})")
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_epoch_token', avg_train_loss_token, epoch)
        writer.add_scalar('Loss/train_epoch_global', avg_train_loss_global, epoch)

        # === VALIDATION ===
        model_hb.eval()
        model_psg.eval()
        val_loss = 0.0
        val_loss_token = 0.0
        val_loss_global = 0.0
        val_samples = 0

        with torch.no_grad():
            for x_hb, x_psg, y in val_loader:
                x_hb = x_hb.to(device)
                x_psg = x_psg.to(device)
                
                tokens_hb = model_hb(x_hb)
                tokens_psg = model_psg(x_psg)
                
                z_hb = tokens_hb.mean(dim=1)
                z_psg = tokens_psg.mean(dim=1)
                
                loss_token = token_contrastive_loss(tokens_hb, tokens_psg, temperature=temperature)
                loss_global = global_contrastive_loss(z_hb, z_psg, temperature=temperature)
                loss = lambda_token * loss_token + lambda_global * loss_global
                
                val_loss += loss.item() * x_hb.size(0)
                val_loss_token += loss_token.item() * x_hb.size(0)
                val_loss_global += loss_global.item() * x_hb.size(0)
                val_samples += x_hb.size(0)

        avg_val_loss = val_loss / val_samples
        avg_val_loss_token = val_loss_token / val_samples
        avg_val_loss_global = val_loss_global / val_samples
        
        print(f"Val Loss: {avg_val_loss:.4f} "
              f"(token: {avg_val_loss_token:.4f}, global: {avg_val_loss_global:.4f})")
        
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Loss/val_epoch_token', avg_val_loss_token, epoch)
        writer.add_scalar('Loss/val_epoch_global', avg_val_loss_global, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            
            checkpoint_file = checkpoint_path / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'encoder_hb_state_dict': model_hb.state_dict(),
                'encoder_psg_state_dict': model_psg.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_loss_token': avg_val_loss_token,
                'val_loss_global': avg_val_loss_global,
                'hyperparameters_hb': model_hb_cfg,
                'hyperparameters_psg': model_psg_cfg,
                'preprocessing': preprocess_config.to_dict(),
                'training_config': training_config,
            }, checkpoint_file)
            print(f" Saved best model (val loss: {avg_val_loss:.4f})")
        else:
            epochs_since_improvement += 1
            print(f"No improvement in val loss for {epochs_since_improvement} epoch(s).")
        
        if epochs_since_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered.")
            break

        # Save latest
        latest_checkpoint = checkpoint_path / "latest_model.pt"
        torch.save({
            'epoch': epoch,
            'encoder_hb_state_dict': model_hb.state_dict(),
            'encoder_psg_state_dict': model_psg.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_loss_token': avg_val_loss_token,
            'val_loss_global': avg_val_loss_global,
            'hyperparameters_hb': model_hb_cfg,
            'hyperparameters_psg': model_psg_cfg,
            'preprocessing': preprocess_config.to_dict(),
            'training_config': training_config,
        }, latest_checkpoint)

    print(f"\n{'='*60}")
    print(f"SSL Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_path}")
    print(f"{'='*60}")
    
    writer.close()
    return model_hb, model_psg


### forget about this function for now ###
def fine_tune_ssl_encoder(
        encoder_checkpoint_path: Path,
        num_epochs: int = 50,
        batch_size: int = 128,
        lr_encoder: float = 1e-5,
        lr_classifier: float = 1e-3,
        experiment_name: str = "ssl_finetuned_v1",
        preprocess_config: Optional[PreprocessingConfig] = None,
        use_cache: bool = True,
        class_weighted_loss: bool = True,
        freeze_encoder: bool = False,       # False for end-to-end, True for two-stage approach
        label_fraction: float = 1.0,
):
    checkpoint_path = CHECKPOINT_DIR / experiment_name
    log_path = LOG_DIR / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(log_path))
    if preprocess_config is None:
        preprocess_config = PreprocessingConfig.no_preprocessing()
    
    print(f"\n{'='*60}")
    print(f"Fine-Tuning SSL Encoder")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Encoder checkpoint: {encoder_checkpoint_path}")
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Label fraction: {label_fraction * 100:.1f}%")
    print(f"LR encoder: {lr_encoder}")
    print(f"LR classifier: {lr_classifier}")
    print(f"{'='*60}\n")

    # === LOAD SSL ENCODER ===
    print("Loading SSL encoder...")
    ssl_checkpoint = torch.load(encoder_checkpoint_path, map_location='cpu')

    encoder_hb_cfg = ssl_checkpoint['hyperparameters_hb']
    encoder = SSLEpochTransformerConv1D_v2(**encoder_hb_cfg).to(device)
    encoder.load_state_dict(ssl_checkpoint['encoder_hb_state_dict'])
    print(f"✓ Loaded encoder from epoch {ssl_checkpoint['epoch'] + 1}")

    # === CREATE CLASSIFIER HEAD ===
    d_model = encoder_hb_cfg['d_model']
    dropout = encoder_hb_cfg['dropout']
    classifier = SSLClassifierHead(
        d_model=d_model,
        dropout=dropout,
        num_classes=5
    ).to(device)

    num_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    num_params_classifier = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Encoder parameters: {num_params_encoder:,}")
    print(f"Classifier parameters: {num_params_classifier:,}")
    print(f"Total: {num_params_encoder + num_params_classifier:,}\n")

    # === FREEZE ENCODER (OPTIONAL) ===
    if freeze_encoder:
        print("Freezing encoder weights (only training classifier)")
        for param in encoder.parameters():
            param.requires_grad = False
        num_trainable = num_params_classifier
    else:
        print("Training encoder + classifier end-to-end")
        num_trainable = num_params_encoder + num_params_classifier

    # === CREATE DATALOADERS ===
    tr_subs = get_train_subjects()
    val_subs = get_val_subjects()

    # Subsample training subjects for low-label experiments
    if label_fraction < 1.0:
        np.random.seed(42)
        n_train = int(len(tr_subs) * label_fraction)
        tr_subs = list(np.random.choice(tr_subs, n_train, replace=False))
        print(f"Using {len(tr_subs)} training subjects ({label_fraction*100:.0f}% of full set)")

    # augment training data for fine-tuning
    train_transform = Compose([
        RandomTimeShift(max_shift_ratio=0.1),      # +-10% time shift
        RandomAmplitudeScale(scale_range=(0.9, 1.1)),  # +-10% amplitude
        RandomGaussianNoise(noise_scale=(0.01, 0.05)),   # Add Gaussian noise
    ])

    train_ds = BoasDataset(
        subjects=tr_subs,
        mode="headband",
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        transform_hb=train_transform
    )
    val_ds = BoasDataset(
        subjects=val_subs,
        mode="headband",
        preprocess_config=preprocess_config,
        use_cache=use_cache
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=7,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=7,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")

    # === LOSS & OPTIMIZER ===
    if class_weighted_loss:
        from sem_proj.training.epoch_models import compute_class_weights
        class_weights = compute_class_weights(train_loader, num_classes=5).to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)     # can also be without label smoothing
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)   # can also be without label smoothing
    
    # Separate parameter groups with different learning rates
    if freeze_encoder:
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=lr_classifier,
            weight_decay=0.01
        )
    else:
        optimizer = torch.optim.AdamW([
            {'params': encoder.parameters(), 'lr': lr_encoder},
            {'params': classifier.parameters(), 'lr': lr_classifier}
        ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6
    )
    # === TRAINING CONFIG ===
    training_config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr_encoder': lr_encoder,
        'lr_classifier': lr_classifier,
        'freeze_encoder': freeze_encoder,
        'label_fraction': label_fraction,
        'class_weighted_loss': class_weighted_loss,
        'num_trainable_params': num_trainable,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'ssl_checkpoint': str(encoder_checkpoint_path),
    }

    # === TRAINING LOOP ===
    early_stop_patience = 10
    epochs_since_improvement = 0
    best_macro_f1 = 0.0
    global_step = 0
    
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")

        # === TRAINING ===
        encoder.train()
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        train_pbar = tqdm(train_loader, desc="Training")
        for x, y in train_pbar:
            x = x.to(device)  # (batch, 2, seq_length)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(not freeze_encoder):
                tokens = encoder(x)  # (batch, num_tokens, d_model)
                z = tokens.mean(dim=1)  # (batch, d_model) - global average pooling
            
            logits = classifier(z)  # (batch, num_classes)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in encoder.parameters() if p.requires_grad] + list(classifier.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            # Metrics
            batch_loss = loss.item()
            preds = logits.argmax(dim=1)
            batch_correct = (preds == y).sum().item()
            batch_size_actual = y.size(0)

            train_loss += batch_loss * batch_size_actual
            train_correct += batch_correct
            train_samples += batch_size_actual

            writer.add_scalar('Loss/train_batch', batch_loss, global_step)
            train_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
            global_step += 1
        
        avg_train_loss = train_loss / train_samples
        train_accuracy = train_correct / train_samples
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")

        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', train_accuracy, epoch)

        # === VALIDATION ===
        encoder.eval()
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                tokens = encoder(x)
                z = tokens.mean(dim=1)
                logits = classifier(z)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_samples += y.size(0)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        avg_val_loss = val_loss / val_samples
        val_accuracy = val_correct / val_samples
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Macro F1: {val_macro_f1:.4f}")
        print("Per-class F1:")
        for cls_name, f1 in zip(class_names, val_per_class_f1):
            print(f"  {cls_name}: {f1:.4f}")
            writer.add_scalar(f'F1/val_{cls_name}', f1, epoch)

        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val_epoch', val_accuracy, epoch)
        writer.add_scalar('F1/val_macro', val_macro_f1, epoch)

        current_lr_encoder = optimizer.param_groups[0]['lr'] if not freeze_encoder else 0.0
        current_lr_classifier = optimizer.param_groups[-1]['lr']
        writer.add_scalar('Learning_Rate/encoder', current_lr_encoder, epoch)
        writer.add_scalar('Learning_Rate/classifier', current_lr_classifier, epoch)

        scheduler.step(val_macro_f1)

        # === SAVE BEST MODEL ===
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            epochs_since_improvement = 0

            checkpoint_file = checkpoint_path / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': avg_val_loss,
                'val_macro_f1': val_macro_f1,
                'val_per_class_f1': val_per_class_f1.tolist(),
                'hyperparameters': encoder_hb_cfg,
                'preprocessing': preprocess_config.to_dict(),
                'training_config': training_config,
            }, checkpoint_file)
            print(f"✓ Saved best model (macro F1: {val_macro_f1:.4f})")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s).")

        if epochs_since_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered.")
            break

        # Save latest
        latest_checkpoint = checkpoint_path / "latest_model.pt"
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'val_loss': avg_val_loss,
            'val_macro_f1': val_macro_f1,
            'val_per_class_f1': val_per_class_f1.tolist(),
            'hyperparameters': encoder_hb_cfg,
            'preprocessing': preprocess_config.to_dict(),
            'training_config': training_config,
        }, latest_checkpoint)

    print(f"\n{'='*60}")
    print(f"Fine-Tuning Complete!")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"Checkpoints saved to: {checkpoint_path}")
    print(f"{'='*60}")

    writer.close()
    return encoder, classifier






if __name__ == "__main__":

    ### in main & in the two SSL training functions, things have to be adjusted when using the final encoder model!! ###
    ### for example, no max_token parameter but target_tokens, ... ###

    MODE = "pretrain"  # "pretrain" or "finetune"
    
    # Shared config
    CONFIG_NAME = "notch_bandpass_resample_znorm"
    USE_CACHE = True
    
    # SSL Pre-training settings (used if MODE == "pretrain")
    SSL_NUM_EPOCHS = 200
    SSL_BATCH_SIZE = 96
    SSL_LEARNING_RATE = 1e-4
    SSL_TEMPERATURE = 0.07
    SSL_LAMBDA_TOKEN = 1.0
    SSL_LAMBDA_GLOBAL = 1.0
    
    SSL_MODEL_CONFIG = {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.2,
        'target_tokens': 240   # when changing to SSLEpochTransformerConv1D_v2 this should be target_tokens = 240
    }
    
    # Fine-tuning settings (used if MODE == "finetune"), manually write the correct checkpoint name from the SSL pre-training run
    SSL_CHECKPOINT_NAME = "ssl_cross_modal_notch_bandpass_resample_znorm_v1"  # Name of SSL experiment
    
    FINETUNE_NUM_EPOCHS = 50
    FINETUNE_BATCH_SIZE = 128
    FINETUNE_LR_ENCODER = 1e-5      # 100× smaller than classifier
    FINETUNE_LR_CLASSIFIER = 1e-3
    FINETUNE_LABEL_FRACTION = 1.0   # fraction of labels to use during fine-tuning
    FINETUNE_FREEZE_ENCODER = False # False = one-stage, True = two-stage (first linear probe, then end-to-end)
    

    # Load preprocessing config
    config_file = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
    preproc_cfg = PreprocessingConfig.from_yaml(config_file)
    
    if MODE == "pretrain":
        print("\n" + "="*60)
        print("PHASE 1: SSL PRE-TRAINING")
        print("="*60)
        
        experiment_name = f"ssl_cross_modal_{CONFIG_NAME}_v1"
        
        model_hb, model_psg = train_ssl_epochtransformer(
            num_epochs=SSL_NUM_EPOCHS,
            batch_size=SSL_BATCH_SIZE,
            lr=SSL_LEARNING_RATE,
            experiment_name=experiment_name,
            model_kwargs=SSL_MODEL_CONFIG,
            preprocess_config=preproc_cfg,
            use_cache=USE_CACHE,
            temperature=SSL_TEMPERATURE,
            lambda_token=SSL_LAMBDA_TOKEN,
            lambda_global=SSL_LAMBDA_GLOBAL,
        )
        
        print(f"\n SSL pre-training complete!")
        print(f"  Checkpoint: {CHECKPOINT_DIR / experiment_name / 'best_model.pt'}")
        print(f"\nTo fine-tune, change MODE to 'finetune' and set:")
        print(f"  SSL_CHECKPOINT_NAME = '{experiment_name}'")
    
    elif MODE == "finetune":
        print("\n" + "="*60)
        print("PHASE 2: SUPERVISED FINE-TUNING")
        print("="*60)
        
        # Resolve SSL checkpoint path
        ssl_ckpt_path = CHECKPOINT_DIR / SSL_CHECKPOINT_NAME / "best_model.pt"
        
        if not ssl_ckpt_path.exists():
            raise FileNotFoundError(
                f"SSL checkpoint not found: {ssl_ckpt_path}\n"
                f"Make sure SSL_CHECKPOINT_NAME = '{SSL_CHECKPOINT_NAME}' is correct.\n"
                f"Available checkpoints:\n" + 
                "\n".join(f"  - {p.name}" for p in CHECKPOINT_DIR.iterdir() if p.is_dir())
            )
        
        # Generate experiment name
        label_pct = int(FINETUNE_LABEL_FRACTION * 100)
        freeze_str = "frozen" if FINETUNE_FREEZE_ENCODER else "endtoend"
        experiment_name = f"ssl_finetuned_{label_pct}pct_{freeze_str}_v1"
        
        print(f"\nLoading SSL checkpoint: {ssl_ckpt_path}")
        print(f"Label fraction: {FINETUNE_LABEL_FRACTION * 100:.0f}%")
        print(f"Freeze encoder: {FINETUNE_FREEZE_ENCODER}")
        
        encoder, classifier = fine_tune_ssl_encoder(
            encoder_checkpoint_path=ssl_ckpt_path,
            num_epochs=FINETUNE_NUM_EPOCHS,
            batch_size=FINETUNE_BATCH_SIZE,
            lr_encoder=FINETUNE_LR_ENCODER,
            lr_classifier=FINETUNE_LR_CLASSIFIER,
            experiment_name=experiment_name,
            preprocess_config=preproc_cfg,
            use_cache=USE_CACHE,
            class_weighted_loss=True,
            freeze_encoder=FINETUNE_FREEZE_ENCODER,
            label_fraction=FINETUNE_LABEL_FRACTION,
        )
        
        print(f"\n✓ Fine-tuning complete!")
        print(f"  Checkpoint: {CHECKPOINT_DIR / experiment_name / 'best_model.pt'}")
    
    else:
        raise ValueError(f"Invalid MODE: {MODE}. Must be 'pretrain' or 'finetune'.")
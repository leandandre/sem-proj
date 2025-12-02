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
from sem_proj.models.model_factory import SSLEpochTransformerConv1D, SSLClassifierHead

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
        num_workers=7,
        drop_last=True,  # Important for contrastive learning
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
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    experiment_name: str = "ssl_transformer_v1",
    model_kwargs: dict | None = None,
    preprocess_config: Optional[PreprocessingConfig] = None,
    use_cache: bool = True,
    use_conv1d: bool = True,  # Recommended for SSL
):
    checkpoint_path = CHECKPOINT_DIR / experiment_name
    log_path = LOG_DIR / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(log_path))

    if preprocess_config is None:
        preprocess_config = PreprocessingConfig.no_preprocessing()
    seq_length = get_expected_seq_length(preprocess_config)

    default_model_cfg = dict(
        input_channels=2,   # Will create separate models for headband (2) and PSG (6)
        seq_length=seq_length,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.2,
        num_classes=5,  # will be ignored in SSL
        max_tokens=512,
    )

    if model_kwargs:
        default_model_cfg.update(model_kwargs)

    assert use_conv1d, "Currently only Conv1D patch embedding is supported for SSL"

    # create two models: one for headband (2 channels), one for PSG (6 channels)
    model_hb_cfg = default_model_cfg.copy()
    model_hb_cfg['input_channels'] = 2  # Headband
    model_hb = SSLEpochTransformerConv1D(**model_hb_cfg).to(device)
    
    model_psg_cfg = default_model_cfg.copy()
    model_psg_cfg['input_channels'] = 6  # PSG
    model_psg = SSLEpochTransformerConv1D(**model_psg_cfg).to(device)

    num_params_hb = sum(p.numel() for p in model_hb.parameters() if p.requires_grad)
    num_params_psg = sum(p.numel() for p in model_psg.parameters() if p.requires_grad)
    print(f"Headband encoder has {num_params_hb:,} trainable parameters")
    print(f"PSG encoder has {num_params_psg:,} trainable parameters\n")

    train_loader, val_loader = make_dataloaders_ssl(
        batch_size=batch_size, 
        preprocess_config=preprocess_config,
        use_cache=use_cache
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")

    # weighting the  two losses (token level and global level)
    lambda_token = 1.0
    lambda_global = 1.0

    # Optimize both models
    optimizer = torch.optim.Adam(
        list(model_hb.parameters()) + list(model_psg.parameters()), 
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6
    )

    early_stop_patience = 12
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
            loss_token = token_contrastive_loss(tokens_hb, tokens_psg, temperature=0.07)
            loss_global = global_contrastive_loss(z_hb, z_psg, temperature=0.07)

            loss = lambda_token * loss_token + lambda_global * loss_global

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model_hb.parameters()) + list(model_psg.parameters()), 
                max_norm=1.0
            )
            optimizer.step()

            batch_loss = loss.item()
            batch_size_actual = x_hb.size(0)
            
            train_loss += batch_loss * batch_size_actual
            train_samples += batch_size_actual
            
            writer.add_scalar('Loss/train_batch', batch_loss, global_step)
            
            train_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
            global_step += 1
        
        avg_train_loss = train_loss / train_samples
        print(f"Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # === VALIDATION ===
        model_hb.eval()
        model_psg.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for x_hb, x_psg, y in val_loader:  # CHANGED: x_hb, x_psg, y
                x_hb = x_hb.to(device)
                x_psg = x_psg.to(device)
                
                out_hb = model_hb(x_hb)
                out_psg = model_psg(x_psg)
                
                loss = token_contrastive_loss(out_hb, out_psg, temperature=0.07)
                
                val_loss += loss.item() * x_hb.size(0)
                val_samples += x_hb.size(0)

        avg_val_loss = val_loss / val_samples
        print(f"Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        
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
                'hyperparameters_hb': model_hb_cfg,
                'hyperparameters_psg': model_psg_cfg,
                'preprocessing': preprocess_config.to_dict(),
            }, checkpoint_file)
            print(f"✓ Saved best model (val loss: {avg_val_loss:.4f})")
        else:
            epochs_since_improvement += 1
            print(f"No improvement in val loss for {epochs_since_improvement} epoch(s).")
        
        if epochs_since_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered.")
            break

        latest_checkpoint = checkpoint_path / "latest_model.pt"
        torch.save({
            'epoch': epoch,
            'encoder_hb_state_dict': model_hb.state_dict(),
            'encoder_psg_state_dict': model_psg.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'hyperparameters_hb': model_hb_cfg,
            'hyperparameters_psg': model_psg_cfg,
            'preprocessing': preprocess_config.to_dict(),
        }, latest_checkpoint)

    print(f"\n{'='*60}")
    print(f"SSL Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_path}")
    print(f"{'='*60}")
    
    writer.close()
    return model_hb, model_psg

def fine_tuning_classifier_head(
    encoder: nn.Module,
    classifier: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
):
    ### to implement###
    pass

def fine_tune_end_to_end(
    encoder: nn.Module,
    classifier: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    device: torch.device,
    lr_encoder: float = 1e-5,
    lr_classifier: float = 1e-4,
):
    ### to implement###
    pass


if __name__ == "__main__":
    CONFIG_NAME = "notch_bandpass_resample_znorm"
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    USE_CACHE = True
    
    config_file = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
    preproc_cfg = PreprocessingConfig.from_yaml(config_file)
    
    model_cfg = {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 512,
        'dropout': 0.2,
        'max_tokens': 512
    }
    
    experiment_name = f"ssl_cross_modal_{CONFIG_NAME}_v1"
    
    model_hb, model_psg = train_ssl_epochtransformer(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        experiment_name=experiment_name,
        model_kwargs=model_cfg,
        preprocess_config=preproc_cfg,
        use_cache=USE_CACHE,
        use_conv1d=True,
)
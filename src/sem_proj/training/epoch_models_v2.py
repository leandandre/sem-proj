"""
Fine-tuning a context-free classifier head on top of a pretrained SSL epoch encoder.

This module parallels sequence_models_variable.py but replaces the GRU with a
simple per-epoch classifier head (SSLClassifierHead). Use when you want to
fine-tune only on individual epochs without sequence context.
"""
from __future__ import annotations

from json import encoder
from json import encoder
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import os
import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from sem_proj.data.datasets import BoasDataset
from sem_proj.data.preprocessing import PreprocessingConfig, get_expected_seq_length
from sem_proj.data.splits import get_train_subjects, get_val_subjects
from sem_proj.models.model_factory import SSLEpochTransformerConv1D_v2, SSLClassifierHead, SSLLinearProbing
from sem_proj.data.transforms import RandomTimeShift, RandomAmplitudeScale, RandomGaussianNoise, Compose

# Seeds
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"


def make_epoch_dataloaders(
    batch_size: int = 32,
    preprocess_config: Optional[PreprocessingConfig] = None,
    use_cache: bool = True,
    use_augmentation: bool = False,
    mode: str = "headband",
    train_subjects: Optional[list] = None,
    val_subjects: Optional[list] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for per-epoch classification (context-free).

    Parameters
    ----------
    batch_size : int
        Batch size for dataloaders.
    preprocess_config : PreprocessingConfig, optional
        Preprocessing configuration.
    use_cache : bool
        Whether to use cached data.
    use_augmentation : bool
        Whether to apply data augmentation (headband only).
    mode : str
        "headband" or "psg".
    train_subjects, val_subjects : list, optional
        Override default subject splits.
    """
    tr_subs = train_subjects if train_subjects is not None else get_train_subjects()
    val_subs = val_subjects if val_subjects is not None else get_val_subjects()

    train_transform = None
    if use_augmentation:
        train_transform = Compose([
            RandomTimeShift(max_shift_ratio=0.10),      # +-% time shift
            RandomAmplitudeScale(scale_range=(0.9, 1.1)),  # +-% amplitude
            RandomGaussianNoise(noise_scale=(0.01, 0.05)),   # Add Gaussian noise
        ])

    train_ds = BoasDataset(
        subjects=tr_subs,
        mode=mode,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        transform_hb=train_transform if mode == "headband" else None,
    )

    val_ds = BoasDataset(
        subjects=val_subs,
        mode=mode,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        transform_hb=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=7,      # 2 for local, 7 for cluster
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,   # set to True if num_workers > 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=7,      # 2 for local, 7 for cluster
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,   # set to True if num_workers > 0
    )

    return train_loader, val_loader


def compute_class_weights(dataloader: DataLoader, num_classes: int = 5) -> torch.Tensor:
    """Inverse-frequency class weights with square-root smoothing."""
    all_labels = []
    for _, y in dataloader:
        all_labels.extend(y.cpu().numpy().tolist())

    counter = Counter(all_labels)
    total = len(all_labels)
    weights = []
    for cls in range(num_classes):
        count = counter.get(cls, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    weights = np.sqrt(np.array(weights, dtype=np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_epoch_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray]:
    """Evaluate per-epoch classifier."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return avg_loss, accuracy, macro_f1, per_class_f1

def train_contextfree_classifierhead(
    num_epochs: int = 50,
    batch_size: int = 32,
    lr_encoder: float = 1e-5,
    lr_head: float = 1e-3,
    mode: str = "headband",
    experiment_name: str = "ctxfree_classifierhead_v1",
    preprocess_config: Optional[PreprocessingConfig] = None,
    use_cache: bool = True,
    ssl_checkpoint: Optional[Path] = None,
    freeze_encoder: bool = False,
    d_model: int = 128,
    nhead: int = 4,
    num_layers_encoder: int = 2,
    dim_feedforward: int = 512,
    dropout_encoder: float = 0.2,
    dropout_head: float = 0.2,
    target_tokens: int = 240,
    class_weighted_loss: bool = True,
    gradient_clip: float = 5.0,
    early_stopping_patience: int = 10,
    num_classes: int = 5,
    train_subjects: Optional[list] = None,
    val_subjects: Optional[list] = None,
) -> Tuple[float, float, np.ndarray]:
    """Fine-tune SSL classifier head (context-free, per-epoch).
    IS ALSO USED FOR THE FULLY-SUPERVISED TRAINING TO COMPARE WITH THE SSL FINE-TUNING RESULTS.
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

    # Determine training mode
    fully_supervised = ssl_checkpoint is None
    
    print(f"\n{'='*70}")
    if fully_supervised:
        print("Context-Free Classifier Head Training (Fully Supervised)")
    else:
        print("Context-Free Classifier Head Fine-Tuning (SSL Pretrained)")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Training mode: {'Fully supervised (random init)' if fully_supervised else 'Fine-tuning from SSL'}")
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Lr encoder: {lr_encoder}, lr head: {lr_head}")
    print(f"Preprocess: {preprocess_config.to_dict()}")
    print(f"SSL checkpoint: {ssl_checkpoint}")
    print(f"{'='*70}\n")

    # Dataloaders
    train_loader, val_loader = make_epoch_dataloaders(
        batch_size=batch_size,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        use_augmentation=True,      # use augmentation for training
        mode=mode,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
    )
    print(f"Training epochs: {len(train_loader.dataset)}")
    print(f"Validation epochs: {len(val_loader.dataset)}\n")

    # Build encoder + head
    if fully_supervised:
        # Random init encoder for fully-supervised training
        encoder = SSLEpochTransformerConv1D_v2(
            input_channels=2 if mode == "headband" else 6,
            seq_length=seq_length,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers_encoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout_encoder,
            num_classes=num_classes,
            target_tokens=target_tokens,
        )
    else:
        # Load pretrained SSL encoder
        # encoder = _load_ssl_encoder(
        #     ssl_checkpoint,
        #     mode=mode,
        #     seq_length=seq_length,
        #     d_model=d_model,
        #     nhead=nhead,
        #     num_layers=num_layers_encoder,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout_encoder,
        #     target_tokens=target_tokens,
        # )
        checkpoint_from_ssl = torch.load(ssl_checkpoint, map_location='cpu')
        encoder_hb_cfg = checkpoint_from_ssl['hyperparameters_hb']      # number of layers, d_model, layers, etc.
        encoder = SSLEpochTransformerConv1D_v2(**encoder_hb_cfg).to(device)
        encoder.load_state_dict(checkpoint_from_ssl['encoder_hb_state_dict'])
        print(f"Loaded SSL encoder from {ssl_checkpoint}")
        print(f"Encoder config from checkpoint: {encoder_hb_cfg}")
        
    head = SSLClassifierHead(d_model=encoder.d_model, dropout=dropout_head, num_classes=num_classes)
    print("-!-!-! Non-linear SSL Classifier Head in use -!-!-!")

    # head = SSLLinearProbing(d_model=encoder.d_model, num_classes=num_classes)
    # print("-!-!-! LINEAR PROBING in use -!-!-!")

    model = nn.Sequential(encoder, head).to(device)

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    # Loss
    if class_weighted_loss:
        print("Computing class weights from training data...")
        class_weights = compute_class_weights(train_loader, num_classes=num_classes)
        print(f"Class weights (sqrt smoothing): {class_weights.numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer with param groups
    if freeze_encoder:
        params = [dict(params=head.parameters(), lr=lr_head)]
    else:
        params = [
            dict(params=encoder.parameters(), lr=lr_encoder),
            dict(params=head.parameters(), lr=lr_head),
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.25, patience=6
    )

    scaler = GradScaler()

    best_macro_f1 = 0.0
    fin_acc = 0.0
    fin_per_class_f1 = np.zeros(num_classes)
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.numel()

            pbar.set_postfix(loss=loss.item())

        train_avg_loss = train_loss / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Validation
        val_loss, val_acc, val_macro_f1, val_per_class_f1 = evaluate_epoch_classifier(
            model, val_loader, criterion, device
        )

        print(f"Train loss: {train_avg_loss:.4f} | acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | acc: {val_acc:.4f} | macro F1: {val_macro_f1:.4f}")
        print(f"Val per-class F1: {np.round(val_per_class_f1, 4)}")

        writer.add_scalar("loss/train", train_avg_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("f1/val_macro", val_macro_f1, epoch)

        scheduler.step(val_macro_f1)

        # Early stopping on macro F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            fin_acc = val_acc
            fin_per_class_f1 = val_per_class_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "hyperparameters": {
                        "mode": mode,
                        "seq_length": seq_length,
                        "d_model": encoder.d_model,
                        "nhead": nhead if fully_supervised else encoder_hb_cfg['nhead'],    # fully_supervised -> we determine cfg before running the script, ...
                        "num_layers": num_layers_encoder if fully_supervised else encoder_hb_cfg['num_layers'],                                   # ...else use loaded cfg (pretrained model)
                        "dim_feedforward": dim_feedforward if fully_supervised else encoder_hb_cfg['dim_feedforward'],
                        "dropout_encoder": dropout_encoder if fully_supervised else encoder_hb_cfg['dropout'],
                        "dropout_head": dropout_head,
                        "target_tokens": target_tokens if fully_supervised else encoder_hb_cfg['target_tokens'],
                        "num_classes": num_classes,
                    },
                    "training_config": {
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "lr_encoder": lr_encoder,
                        "lr_head": lr_head,
                        "freeze_encoder": freeze_encoder,
                        "class_weighted_loss": class_weighted_loss,
                        "gradient_clip": gradient_clip,
                        "use_cache": use_cache,
                    },
                },
                checkpoint_path / "best_model.pt",
            )
            print("Saved new best model (macro F1)")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break

    writer.close()
    print(f"Best validation macro F1: {best_macro_f1:.4f}")
    print(f"Final validation accuracy: {fin_acc:.4f}")
    print(f"Final validation per-class F1: {np.round(fin_per_class_f1, 4)}")
    print(f"Checkpoints saved to: {checkpoint_path}")
    return best_macro_f1, fin_acc, fin_per_class_f1


__all__ = [
    "make_epoch_dataloaders",
    "train_contextfree_classifierhead",
    "evaluate_epoch_classifier",
]

"""
Fine-tune a context-sensitive sequence model using the pretrained encoder for epoch-represetation and a GRU-based classifier head on top of L epochs.
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple
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

from sem_proj.data.datasets import BoasSequenceDataset
from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.models.model_factory import SSLEpochTransformerConv1D_v2, SequenceGRUClassifier
from sem_proj.data.transforms import (
    Compose,
    RandomTimeShift,
    RandomAmplitudeScale,
    RandomGaussianNoise,
)

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

def make_sequence_dataloaders(
        batch_size: int = 16,
        seq_length: int = 20,
        stride: int = 5,
        mode: str = "headband",
        train_subjects: Optional[List[str]] = None,
        val_subjects: Optional[List[str]] = None,
        preprocess_config: Optional[PreprocessingConfig] = None,
        use_cache: bool = True,
        use_augmentation: bool = True,
):
    """
    Create sequence dataloaders for SequenceGRUClassifier.
    
    Parameters
    ----------
    batch_size : int
        Batch size.
    seq_length : int
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
    train_transform = None
    if use_augmentation:
        train_transform = Compose([
            RandomTimeShift(max_shift_ratio=0.10),      # +-% time shift
            RandomAmplitudeScale(scale_range=(0.9, 1.1)),  # +-% amplitude
            RandomGaussianNoise(noise_scale=(0.01, 0.05)),   # Add Gaussian noise
        ])
    train_ds = BoasSequenceDataset(
        subjects=train_subjects,
        mode=mode,
        seq_len=seq_length,
        stride=stride,
        transform_hb=train_transform,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
    )
    val_ds = BoasSequenceDataset(
        subjects=val_subjects,
        mode=mode,
        seq_len=seq_length,
        stride=stride,
        transform_hb=None,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    # will return loaders with data in shape (B, L, C, T)
    return train_loader, val_loader

def compute_class_weights(dataloader: DataLoader, num_classes: int) -> torch.Tensor:
    """Inverse-frequence class weights with square-root smoothing."""
    all_labels = []
    for _, Y in dataloader: # Y is shape (B, L)
        y = Y.flatten().cpu().numpy().tolist() # flatten to 1D
        all_labels.extend(y)
    counter = Counter(all_labels)
    total = len(all_labels)
    weights = []
    for cls in range(num_classes):
        count = counter.get(cls, 0)
        weight = total / (num_classes * (count + 1e-6))  # avoid div by zero
        weights.append(weight)
    weights = np.sqrt(np.array(weights, dtype=np.float32))  # sqrt smoothing
    return torch.tensor(weights, dtype=torch.float32)

def evaluate_sequence_classifier(model, dataloader, criterion, device) -> Tuple[float, float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)  # (B, L, C, T)
            y = y.to(device)  # (B, L)

            output = model(x)  # (B, L, num_classes)
            B, L, K = output.shape
            loss = criterion(
                output.view(B * L, K),
                y.view(B * L)
            )

            total_loss += loss.item() * B

            preds = output.argmax(dim=-1)  # (B, L)
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()

            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(y.cpu().numpy().flatten())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return avg_loss, accuracy, macro_f1, per_class_f1

def train_contextsensitive_classifier(
        num_epochs: int = 100,
        batch_size: int = 16,
        seq_len: int = 20,
        stride: int = 5,
        lr_encoder: float = 1e-5,
        lr_gru: float = 1e-4,
        mode: str = "headband",
        experiment_name: str = "seqmodel_finetuning",
        preprocess_config: Optional[PreprocessingConfig] = None,
        use_cache: bool = True,
        ssl_checkpoint: Optional[Path] = None,
        freeze_encoder: bool = False,
        d_model: int = 128,
        nhead: int = 4,
        num_layers_encoder: int = 2,
        dim_feedforward: int = 512,
        dropout_encoder: float = 0.2,
        dropout_gru: float = 0.2,
        target_tokens: int = 240,
        class_weighted_loss: bool = True,
        gradient_clip: float = 5.0,
        early_stopping_patience: int = 12,
        num_classes: int = 5,
        train_subjects: Optional[List[str]] = None,
        val_subjects: Optional[List[str]] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Fine-tune a context-sensitive sequence model (GRU) using a pretrained encoder (epoch-embeddings).
    Returns the best validation MF1 score achieved.
    """
    checkpoint_path = CHECKPOINT_DIR / experiment_name
    log_path = LOG_DIR / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(log_path))

    # default preprocess config
    if preprocess_config is None:
        preprocess_config = PreprocessingConfig.from_yaml(
            PROJECT_ROOT / "configs" / "preprocess" / "notch_bandpass_resample_znorm.yaml"
        )
    
    fully_supervised = ssl_checkpoint is None
    print(f"\n{'='*70}")
    if fully_supervised:
        print("Context-Sensitive Model Training (Fully Supervised)")
    else:
        print("Context-Sensitive Model Fine-Tuning (SSL Pretrained)")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Training mode: {'Fully supervised (random init)' if fully_supervised else 'Fine-tuning from SSL'}")
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Lr encoder: {lr_encoder}, lr GRU: {lr_gru}")
    print(f"Preprocess: {preprocess_config.to_dict()}")
    print(f"SSL checkpoint: {ssl_checkpoint}")
    print(f"{'='*70}\n")

    train_loader, val_loader = make_sequence_dataloaders(
        batch_size=batch_size,
        seq_length=seq_len,
        stride=stride,
        mode=mode,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        preprocess_config=preprocess_config,
        use_cache=use_cache,
        use_augmentation=True,
    )
    print(f"Train sequences: {len(train_loader.dataset)} | Val sequences: {len(val_loader.dataset)}")

    if fully_supervised:
        encoder = SSLEpochTransformerConv1D_v2(
            input_channels=2,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers_encoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout_encoder,
            num_classes=num_classes,
            target_tokens=target_tokens,
        )
    else:
        checkpoint_from_ssl = torch.load(ssl_checkpoint, map_location="cpu")
        encoder_hb_cfg = checkpoint_from_ssl['hyperparameters_hb']
        encoder = SSLEpochTransformerConv1D_v2(**encoder_hb_cfg).to(device)
        encoder.load_state_dict(checkpoint_from_ssl['encoder_hb_state_dict'])
        print(f"Loaded SSL encoder from {ssl_checkpoint}")
        print(f"Encoder config from checkpoint: {encoder_hb_cfg}")

    context_model = SequenceGRUClassifier(
        epoch_model=encoder,
        hidden_size=128,
        num_layers=1,
        num_classes=num_classes,
        bidirectional=True,
    ).to(device)

    if class_weighted_loss:
        print("Computing class weights from training data...")
        class_weights = compute_class_weights(train_loader, num_classes)
        print(f"Class weights (srt smoothing): {class_weights.numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    encoder_params = list(context_model.epoch_model.parameters())
    rest_params = [
        p for name, p in context_model.named_parameters()
        if not name.startswith("epoch_model.")
    ]

    if freeze_encoder:
        for param in context_model.epoch_model.parameters():
            param.requires_grad = False
        print("Encoder frozen. Only training context model parameters.")
        params = [dict(params=rest_params, lr=lr_gru)]
    else:
        print("Training happens in whole model (encoder + context model).")
        params = [
            dict(params=encoder_params, lr=lr_encoder),
            dict(params=rest_params, lr=lr_gru),
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.25, patience=6
    )

    best_macro_f1 = 0.0
    fin_acc = 0.0
    fin_per_class_f1 = np.zeros(num_classes)
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)

        context_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for x, y in pbar:
            x = x.to(device)  # (B, L, C, T)
            y = y.to(device)  # (B, L)

            optimizer.zero_grad()
            output = context_model(x)  # (B, L, num_classes)
            B, L, K = output.shape
            loss = criterion(
                output.view(B * L, K), 
                y.view(B * L)
            )

            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(context_model.parameters(), gradient_clip)
            optimizer.step()

            batch_loss = loss.item()
            preds = output.argmax(dim=-1)  # (B, L)
            batch_correct = (preds == y).sum().item()
            batch_samples = y.numel()

            train_loss += batch_loss * B
            train_correct += batch_correct
            train_total += batch_samples

            batch_acc = batch_correct / batch_samples
            pbar.set_postfix({'Loss': f"{batch_loss:.4f}", 'Acc': f"{batch_acc:.4f}"})

        train_avg_loss = train_loss / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        val_loss, val_acc, val_macro_f1, val_per_class_f1 = evaluate_sequence_classifier(
            context_model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f}")
        print(f"Val Per-Class F1: {np.round(val_per_class_f1, 4)}")

        writer.add_scalar("loss/train", train_avg_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("f1/val_macro", val_macro_f1, epoch)
        for i, f1_score in enumerate(val_per_class_f1):
            writer.add_scalar(f"f1/val_class_{i}", f1_score, epoch)
        
        scheduler.step(val_macro_f1)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            fin_acc = val_acc
            fin_per_class_f1 = val_per_class_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": context_model.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "gru_state_dict": {
                        "gru": context_model.gru.state_dict(),
                        "classifier": context_model.classifier.state_dict(),
                    },
                    "hyperparameters": {
                        "mode": mode,
                        "d_model": encoder.d_model,
                        "nhead": nhead if fully_supervised else encoder_hb_cfg['nhead'],    # fully_supervised -> we determine cfg before running the script, ...
                        "num_layers": num_layers_encoder if fully_supervised else encoder_hb_cfg['num_layers'],                                   # ...else use loaded cfg (pretrained model)
                        "dim_feedforward": dim_feedforward if fully_supervised else encoder_hb_cfg['dim_feedforward'],
                        "dropout_encoder": dropout_encoder if fully_supervised else encoder_hb_cfg['dropout'],
                        "dropout_gru": dropout_gru,
                        "target_tokens": target_tokens if fully_supervised else encoder_hb_cfg['target_tokens'],
                        "num_classes": num_classes,
                    },
                    "training_config": {
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "lr_encoder": lr_encoder,
                        "lr_gru": lr_gru,
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
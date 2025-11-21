import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheeg.models import Conformer

from sem_proj.data.datasets import BoasSequenceDataset
from sem_proj.data.boa_loader import split_by_pid
from sem_proj.models.conformer_gru import ConformerFeatureExtractor, ConformerGRUClassifier

# def set_seed(seed: int):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

def make_seq_dataloaders(
        batch_size: int = 16,
        seed: int = 42,
        mode: str = "headband",
        seq_len: int = 20,
        stride: int = 5,
        transform_hb=None,
        transform_psg=None,
        target_transform=None,
        add_channel_dim: bool = True):

    sub_splits = split_by_pid(seed=seed)

    train_ds = BoasSequenceDataset(
        subjects=sub_splits["train_subjects"],
        mode=mode,
        seq_len=seq_len,
        stride=stride,
        transform_hb=None,
        transform_psg=None,
        target_transform=None,
        add_channel_dim=add_channel_dim
    )
    val_ds = BoasSequenceDataset(
        subjects=sub_splits["val_subjects"],
        mode=mode,
        seq_len=seq_len,
        stride=stride,
        transform_hb=None,
        transform_psg=None,
        target_transform=None,
        add_channel_dim=add_channel_dim
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

def train_conformer_gru_model(
    batch_size: int = 8,
    seq_len: int = 20,
    stride: int = 5,
    mode: str = "headband",
    num_classes: int = 5,
    num_epochs: int = 20,
    lr: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_seq_dataloaders(
        batch_size=batch_size,
        seed=42,
        mode=mode,
        seq_len=seq_len,
        stride=stride,
        transform_hb=None,
        transform_psg=None,
        target_transform=None,
        add_channel_dim=True,  # needed for Conformer
    )

    # Take one batch to infer num_electrodes and T_samples
    example_x, example_y = next(iter(train_loader))
    # example_x: (B, seq_len, 1, C, T)
    B, L, _, C, T_samples = example_x.shape
    print(f"Example batch shape: B={B}, L={L}, C={C}, T={T_samples}")
    assert (T_samples // 30) == 256, "Expected 30 seconds at 256 Hz sampling rate"

    emb_size = 40   # must equal hid_channels
    depth = 6
    heads = 10
    dropout = 0.5
    forward_expansion = 4
    forward_dropout = 0.5

    base_conformer = Conformer(
        num_electrodes=C,
        sampling_rate=256, 
        hid_channels=emb_size,
        depth=depth,
        heads=heads,
        dropout=dropout,
        forward_expansion=forward_expansion,
        forward_dropout=forward_dropout,
        num_classes=num_classes,   # not really used, we bypass cls head
    ).to(device)

    backbone = ConformerFeatureExtractor(
        base_conformer,
        pool="mean",  # (B, N, D) -> (B, D) for GRU
    ).to(device)

    gru_hidden = 128
    gru_layers = 1

    model = ConformerGRUClassifier(
        backbone=backbone,
        emb_dim=emb_size,
        gru_hidden=gru_hidden,
        num_layers=gru_layers,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_count = 0

        for x_seq, y_seq in train_loader:
            # x_seq: (B, L, 1, C, T)
            # y_seq: (B, L)
            x_seq = x_seq.to(device, dtype=torch.float32)
            y_seq = y_seq.to(device, dtype=torch.long)

            optimizer.zero_grad()

            logits = model(x_seq)  # (B, L, num_classes)
            B_cur, L_cur, K = logits.shape

            loss = F.cross_entropy(
                logits.view(B_cur * L_cur, K),
                y_seq.view(B_cur * L_cur),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * B_cur
            train_count += B_cur
        
        avg_train_loss = train_loss / train_count
        print(f"[Epoch {epoch+1}/{num_epochs}] train loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_count = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_seq, y_seq in val_loader:
                x_seq = x_seq.to(device, dtype=torch.float32)
                y_seq = y_seq.to(device, dtype=torch.long)

                logits = model(x_seq)  # (B, L, K)
                B_cur, L_cur, K = logits.shape

                loss = F.cross_entropy(
                    logits.view(B_cur * L_cur, K),
                    y_seq.view(B_cur * L_cur),
                )
                val_loss += loss.item() * B_cur
                val_count += B_cur

                preds = logits.argmax(dim=-1)  # (B, L)
                correct += (preds == y_seq).sum().item()
                total += y_seq.numel()

        avg_val_loss = val_loss / val_count
        val_acc = correct / total if total > 0 else 0.0
        print(
            f"           val loss: {avg_val_loss:.4f}, "
            f"val acc: {val_acc:.3f}"
        )
    return model
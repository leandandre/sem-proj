import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sem_proj.data.boa_loader import split_by_pid
from sem_proj.data.datasets import BoasDataset

def make_dataloaders(batch_size: int = 16):
    splits = split_by_pid(seed=42)

    train_ds = BoasDataset(subjects=splits["train_subjects"], mode="headband")
    val_ds = BoasDataset(subjects=splits["val_subjects"], mode="headband")

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

class TinyEEGNet(nn.Module):
    def __init__(self, in_channels: int = 2, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

def train_one_epoch(model, loader, optimizer, device, epoch: int):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    loop = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)          # [B, 5]
        loss = loss_fn(logits, y)  # y: [B]
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        # show current batch loss
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, device, epoch: int):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0

    loop = tqdm(loader, desc=f"Val   {epoch}", leave=False)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

def main():
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader = make_dataloaders(batch_size=32)

    model = TinyEEGNet(in_channels=2, n_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):  # or more
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device, epoch)
        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

if __name__ == "__main__":
    main()

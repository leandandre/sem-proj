"""
Final test set evaluation.
ONLY RUN THIS ONCE at the end of the project!
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sem_proj.data.splits import get_test_subjects
from sem_proj.data.datasets import BoasDataset
from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.models.model_factory import EpochTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


def evaluate_on_test_set(
    checkpoint_path: Path,
    preprocess_config: PreprocessingConfig,
    batch_size: int = 16,
):
    """
    Evaluate trained model on test set.
    
    WARNING: Only run this ONCE at the very end!
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model
    model_cfg = checkpoint['hyperparameters']
    model = EpochTransformer(**model_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation metrics: acc={checkpoint['val_accuracy']:.4f}, macro_f1={checkpoint['val_macro_f1']:.4f}")
    
    # Create test dataloader
    test_subjects = get_test_subjects()
    print(f"\nWARNING: Evaluating on {len(test_subjects)} TEST subjects")
    print("WARNING: This should only be done ONCE at the end!\n")
    
    test_ds = BoasDataset(
        subjects=test_subjects,
        mode="headband",
        preprocess_config=preprocess_config
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    print("Running inference on test set...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    test_loss = total_loss / len(all_labels)
    test_acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    # Print results
    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\nPer-class F1 scores:")
    for name, f1 in zip(class_names, per_class_f1):
        print(f"  {name}: {f1:.4f}")
    print("="*60)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Test Set Confusion Matrix\nAccuracy: {test_acc:.4f}, Macro F1: {macro_f1:.4f}')
    plt.tight_layout()
    
    cm_path = RESULTS_DIR / "test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix to {cm_path}")
    
    # Save results to file
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'macro_f1': float(macro_f1),
        'per_class_f1': {name: float(f1) for name, f1 in zip(class_names, per_class_f1)},
        'checkpoint_path': str(checkpoint_path),
        'n_test_samples': len(all_labels),
    }
    
    import json
    results_path = RESULTS_DIR / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {results_path}")
    
    return results


if __name__ == "__main__":
    # Example: Evaluate best model from training
    experiment_name = "epoch_transformer_v1"
    checkpoint = CHECKPOINT_DIR / experiment_name / "best_model.pt"
    
    preproc_cfg = PreprocessingConfig(
        notch_freqs=[50.0, 100.0],
        bandpass_l_freq=0.5,
        bandpass_h_freq=40.0,
        resample_freq=128.0,
        apply_preprocessing=True,
    )
    
    results = evaluate_on_test_set(
        checkpoint_path=checkpoint,
        preprocess_config=preproc_cfg,
        batch_size=16,
    )
"""
Fine-tune a context-sensitive model using a pretrained encoder, controlling the fraction of training labels used.
This script supports a two-stage process:
- Stage 1: Hyperparameter tuning on a fraction p of the training data, validated on the full validation set.
- Stage 2: Final evaluation training on a fraction p of the combined training and validation data, tested on the full test set.
This script is similar to finetune_contextfree_classifierhead.py but adapted for context-sensitive models.
--> also here: used this script to compare linear probe vs. fine-tuned vs. fully-supervised, depending on proportion p of labeled data used.
"""
import sys
import json
from pathlib import Path
import random
from typing import List
from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.training.sequence_models_v2 import train_contextsensitive_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "preprocess"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_LEOMED_DIR = PROJECT_ROOT / "checkpoints_leomed"

SPLITS_FILE = PROJECT_ROOT / "data" / "processed" / "data_splits_70_15_15.json"

def sample_subjects(subjects: List[str], fraction: float, seed: int = 42) -> List[str]:
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be between 0.0 and 1.0, got {fraction}")
    if fraction == 1.0:
        return sorted(subjects)
    rng = random.Random(seed)
    n_sample = max(1, int(len(subjects) * fraction))
    return sorted(rng.sample(subjects, n_sample))

def run_stage1(
    fraction: float,
    seed: int,
    ssl_checkpoint: Path | None,
    num_epochs: int,
    batch_size: int,
    seq_length: int,
    stride: int,
    lr_encoder: float,
    freeze_encoder_flag: bool,
    lr_gru: float,
    experiment_name: str,
):
    fully_supervised = ssl_checkpoint is None
    mode_str = "Fully-Supervised Training" if fully_supervised else "SSL Fine-Tuning"
    
    print("\n" + "=" * 80)
    print(f"STAGE 1: Hyperparameter Tuning ({mode_str})")
    print("=" * 80)
    print(f"Train on {fraction*100:.0f}% of train_subjects; validate on full val_subjects")
    print("=" * 80 + "\n")

    assert SPLITS_FILE.exists(), f"Splits file not found: {SPLITS_FILE}"
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)

    train_subjects = splits['train_subjects']
    val_subjects = splits['val_subjects']
    sampled_train = sample_subjects(train_subjects, fraction, seed)

    print("Subject allocation:")
    print(f"  Total train: {len(train_subjects)} | sampled: {len(sampled_train)}")
    print(f"  Validation: {len(val_subjects)}")
    print(f"  Seed: {seed}\n")

    config = PreprocessingConfig.from_yaml(
        CONFIG_DIR / "notch_bandpass_resample_znorm.yaml"
    )

    best_mf1 = train_contextsensitive_classifier(
        num_epochs=num_epochs,
        batch_size=batch_size,
        seq_len=seq_length,
        stride=stride,
        lr_encoder=lr_encoder,
        lr_gru=lr_gru,
        mode="headband",
        experiment_name=experiment_name,
        preprocess_config=config,
        use_cache=True,
        ssl_checkpoint=ssl_checkpoint,
        freeze_encoder=freeze_encoder_flag,
        d_model=128,
        nhead=4,
        num_layers_encoder=2,
        dim_feedforward=512,
        dropout_encoder=0.2,
        dropout_gru=0.2,
        target_tokens=240,
        class_weighted_loss=True,
        gradient_clip=5.0,
        early_stopping_patience=12,
        num_classes=5,
        train_subjects=sampled_train,
        val_subjects=val_subjects,
    )

    print("\n" + "=" * 80)
    print(f"STAGE 1 COMPLETE: Best Validation Macro F1: {best_mf1:.4f}")
    print("=" * 80 + "\n")
    return best_mf1

def run_stage2(
    fraction: float,
    seed: int,
    ssl_checkpoint: Path | None,
    num_epochs: int,
    batch_size: int,
    seq_length: int,
    stride: int,
    lr_encoder: float,
    freeze_encoder_flag: bool,
    lr_gru: float,
    experiment_name: str,
):
    pass


def main():
    # Fraction of labeled data to use (e.g., 0.1 = 10%, 1.0 = 100%)
    FRACTION = 0.5
    SEED = 42
    SSL_CHECKPOINT = None

    # Choose training mode
    # Option 1: FINE-TUNING (set to your SSL checkpoint path)
    ### when running on laptop:
    # SSL_CHECKPOINT = CHECKPOINT_LEOMED_DIR / "ssl_cross_modal_notch_bandpass_resample_znorm_v1" / "best_model.pt"
    ### else when running on cluster:
    SSL_CHECKPOINT = CHECKPOINT_DIR / "ssl_cross_modal_notch_bandpass_resample_znorm_v1" / "best_model.pt"
    
    # Option 2: FULLY-SUPERVISED END-TO-END (set to None)
    # SSL_CHECKPOINT = None
    
    NUM_EPOCHS = 200
    BATCH_SIZE = 512
    SEQ_LENGTH = 20
    STRIDE = 5
    
    if SSL_CHECKPOINT is None:
        # Fully-supervised: use same LR for encoder and GRU
        LR_ENCODER = 1e-4
        LR_GRU = 1e-4
        FREEZE_ENCODER = False
    else:
        # Fine-tuning: smaller encoder LR, larger GRU LR, set freeze flag to train GRU only
        LR_ENCODER = 1e-5
        FREEZE_ENCODER = False
        LR_GRU = 1e-4

    mode_str = "Fully-Supervised" if SSL_CHECKPOINT is None else "SSL-FineTuning"
    
    print("\n" + "=" * 80)
    print(f"Context-Sensitive Model Training - {mode_str}")
    print("=" * 80)
    print(f"Fraction: {FRACTION*100:.0f}% | Seed: {SEED}")
    print(f"SSL checkpoint: {SSL_CHECKPOINT}")
    print(f"Epochs: {NUM_EPOCHS} | Batch size: {BATCH_SIZE}")
    print(f"Seq length: {SEQ_LENGTH} | Stride: {STRIDE}")
    print(f"LR encoder: {LR_ENCODER} | LR GRU: {LR_GRU}")
    print("=" * 80 + "\n")

    stage1_mf1 = run_stage1(
        fraction=FRACTION,
        seed=SEED,
        ssl_checkpoint=SSL_CHECKPOINT,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        seq_length=SEQ_LENGTH,
        stride=STRIDE,
        lr_encoder=LR_ENCODER,
        freeze_encoder_flag=FREEZE_ENCODER,
        lr_gru=LR_GRU,
        experiment_name=f"ctxsensitive_stage1_p{FRACTION}_{mode_str.lower().replace('-', '_')}",
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Mode: {mode_str}")
    print(f"Fraction: {FRACTION*100:.0f}%")
    print(f"Stage 1 best Val F1: {stage1_mf1:.4f}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
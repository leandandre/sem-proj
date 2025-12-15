"""
Fine-tuning script for context-free SSL classifier head (per-epoch) with
controlled labeled data fractions.
SOLELY FOR FINE-TUNING (FOR FULLY-SUPERVISED END-TO-END RUN epoch_models.py)

Two-stage workflow (mirrors finetune_variable_gru.py):

Stage 1: Hyperparameter Tuning
- Train on fraction p of train_subjects
- Validate on full val_subjects

Stage 2: Final Evaluation
- Train on fraction p of (train_subjects + val_subjects)
- Test on full test_subjects
"""
import sys
from pathlib import Path
import random
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.data.splits import get_train_subjects, get_val_subjects, get_test_subjects
from sem_proj.training.epoch_models_v2 import train_contextfree_classifierhead

CONFIG_DIR = PROJECT_ROOT / "configs" / "preprocess"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


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
    ssl_checkpoint: Path,
    num_epochs: int,
    batch_size: int,
    lr_encoder: float,
    lr_head: float,
    experiment_name: str,
):
    print("\n" + "=" * 80)
    print("STAGE 1: Hyperparameter Tuning (context-free head)")
    print("=" * 80)
    print(f"Train on {fraction*100:.0f}% of train_subjects; validate on full val_subjects")
    print("=" * 80 + "\n")

    train_subjects = get_train_subjects()
    val_subjects = get_val_subjects()
    sampled_train = sample_subjects(train_subjects, fraction, seed)

    print("Subject allocation:")
    print(f"  Total train: {len(train_subjects)} | sampled: {len(sampled_train)}")
    print(f"  Validation: {len(val_subjects)}")
    print(f"  Seed: {seed}\n")

    config = PreprocessingConfig.from_yaml(
        CONFIG_DIR / "notch_bandpass_resample_znorm.yaml"
    )

    best_f1 = train_contextfree_classifierhead(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        mode="headband",
        experiment_name=experiment_name,
        preprocess_config=config,
        use_cache=True,
        ssl_checkpoint=ssl_checkpoint,
        freeze_encoder=False,
        d_model=128,
        nhead=4,
        num_layers_encoder=3,
        dim_feedforward=512,
        dropout_encoder=0.2,
        target_tokens=240,
        class_weighted_loss=True,
        gradient_clip=5.0,
        early_stopping_patience=8,
        num_classes=5,
        train_subjects=sampled_train,
        val_subjects=val_subjects,
    )

    print("\n" + "=" * 80)
    print(f"STAGE 1 COMPLETE - Best Val F1: {best_f1:.4f}")
    print("=" * 80 + "\n")
    return best_f1


def run_stage2(
    fraction: float,
    seed: int,
    ssl_checkpoint: Path,
    num_epochs: int,
    batch_size: int,
    lr_encoder: float,
    lr_head: float,
    experiment_name: str,
):
    print("\n" + "=" * 80)
    print("STAGE 2: Final Evaluation (context-free head)")
    print("=" * 80)
    print(f"Train on {fraction*100:.0f}% of (train + val); test on full test_subjects")
    print("=" * 80 + "\n")

    train_subjects = get_train_subjects()
    val_subjects = get_val_subjects()
    test_subjects = get_test_subjects()

    combined = train_subjects + val_subjects
    sampled_train = sample_subjects(combined, fraction, seed)

    print("Subject allocation:")
    print(f"  Train: {len(train_subjects)} | Val: {len(val_subjects)} | Combined: {len(combined)}")
    print(f"  Sampled for training: {len(sampled_train)}")
    print(f"  Test: {len(test_subjects)}")
    print(f"  Seed: {seed}\n")

    config = PreprocessingConfig.from_yaml(
        CONFIG_DIR / "notch_bandpass_resample_znorm.yaml"
    )

    best_f1 = train_contextfree_classifierhead(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        mode="headband",
        experiment_name=experiment_name,
        preprocess_config=config,
        use_cache=True,
        ssl_checkpoint=ssl_checkpoint,
        freeze_encoder=False,
        d_model=128,
        nhead=4,
        num_layers_encoder=3,
        dim_feedforward=512,
        dropout_encoder=0.2,
        target_tokens=240,
        class_weighted_loss=True,
        gradient_clip=5.0,
        early_stopping_patience=8,
        num_classes=5,
        train_subjects=sampled_train,
        val_subjects=test_subjects,  # treat test set as held-out eval
    )

    print("\n" + "=" * 80)
    print(f"STAGE 2 COMPLETE - Test F1: {best_f1:.4f}")
    print("=" * 80 + "\n")
    return best_f1


def main():
    # Configuration
    FRACTION = 0.2  # Fraction of labeled data to use
    SEED = 42
    SSL_CHECKPOINT = CHECKPOINT_DIR / "ssl_transformer_v1" / "best_model_hb.pt"

    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    LR_ENCODER = 1e-5
    LR_HEAD = 1e-3  # if fully-supervised end-to-end desired, set LR_HEAD = LR_ENCODER

    print("\n" + "=" * 80)
    print("Context-Free SSL Classifier Head Fine-Tuning")
    print("=" * 80)
    print(f"Fraction: {FRACTION*100:.0f}% | Seed: {SEED}")
    print(f"SSL checkpoint: {SSL_CHECKPOINT}")
    print(f"Epochs: {NUM_EPOCHS} | Batch size: {BATCH_SIZE}")
    print(f"LR encoder: {LR_ENCODER} | LR head: {LR_HEAD}")
    print("=" * 80 + "\n")

    # Stage 1 (tuning)
    stage1_f1 = run_stage1(
        fraction=FRACTION,
        seed=SEED,
        ssl_checkpoint=SSL_CHECKPOINT,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr_encoder=LR_ENCODER,
        lr_head=LR_HEAD,
        experiment_name=f"ctxfree_stage1_p{FRACTION}",
    )

    # Stage 2 (final evaluation) - enable when ready
    # stage2_f1 = run_stage2(
    #     fraction=FRACTION,
    #     seed=SEED,
    #     ssl_checkpoint=SSL_CHECKPOINT,
    #     num_epochs=NUM_EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     lr_encoder=LR_ENCODER,
    #     lr_head=LR_HEAD,
    #     experiment_name=f"ctxfree_stage2_p{FRACTION}",
    # )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Stage 1 best Val F1: {stage1_f1:.4f}")
    # print(f"Stage 2 Test F1: {stage2_f1:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.training.sequence_models_v2 import train_contextsensitive_classifier

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

    best_mf1, fin_acc, fin_per_class_f1 = train_contextsensitive_classifier(
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
    print(f"STAGE 1 COMPLETE: Best Val MF1: {best_mf1:.4f}")
    print(f"Final Val Accuracy: {fin_acc:.4f}")
    print(f"Final Val Per-Class F1: {fin_per_class_f1}")
    print("=" * 80 + "\n")
    return best_mf1, fin_acc, fin_per_class_f1

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
    fully_supervised = ssl_checkpoint is None
    mode_str = "Fully-Supervised Training" if fully_supervised else "SSL Fine-Tuning"
    print("\n" + "=" * 80)
    print(f"STAGE 2: Final Evaluation ({mode_str})")
    print("=" * 80)
    print(f"Train on {fraction*100:.0f}% of (train + val) subjects (nights); test on full test_subjects")
    print("=" * 80 + "\n")

    assert SPLITS_FILE.exists(), f"Splits file not found: {SPLITS_FILE}"
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)

    train_subjects = splits['train_subjects']
    val_subjects = splits['val_subjects']
    test_subjects = splits['test_subjects']

    combined = train_subjects + val_subjects
    sampled_train = sample_subjects(combined, fraction, seed)

    print("Subject allocation:")
    print(f"Train: {len(train_subjects)} nights | Val: {len(val_subjects)} nights | Combined: {len(combined)} nights")
    print(f"Sampled for training: {len(sampled_train)} nights")
    print(f"Test: {len(test_subjects)} nights")
    print(f"Seed: {seed}\n")

    config = PreprocessingConfig.from_yaml(
        CONFIG_DIR / "notch_bandpass_resample_znorm.yaml"
    )

    best_mf1, fin_acc, fin_per_class_f1 = train_contextsensitive_classifier(
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
        val_subjects=test_subjects,
    )

    print("\n" + "=" * 80)
    print(f"STAGE 2 COMPLETE: Test MF1: {best_mf1:.4f}")
    print(f"Final Test Accuracy: {fin_acc:.4f}")
    print(f"Final Test Per-Class F1: {fin_per_class_f1}")
    print("=" * 80 + "\n")
    return best_mf1, fin_acc, fin_per_class_f1


def main():
    SEED = 42
    NUM_EPOCHS = 200
    BATCH_SIZE = 64
    SEQ_LENGTH = 20
    STRIDE = 5
    res_finetuning_dict = {}
    res_fullysuperv_dict = {}
    TARGET_DIR = PROJECT_ROOT / "reports" / "metrics"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for mode in ["ssl-finetuning", "fully-supervised"]:
        if mode == "ssl-finetuning":
            SSL_CHECKPOINT = CHECKPOINT_DIR / "ssl_cross_modal_notch_bandpass_resample_znorm_v1" / "best_model.pt"
            FREEZE_ENCODER = False  # allow some learning in the encoder
            LR_GRU = 1e-4
            LR_ENCODER = 1e-5
        elif mode == "fully-supervised":
            SSL_CHECKPOINT = None
            FREEZE_ENCODER = False
            LR_GRU = 1e-4
            LR_ENCODER = LR_GRU
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for p in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            print("\n" + "#" * 100)
            print(f"RUNNING MODE: {mode.upper()} | LABELED FRACTION: {p*100:.0f}%")
            print("#" * 100 + "\n")

            stage2_mf1, stage2_acc, stage2_per_class_f1 = run_stage2(
                fraction=p,
                seed=SEED,
                ssl_checkpoint=SSL_CHECKPOINT,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                seq_length=SEQ_LENGTH,
                stride=STRIDE,
                lr_encoder=LR_ENCODER,
                freeze_encoder_flag=FREEZE_ENCODER,
                lr_gru=LR_GRU,
                experiment_name=f"ctxsensitive_stage2_p{p}_{mode.lower().replace('-', '_')}",
            )
            if mode == "ssl-finetuning":
                res_finetuning_dict[str(p)] = {
                    "stage2_mf1": stage2_mf1,
                    "stage2_acc": stage2_acc,
                    "stage2_per_class_f1": stage2_per_class_f1.tolist(),
                }
            elif mode == "fully-supervised":
                res_fullysuperv_dict[str(p)] = {
                    "stage2_mf1": stage2_mf1,
                    "stage2_acc": stage2_acc,
                    "stage2_per_class_f1": stage2_per_class_f1.tolist(),
                }
            print("\n" + "=" * 80)
            print("RUN COMPLETE")
            print("=" * 80)
            print(f"Mode: {mode.upper()}")
            print(f"Fraction: {p*100:.0f}%")
            print(f"Stage 2 best Val F1: {stage2_mf1:.4f}")
            print(f"Stage 2 final Val Acc: {stage2_acc:.4f}")
            print(f"Stage 2 final Per-Class F1: {stage2_per_class_f1}")
            print("=" * 80 + "\n")
    with open(TARGET_DIR / "ctxsensitive_finetuning_results_stage2.json", 'w') as f:
        json.dump(res_finetuning_dict, f, indent=4)
    with open(TARGET_DIR / "ctxsensitive_fullysupervised_results_stage2.json", 'w') as f:
        json.dump(res_fullysuperv_dict, f, indent=4)
    
            



    # # Fraction of labeled data to use (e.g., 0.1 = 10%, 1.0 = 100%)
    # FRACTION = 0.5
    # SEED = 42
    # SSL_CHECKPOINT = None

    # # Choose training mode
    # # Option 1: FINE-TUNING (set to your SSL checkpoint path)
    # ### when running on laptop:
    # # SSL_CHECKPOINT = CHECKPOINT_LEOMED_DIR / "ssl_cross_modal_notch_bandpass_resample_znorm_v1" / "best_model.pt"
    # ### else when running on cluster:
    # SSL_CHECKPOINT = CHECKPOINT_DIR / "ssl_cross_modal_notch_bandpass_resample_znorm_v1" / "best_model.pt"
    
    # # Option 2: FULLY-SUPERVISED END-TO-END (set to None)
    # # SSL_CHECKPOINT = None
    
    # NUM_EPOCHS = 200
    # BATCH_SIZE = 64
    # SEQ_LENGTH = 20
    # STRIDE = 5
    
    # if SSL_CHECKPOINT is None:
    #     # Fully-supervised: use same LR for encoder and GRU
    #     LR_ENCODER = 1e-4
    #     LR_GRU = 1e-4
    #     FREEZE_ENCODER = False
    # else:
    #     # Fine-tuning: smaller encoder LR, larger GRU LR, set freeze flag to train GRU only
    #     LR_ENCODER = 1e-5
    #     FREEZE_ENCODER = False
    #     LR_GRU = 1e-4

    # mode_str = "Fully-Supervised" if SSL_CHECKPOINT is None else "SSL-FineTuning"
    
    # print("\n" + "=" * 80)
    # print(f"Context-Sensitive Model Training - {mode_str}")
    # print("=" * 80)
    # print(f"Fraction: {FRACTION*100:.0f}% | Seed: {SEED}")
    # print(f"SSL checkpoint: {SSL_CHECKPOINT}")
    # print(f"Epochs: {NUM_EPOCHS} | Batch size: {BATCH_SIZE}")
    # print(f"Seq length: {SEQ_LENGTH} | Stride: {STRIDE}")
    # print(f"LR encoder: {LR_ENCODER} | LR GRU: {LR_GRU}")
    # print("=" * 80 + "\n")

    # stage1_mf1 = run_stage1(
    #     fraction=FRACTION,
    #     seed=SEED,
    #     ssl_checkpoint=SSL_CHECKPOINT,
    #     num_epochs=NUM_EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     seq_length=SEQ_LENGTH,
    #     stride=STRIDE,
    #     lr_encoder=LR_ENCODER,
    #     freeze_encoder_flag=FREEZE_ENCODER,
    #     lr_gru=LR_GRU,
    #     experiment_name=f"ctxsensitive_stage1_p{FRACTION}_{mode_str.lower().replace('-', '_')}",
    # )

    # print("\n" + "=" * 80)
    # print("DONE")
    # print("=" * 80)
    # print(f"Mode: {mode_str}")
    # print(f"Fraction: {FRACTION*100:.0f}%")
    # print(f"Stage 1 best Val F1: {stage1_mf1:.4f}")
    # print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
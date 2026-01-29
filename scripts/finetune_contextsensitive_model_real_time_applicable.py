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
        bidirectional_gru=False,   # changed to False for real-time applicability
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
    # SEQ_LENGTH = 10     # NOW L=15 instead of L=20 (shorter sequences for real-time applicability)
    STRIDE = 1  # changed to smaller stride, best match to inference setting
    res_finetuning_dict = {}
    res_fullysuperv_dict = {}
    TARGET_DIR = PROJECT_ROOT / "reports" / "metrics"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for l in [4, 8, 15]:
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
                    seq_length=l,
                    stride=STRIDE,
                    lr_encoder=LR_ENCODER,
                    freeze_encoder_flag=FREEZE_ENCODER,
                    lr_gru=LR_GRU,
                    experiment_name=f"ctxsensitive_stage2_p{p}_{mode.lower().replace('-', '_')}_real_time_applicable_l{l}",
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
        with open(TARGET_DIR / f"ctxsensitive_finetuning_results_stage2_real_time_applicable_l{l}.json", 'w') as f:
            json.dump(res_finetuning_dict, f, indent=4)
        with open(TARGET_DIR / f"ctxsensitive_fullysupervised_results_stage2_real_time_applicable_l{l}.json", 'w') as f:
            json.dump(res_fullysuperv_dict, f, indent=4)

if __name__ == "__main__":
    main()
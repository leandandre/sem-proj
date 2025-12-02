import sys
import json
import csv
import time
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from sem_proj.training.epoch_models import train_epochtransformer
from sem_proj.data.preprocessing import PreprocessingConfig

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

def load_best_metrics(experiment_name: str):
    """Load metrics from best_model.pt for an experiment."""
    ckpt_path = CHECKPOINT_DIR / experiment_name / "best_model.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        return {
            "epoch": ckpt.get("epoch", None),
            "val_loss": ckpt.get("val_loss", None),
            "val_accuracy": ckpt.get("val_accuracy", None),
            "val_macro_f1": ckpt.get("val_macro_f1", None),
            "val_per_class_f1": ckpt.get("val_per_class_f1", None),
            "hyperparameters": ckpt.get("hyperparameters", {}),
            "preprocessing": ckpt.get("preprocessing", {}),
            "training_config": ckpt.get("training_config", {}),
        }
    return None


def main():
    # Grid of hyperparameters to test
    D_MODELS = [64, 96, 128]             # embedding dims
    NUM_LAYERS_LIST = [4, 6]             # transformer depth
    TARGET_TOKENS_LIST = [240, 480]         # token counts
    PREPROC_LIST = [
        "notch_bandpass_resample_znorm",
        # "notch_bandpass_resample",
    ]

    # Fixed training settings
    NUM_EPOCHS = 120
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    USE_CACHE = True
    CLASS_WEIGHTED_LOSS = False
    USE_CONV1D = True

    # Fixed transformer architecture
    NHEAD = 8
    DROPOUT = 0.2
    NUM_CLASSES = 5

    results = []
    start_all = time.time()

    total_runs = len(PREPROC_LIST) * len(TARGET_TOKENS_LIST) * len(D_MODELS) * len(NUM_LAYERS_LIST)
    current_run = 0

    for preproc_name in PREPROC_LIST:
        cfg_path = PROJECT_ROOT / "configs" / "preprocess" / f"{preproc_name}.yaml"
        if not cfg_path.exists():
            print(f"Skipping: missing preprocessing config: {cfg_path}")
            continue
        preproc_cfg = PreprocessingConfig.from_yaml(cfg_path)

        for target_tokens in TARGET_TOKENS_LIST:
            for d_model in D_MODELS:
                if d_model % NHEAD != 0:
                    print(f"Skipping d_model={d_model} (not divisible by nhead={NHEAD})")
                    continue

                for num_layers in NUM_LAYERS_LIST:
                    current_run += 1
                    dim_feedforward = 4 * d_model
                    model_type = "conv1d_v2" if USE_CONV1D else "meanpool"
                    experiment_name = (
                        f"ablate_{model_type}_{preproc_name}_"
                        f"d{d_model}_tok{target_tokens}_h{NHEAD}_L{num_layers}"
                    )

                    print("\n" + "=" * 80)
                    print(f"RUN {current_run}/{total_runs}: {experiment_name}")
                    print("=" * 80)

                    model_cfg = {
                        "input_channels": 2,
                        "d_model": d_model,
                        "nhead": NHEAD,
                        "num_layers": num_layers,
                        "dim_feedforward": dim_feedforward,
                        "dropout": DROPOUT,
                        "num_classes": NUM_CLASSES,
                        "target_tokens": target_tokens,
                    }

                    run_start = time.time()
                    try:
                        _ = train_epochtransformer(
                            num_epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            lr=LEARNING_RATE,
                            experiment_name=experiment_name,
                            model_kwargs=model_cfg,
                            preprocess_config=preproc_cfg,
                            use_cache=USE_CACHE,
                            class_weighted_loss=CLASS_WEIGHTED_LOSS,
                            use_conv1d=USE_CONV1D,
                        )
                    except Exception as e:
                        print(f"ERROR in run {experiment_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        results.append({
                            "experiment": experiment_name,
                            "preprocessing": preproc_name,
                            "d_model": d_model,
                            "dim_feedforward": dim_feedforward,
                            "target_tokens": target_tokens,
                            "nhead": NHEAD,
                            "num_layers": num_layers,
                            "dropout": DROPOUT,
                            "class_weighted_loss": CLASS_WEIGHTED_LOSS,
                            "use_conv1d": USE_CONV1D,
                            "batch_size": BATCH_SIZE,
                            "learning_rate": LEARNING_RATE,
                            "status": "failed",
                            "error": str(e),
                            "duration_min": round((time.time() - run_start) / 60, 2),
                        })
                        continue

                    metrics = load_best_metrics(experiment_name)
                    duration_min = round((time.time() - run_start) / 60, 2)

                    if metrics is None:
                        print(f"WARNING: No best_model.pt for {experiment_name}")
                        results.append({
                            "experiment": experiment_name,
                            "preprocessing": preproc_name,
                            "d_model": d_model,
                            "dim_feedforward": dim_feedforward,
                            "target_tokens": target_tokens,
                            "nhead": NHEAD,
                            "num_layers": num_layers,
                            "dropout": DROPOUT,
                            "class_weighted_loss": CLASS_WEIGHTED_LOSS,
                            "use_conv1d": USE_CONV1D,
                            "batch_size": BATCH_SIZE,
                            "learning_rate": LEARNING_RATE,
                            "status": "no_checkpoint",
                            "duration_min": duration_min,
                        })
                    else:
                        training_cfg = metrics.get('training_config', {})
                        results.append({
                            "experiment": experiment_name,
                            "preprocessing": preproc_name,
                            "d_model": d_model,
                            "dim_feedforward": dim_feedforward,
                            "target_tokens": target_tokens,
                            "nhead": NHEAD,
                            "num_layers": num_layers,
                            "dropout": DROPOUT,
                            "class_weighted_loss": CLASS_WEIGHTED_LOSS,
                            "use_conv1d": USE_CONV1D,
                            "batch_size": BATCH_SIZE,
                            "learning_rate": LEARNING_RATE,
                            "num_trainable_params": training_cfg.get('num_trainable_params', None),
                            "status": "ok",
                            "val_loss": float(metrics["val_loss"]) if metrics["val_loss"] is not None else None,
                            "val_accuracy": float(metrics["val_accuracy"]) if metrics["val_accuracy"] is not None else None,
                            "val_macro_f1": float(metrics["val_macro_f1"]) if metrics["val_macro_f1"] is not None else None,
                            "val_per_class_f1": metrics["val_per_class_f1"] or [],
                            "best_epoch": int(metrics["epoch"]) + 1 if metrics["epoch"] is not None else None,
                            "duration_min": duration_min,
                        })

    total_min = round((time.time() - start_all) / 60, 2)
    total_hours = round(total_min / 60, 2)
    
    print("\n" + "=" * 80)
    print(f"ABLATION STUDY COMPLETE!")
    print(f"Total time: {total_min:.1f} min ({total_hours:.1f} hours)")
    print(f"Completed runs: {len([r for r in results if r['status'] == 'ok'])}/{total_runs}")
    print("=" * 80)

    # Save JSON and CSV summaries
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = CHECKPOINT_DIR / "ablation_summary.json"
    summary_csv = CHECKPOINT_DIR / "ablation_summary.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # CSV with key fields
    fieldnames = [
        "experiment", "status", "preprocessing", "use_conv1d",
        "d_model", "dim_feedforward", "nhead", "num_layers",
        "dropout", "target_tokens",
        "batch_size", "learning_rate",
        "class_weighted_loss", "num_trainable_params",
        "val_loss", "val_accuracy", "val_macro_f1",
        "best_epoch", "duration_min",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, None) for k in fieldnames}
            writer.writerow(row)

    # Print top-10 by macro F1
    ok_runs = [r for r in results if r.get("status") == "ok" and r.get("val_macro_f1") is not None]
    ok_runs.sort(key=lambda r: r["val_macro_f1"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 10 RUNS BY MACRO F1:")
    print("=" * 80)
    for i, r in enumerate(ok_runs[:10], 1):
        print(f"{i:>2}. {r['experiment']}")
        print(f"    Macro F1:  {r['val_macro_f1']:.4f}")
        print(f"    Accuracy:  {r['val_accuracy']:.4f}")
        print(f"    d_model:   {r['d_model']}, target_tokens: {r['target_tokens']}")
        print(f"    Preproc:   {r['preprocessing']}")
        print(f"    Duration:  {r['duration_min']:.1f} min\n")

    print(f"\nSummary files saved:")
    print(f"  JSON: {summary_json}")
    print(f"  CSV:  {summary_csv}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
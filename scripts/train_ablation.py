import sys
import json
import csv
import time
from pathlib import Path

# Add project root to PYTHONPATH so we can import sem_proj.*
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
    ckpt = torch.load(ckpt_path, map_name='cpu' if hasattr(torch, 'load') else 'cpu')
    # Fallback for standard torch.load API
    if isinstance(ckpt, dict):
        return {
            "epoch": ckpt.get("epoch", None),
            "val_loss": ckpt.get("val_loss", None),
            "val_accuracy": ckpt.get("val_accuracy", None),
            "val_macro_f1": ckpt.get("val_macro_f1", None),
            "val_per_class_f1": ckpt.get("val_per_class_f1", None),
            "hyperparameters": ckpt.get("hyperparameters", {}),
            "preprocessing": ckpt.get("preprocessing", {}),
        }
    return None


def main():
    # Grid
    D_MODELS = [32, 64, 128]
    MAX_TOKENS_LIST = [1024, 512]
    PREPROC_LIST = [
        "notch_bandpass_resample_znorm",
        "notch_bandpass_resample",
        "notch_bandpass",
    ]

    # Fixed training settings (adjust as needed)
    NUM_EPOCHS = 150
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    USE_CACHE = True
    CLASS_WEIGHTED_LOSS = True
    USE_CONV1D = True  # Set True to use the Conv1D patch embedding variant

    # Fixed transformer depth/heads; ensure d_model % nhead == 0
    NHEAD = 8
    NUM_LAYERS = 4
    DROPOUT = 0.2
    NUM_CLASSES = 5

    results = []
    start_all = time.time()

    for preproc_name in PREPROC_LIST:
        # Load preprocessing config
        cfg_path = PROJECT_ROOT / "configs" / "preprocess" / f"{preproc_name}.yaml"
        if not cfg_path.exists():
            print(f"Skipping: missing preprocessing config: {cfg_path}")
            continue
        preproc_cfg = PreprocessingConfig.from_yaml(cfg_path)

        for max_tokens in MAX_TOKENS_LIST:
            for d_model in D_MODELS:
                if d_model % NHEAD != 0:
                    print(f"Skipping d_model={d_model} (not divisible by nhead={NHEAD})")
                    continue

                dim_feedforward = 4 * d_model
                model_type = "conv1d" if USE_CONV1D else "meanpool"
                experiment_name = (
                    f"ablate_{model_type}_{preproc_name}_"
                    f"d{d_model}_tok{max_tokens}_h{NHEAD}_L{NUM_LAYERS}"
                )

                print("\n" + "=" * 80)
                print(f"RUN: {experiment_name}")
                print("=" * 80)

                model_cfg = {
                    "input_channels": 2,
                    "d_model": d_model,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dim_feedforward": dim_feedforward,
                    "dropout": DROPOUT,
                    "num_classes": NUM_CLASSES,
                    "max_tokens": max_tokens,
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
                    results.append({
                        "experiment": experiment_name,
                        "preprocessing": preproc_name,
                        "d_model": d_model,
                        "dim_feedforward": dim_feedforward,
                        "max_tokens": max_tokens,
                        "nhead": NHEAD,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "class_weighted_loss": CLASS_WEIGHTED_LOSS,
                        "use_conv1d": USE_CONV1D,
                        "status": "failed",
                        "error": str(e),
                        "duration_min": round((time.time() - run_start) / 60, 2),
                    })
                    continue

                # Load metrics from best checkpoint
                metrics = load_best_metrics(experiment_name)
                duration_min = round((time.time() - run_start) / 60, 2)

                if metrics is None:
                    print(f"WARNING: No best_model.pt for {experiment_name}")
                    results.append({
                        "experiment": experiment_name,
                        "preprocessing": preproc_name,
                        "d_model": d_model,
                        "dim_feedforward": dim_feedforward,
                        "max_tokens": max_tokens,
                        "nhead": NHEAD,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "class_weighted_loss": CLASS_WEIGHTED_LOSS,
                        "use_conv1d": USE_CONV1D,
                        "status": "no_checkpoint",
                        "duration_min": duration_min,
                    })
                else:
                    print(f"✓ Completed {experiment_name} | "
                          f"Val Macro F1={metrics['val_macro_f1']:.4f} | "
                          f"Val Acc={metrics['val_accuracy']:.4f} | "
                          f"Best epoch={metrics['epoch'] + 1 if metrics['epoch'] is not None else 'NA'} | "
                          f"Time={duration_min} min")

                    results.append({
                        "experiment": experiment_name,
                        "preprocessing": preproc_name,
                        "d_model": d_model,
                        "dim_feedforward": dim_feedforward,
                        "max_tokens": max_tokens,
                        "nhead": NHEAD,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "class_weighted_loss": CLASS_WEIGHTED_LOSS,
                        "use_conv1d": USE_CONV1D,
                        "status": "ok",
                        "val_loss": round(float(metrics["val_loss"]), 6) if metrics["val_loss"] is not None else None,
                        "val_accuracy": round(float(metrics["val_accuracy"]), 6) if metrics["val_accuracy"] is not None else None,
                        "val_macro_f1": round(float(metrics["val_macro_f1"]), 6) if metrics["val_macro_f1"] is not None else None,
                        "val_per_class_f1": [round(float(x), 6) for x in (metrics["val_per_class_f1"] or [])],
                        "best_epoch": int(metrics["epoch"]) + 1 if metrics["epoch"] is not None else None,
                        "duration_min": duration_min,
                    })

    total_min = round((time.time() - start_all) / 60, 2)
    print("\n" + "=" * 80)
    print(f"Ablation finished in {total_min} min. Writing summary...")
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
        "dropout", "max_tokens", "class_weighted_loss",
        "val_loss", "val_accuracy", "val_macro_f1",
        "best_epoch", "duration_min",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, None) for k in fieldnames}
            writer.writerow(row)

    # Print top-5 by macro F1
    ok_runs = [r for r in results if r.get("status") == "ok" and r.get("val_macro_f1") is not None]
    ok_runs.sort(key=lambda r: r["val_macro_f1"], reverse=True)

    print("\nTop runs by Macro F1:")
    for i, r in enumerate(ok_runs[:5], 1):
        print(f"{i:>2}. {r['experiment']}: Macro F1={r['val_macro_f1']:.4f}, "
              f"Acc={r['val_accuracy']:.4f}, Preproc={r['preprocessing']}, "
              f"d_model={r['d_model']}, tok={r['max_tokens']}, conv1d={r['use_conv1d']}")

    print(f"\nSummary files:")
    print(f"  JSON: {summary_json}")
    print(f"  CSV : {summary_csv}")


if __name__ == "__main__":
    main()
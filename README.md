# Semester Project - Cross-Modal Representation Learning for Sleep Staging

This repository contains the code for my semester project on cross-modal representation learning for sleep staging from EEG. The goal is to learn shared representations between a wearable headband (2-channel EEG) and a PSG reference (6-channel EEG), then fine-tune for sleep stage classification in both context-free and context-sensitive settings.

The thesis document is the primary reference for methodology and results; this repo is meant as a clean code companion for my supervisor and colleagues.

## What is in here

- **Cross-modal SSL pretraining** between headband and PSG signals with token-level contrastive learning.
- **Context-free classification** on single 30s epochs (SSL fine-tuning or fully supervised).
- **Context-sensitive classification** using a GRU over sequences of epochs.
- **Data preprocessing + caching** for faster iteration.

## Repository layout

- [src/sem_proj/data](src/sem_proj/data) dataset loading, preprocessing, caching, splits, and augmentations.
- [src/sem_proj/models](src/sem_proj/models) model zoo (final models marked in docstrings).
- [src/sem_proj/training](src/sem_proj/training) training loops for SSL, context-free, and context-sensitive models.
- [scripts](scripts) runnable entry points, analysis scripts, and SLURM launchers.
- [configs/preprocess](configs/preprocess) preprocessing YAMLs.
- [data](data) local data cache and BOAS raw data layout.
- [checkpoints](checkpoints) model checkpoints (local).
- [checkpoints_leomed](checkpoints_leomed) checkpoints from laptop/other machine.
- [logs](logs) TensorBoard logs.
- [plots](plots) generated figures.
- [reports](reports) metrics JSONs and report artifacts.
- [notebooks](notebooks) exploratory notebooks, evaluation helpers, and final test set runs.

## Data layout

The code expects the BOAS dataset under the repo root:

```
data/
	raw/
		boas/
			participants.tsv
			sub-*/
				eeg/
					*_acq-headband_eeg.edf
					*_acq-psg_eeg.edf
					*_acq-headband_events.tsv
					*_acq-psg_events.tsv
	processed/
		data_splits_70_15_15.json
		data_splits_80_05_15.json
		cache/
```

The `cache/` directory stores preprocessed subjects keyed by preprocessing config.

## Preprocessing

Preprocessing is configured via YAML in [configs/preprocess](configs/preprocess). The main options include notch filtering, bandpass, resampling, and z-normalization (epoch-wise or per-night). See [src/sem_proj/data/preprocessing.py](src/sem_proj/data/preprocessing.py) for details.

## How to run (light)

These are the main entry points used in the project. They assume the BOAS data is present and a Python environment with the required packages is available.

- **SSL pretraining** (cross-modal):
	- `python -m sem_proj.training.epoch_models_ssl`
- **Context-free fine-tuning / supervised training**:
	- `python scripts/finetune_contextfree_classifierhead.py`
- **Context-sensitive fine-tuning / supervised training**:
	- `python scripts/finetune_contextsensitive_model_v3.py`
- **Cache management**:
	- `python scripts/manage_cache.py --info`
	- `python scripts/manage_cache.py --clear`

For cluster runs, SLURM launchers are in:

- [scripts/run_epoch_models_ssl.sh](scripts/run_epoch_models_ssl.sh)
- [scripts/run_finetune_contextfree_classifierhead.sh](scripts/run_finetune_contextfree_classifierhead.sh)
- [scripts/run_finetune_contextsensitive_model_v3.sh](scripts/run_finetune_contextsensitive_model_v3.sh)

## Notes on models

The primary models used in the final experiments are:

- `SSLEpochTransformerConv1D_v2` (SSL encoder)
- `SSLClassifierHead` / `SSLLinearProbing` (context-free head)
- `SequenceGRUClassifier` (context-sensitive sequence model)

These are defined in [src/sem_proj/models/model_factory.py](src/sem_proj/models/model_factory.py). Other models are marked as “ignore” in comments and were exploratory.

## Outputs

- Checkpoints: [checkpoints](checkpoints) and [checkpoints_leomed](checkpoints_leomed)
- Downloadable checkpoints (pretrained and fine-tuned): https://polybox.ethz.ch/index.php/s/dPfnBQbcbtxLyfi
- TensorBoard logs: [logs](logs)
- Metrics and plots: [reports](reports), [plots](plots)

## Checkpoint download and contents

The Polybox link above contains the pretrained SSL encoder and the main fine-tuned checkpoints used in the thesis. After downloading, place them under [checkpoints_leomed](checkpoints_leomed) (same structure as in this repo).

Typical layout:

```
checkpoints_leomed/
	ssl_cross_modal_notch_bandpass_resample_znorm_v1/
		best_model.pt
	ctxfree_stage1_p*_ssl_finetuning_stronger_MLP_v1/
		best_model.pt
	ctxfree_stage1_p*_fully_supervised_stronger_MLP_v1/
		best_model.pt
	ctxfree_stage1_p*_ssl_finetuning_linearprobing/
		best_model.pt
	ctxsensitive_val_step_p*_ssl_finetuning_bidirTrue_L10_s2/
		best_model.pt
	ctxsensitive_val_step_p*_fully_supervised_bidirTrue_L10_s2/
		best_model.pt
	ctxsensitive_val_step_p*_ssl_finetuning_bidirFalse_L5_s1/
		best_model.pt
	ctxsensitive_val_step_p*_fully_supervised_bidirFalse_L5_s1/
		best_model.pt
```

Legend: `p*` is the labeled-data fraction {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}.

## Best-model setup (summary)

These are the configurations used for the best context-free and context-sensitive models. They match the checkpoints above and the evaluation notebooks.

Context-free (epoch classifier, headband):

- Preprocess: `notch_bandpass_resample_znorm.yaml`
- Encoder: `SSLEpochTransformerConv1D_v2` with `d_model=128`, `nhead=4`, `num_layers=2`, `dim_feedforward=512`, `dropout=0.2`, `target_tokens=240`
- Head: `SSLClassifierHead` with `dropout=0.2`
- Training: `batch_size=512`, `lr_encoder=1e-5` (supervised from scratch: `1e-3`), `lr_head=1e-3`, `gradient_clip=5.0`, `early_stopping_patience=12`, class-weighted loss

Context-sensitive (sequence GRU, headband):

- Preprocess: `notch_bandpass_resample_znorm.yaml`
- Encoder: same as context-free
- Sequence model: `SequenceGRUClassifier` with `hidden_size=128`, `num_layers=1`, `bidirectional=True`
- Training: `batch_size=64`, `seq_len=10`, `stride=2`, `lr_encoder=1e-5` (supervised from scratch: `1e-4`), `lr_gru=1e-4`, `gradient_clip=5.0`, `early_stopping_patience=12`, class-weighted loss
- Note: older checkpoints may not store `seq_len` and `stride`; the intended values are listed above.

Model construction (context-free):

```python
encoder = SSLEpochTransformerConv1D_v2(
		d_model=128,
		nhead=4,
		num_layers=2,
		dim_feedforward=512,
		dropout=0.2,
		target_tokens=240,
)
head = SSLClassifierHead(d_model=128, dropout=0.2, num_classes=5)
model = nn.Sequential(encoder, head)
```

Model construction (context-sensitive):

```python
encoder = SSLEpochTransformerConv1D_v2(
		d_model=128,
		nhead=4,
		num_layers=2,
		dim_feedforward=512,
		dropout=0.2,
		target_tokens=240,
)
context_model = SequenceGRUClassifier(
		epoch_model=encoder,
		hidden_size=128,
		num_layers=1,
		num_classes=5,
		bidirectional=True,
)
```

## Dependencies (high level)

This code uses common scientific Python tooling including PyTorch, MNE, NumPy, Pandas, scikit-learn, Matplotlib/Seaborn, and TensorBoard. The project targets Python >= 3.8.


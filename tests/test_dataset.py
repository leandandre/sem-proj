from sem_proj.data.datasets import BoasSequenceDataset
from sem_proj.data.preprocessing import PreprocessingConfig

config = PreprocessingConfig.from_yaml("configs/preprocess/notch_bandpass_resample_znorm.yaml")

# Create sequence dataset
ds_seq = BoasSequenceDataset(
    subjects=["sub-1", "sub-2"],
    mode="headband",
    seq_len=10,
    stride=5,
    preprocess_config=config,
    use_cache=True,
    add_channel_dim=False,
)

print(f"Total sequences: {len(ds_seq)}")
print(f"Underlying epochs: {len(ds_seq.epoch_dataset)}")

# Get first sequence
x_seq, y_seq = ds_seq[0]
print(f"Sequence shape: {x_seq.shape}")  # (seq_len, 2, 3840)
print(f"Labels shape: {y_seq.shape}")    # (seq_len,)
print(f"Labels: {y_seq}")
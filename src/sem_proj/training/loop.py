from sem_proj.data.boa_loader import split_by_pid
from sem_proj.data.datasets import BoasDataset

splits = split_by_pid(seed=42)

train_ds_hb = BoasDataset(subjects=splits["train_subjects"], mode="headband")
val_ds_hb   = BoasDataset(subjects=splits["val_subjects"],   mode="headband")
test_ds_hb  = BoasDataset(subjects=splits["test_subjects"],  mode="headband")

### above is how it should look like ###

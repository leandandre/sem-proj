import numpy as np
import json
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sem_proj.data.datasets import BoasSequenceDataset
from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.models.model_factory import SSLEpochTransformerConv1D_v2, SequenceGRUClassifier
from sem_proj.data.boa_loader import build_pid_mappings

CONFIG_DIR = PROJECT_ROOT / "configs" / "preprocess"
CHECKPOINT_LEOMED_DIR = PROJECT_ROOT / "checkpoints_leomed"
SPLITS_FILE = PROJECT_ROOT / "data" / "processed" / "data_splits_70_15_15.json"
TARGET_DIR_METRICS = PROJECT_ROOT / "reports" / "metrics"
TARGET_DIR_METRICS.mkdir(parents=True, exist_ok=True)
TARGET_DIR_PLOTS = PROJECT_ROOT / "plots"
TARGET_DIR_PLOTS.mkdir(parents=True, exist_ok=True)
"""
Enhanced preprocessing with per-subject z-normalization.
Each preprocessing step can be toggled independently.
"""
import mne
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class PreprocessingConfig:
    """
    Configuration for EEG preprocessing pipeline.
    All steps are optional and can be toggled independently.
    """
    apply_notch: bool = True
    notch_freqs: list[float] = None
    apply_bandpass: bool = True
    bandpass_l_freq: Optional[float] = 0.5
    bandpass_h_freq: Optional[float] = 40.0
    apply_resample: bool = True
    resample_freq: Optional[float] = 128.0
    apply_znorm: bool = True
    apply_per_night_znorm: bool = False  # False = epoch-wise (local), True = whole-night per-channel
    
    def __post_init__(self):
        """Set default notch frequencies if not provided."""
        if self.notch_freqs is None:
            self.notch_freqs = [50.0, 100.0]  # European power line
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config for logging."""
        return asdict(self)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'PreprocessingConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        return cls(**cfg_dict)
    
    def to_yaml(self, path: Path):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def default_config(cls) -> 'PreprocessingConfig':
        return cls(
            apply_notch=True,
            notch_freqs=[50.0, 100.0],
            apply_bandpass=True,
            bandpass_l_freq=0.5,
            bandpass_h_freq=40.0,
            apply_resample=True,
            resample_freq=128.0,
            apply_znorm=True,
            apply_per_night_znorm=False,  # default to epoch-wise
        )
    
    @classmethod
    def no_preprocessing(cls) -> 'PreprocessingConfig':
        """No preprocessing at all."""
        return cls(
            apply_notch=False,
            apply_bandpass=False,
            apply_resample=False,
            apply_znorm=False,
        )
    
    @classmethod
    def only_znorm(cls) -> 'PreprocessingConfig':
        """Only z-normalization, no filtering."""
        return cls(
            apply_notch=False,
            apply_bandpass=False,
            apply_resample=False,
            apply_znorm=True,
        )
    
    @classmethod
    def no_znorm(cls) -> 'PreprocessingConfig':
        """All preprocessing except z-normalization."""
        return cls(
            apply_notch=True,
            notch_freqs=[50.0, 100.0],
            apply_bandpass=True,
            bandpass_l_freq=0.5,
            bandpass_h_freq=40.0,
            apply_resample=True,
            resample_freq=128.0,
            apply_znorm=False,
        )


def preprocess_raw_basic(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
) -> mne.io.BaseRaw:
    """
    Apply basic preprocessing: notch, bandpass, resample.
    Does NOT apply z-normalization (done later per-subject).
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data (will be copied).
    config : PreprocessingConfig
        Preprocessing configuration.
    
    Returns
    -------
    mne.io.BaseRaw
        Preprocessed raw object (without z-norm).
    """
    # Work on a copy
    raw = raw.copy()
    
    # 1. Notch filter
    if config.apply_notch and config.notch_freqs:
        print(f"  Notch filter: {config.notch_freqs} Hz")
        raw.notch_filter(
            freqs=config.notch_freqs,
            picks='eeg',
            method='fir',
            verbose=False
        )
    
    # 2. Bandpass filter
    if config.apply_bandpass:
        print(f"  Bandpass: {config.bandpass_l_freq}-{config.bandpass_h_freq} Hz")
        raw.filter(
            l_freq=config.bandpass_l_freq,
            h_freq=config.bandpass_h_freq,
            picks='eeg',
            method='fir',
            verbose=False
        )
    
    # 3. Resample
    if config.apply_resample and config.resample_freq is not None:
        current_sfreq = raw.info['sfreq']
        if current_sfreq != config.resample_freq:
            print(f"  Resample: {current_sfreq} Hz → {config.resample_freq} Hz")
            raw.resample(config.resample_freq, npad='auto', verbose=False)
    
    return raw


def compute_subject_normalization_stats(
    raw: mne.io.BaseRaw,
    valid_mask: np.ndarray,
    bounds: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    If apply_per_night_znorm is True: compute per-channel mean/std across all valid epochs.
    Else: return (None, None) so epoch-wise z-norm will be used later.
    """
    if not config.apply_znorm or not config.apply_per_night_znorm:
        return None, None  # epoch-wise path
    
    valid_data = []
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            s, e = bounds[i]
            valid_data.append(raw.get_data(start=s, stop=e))
    if len(valid_data) == 0:
        n_channels = raw.get_data().shape[0]
        return np.zeros(n_channels), np.ones(n_channels)
    valid_data = np.concatenate(valid_data, axis=1)  # (channels, samples)
    mean = valid_data.mean(axis=1)
    std = valid_data.std(axis=1) + 1e-8
    return mean, std


def apply_znorm_to_epoch(
    epoch_data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Epoch-wise per-channel z-norm (default).
    If apply_per_night_znorm True and mean/std provided, use those.
    """
    if not config.apply_znorm:
        return epoch_data
    # Night (global) stats path
    if config.apply_per_night_znorm and mean is not None and std is not None:
        return (epoch_data - mean[:, None]) / (std[:, None])
    # Epoch-wise path
    m = epoch_data.mean(axis=1, keepdims=True)
    s = epoch_data.std(axis=1, keepdims=True) + 1e-8
    return (epoch_data - m) / s


def get_expected_seq_length(config: PreprocessingConfig, epoch_duration_sec: int = 30) -> int:
    """
    Calculate expected sequence length based on preprocessing config.
    
    Parameters
    ----------
    config : PreprocessingConfig
        Preprocessing configuration.
    epoch_duration_sec : int
        Duration of each epoch in seconds (default: 30).
    
    Returns
    -------
    int
        Expected number of samples per epoch.
    
    Examples
    --------
    >>> config = PreprocessingConfig(apply_resample=True, resample_freq=128.0)
    >>> get_expected_seq_length(config)
    3840
    
    >>> config = PreprocessingConfig(apply_resample=False)
    >>> get_expected_seq_length(config)
    7680
    """
    DEFAULT_BOAS_SFREQ = 256  # Hz
    
    if config.apply_resample and config.resample_freq is not None:
        return int(config.resample_freq * epoch_duration_sec)
    else:
        return DEFAULT_BOAS_SFREQ * epoch_duration_sec
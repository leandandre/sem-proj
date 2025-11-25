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
    # Notch filtering (power line noise removal)
    apply_notch: bool = True
    notch_freqs: list[float] = None  # e.g., [50.0, 100.0]
    
    # Bandpass filtering
    apply_bandpass: bool = True
    bandpass_l_freq: Optional[float] = 0.5  # Hz
    bandpass_h_freq: Optional[float] = 40.0  # Hz
    
    # Resampling
    apply_resample: bool = True
    resample_freq: Optional[float] = 128.0  # Hz
    
    # Z-normalization (per-subject, artifact-free)
    apply_znorm: bool = True
    znorm_per_channel: bool = True  # If False, normalize across all channels
    # apply_per_night_znorm: bool = False # If True, normalize per night recording, else per window
    
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
        """Default: all preprocessing enabled."""
        return cls(
            apply_notch=True,
            notch_freqs=[50.0, 100.0],
            apply_bandpass=True,
            bandpass_l_freq=0.5,
            bandpass_h_freq=40.0,
            apply_resample=True,
            resample_freq=128.0,
            apply_znorm=True,
            znorm_per_channel=True,
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
    Compute mean and std for z-normalization across valid (non-artifact/interruptic) epochs.
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw data (after notch/bandpass/resample).
    valid_mask : np.ndarray, shape (N_epochs,)
        Boolean mask: True for valid epochs (not artifacts/interruptions).
    bounds : np.ndarray, shape (N_epochs, 2)
        [start, stop) sample indices for each epoch.
    config : PreprocessingConfig
        Config to check if normalization is enabled.
    
    Returns
    -------
    mean : np.ndarray, shape (n_channels,) or scalar
        Mean per channel (or global).
    std : np.ndarray, shape (n_channels,) or scalar
        Std per channel (or global).
    """
    if not config.apply_znorm:
        return None, None
    
    # Collect all valid samples
    valid_data = []
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            start, stop = bounds[i]
            epoch_data = raw.get_data(start=start, stop=stop)  # (n_channels, n_samples)
            valid_data.append(epoch_data)
    
    if len(valid_data) == 0:
        print("  WARNING: No valid epochs for normalization!")
        n_channels = raw.get_data().shape[0]
        return np.zeros(n_channels), np.ones(n_channels)
    
    # Concatenate all valid epochs
    valid_data = np.concatenate(valid_data, axis=1)  # (n_channels, total_valid_samples)
    
    # Compute statistics
    if config.znorm_per_channel:
        mean = valid_data.mean(axis=1, keepdims=True)  # (n_channels, 1)
        std = valid_data.std(axis=1, keepdims=True) + 1e-8  # (n_channels, 1)
        mean = mean.squeeze()  # (n_channels,)
        std = std.squeeze()    # (n_channels,)
    else:
        # Global normalization across all channels
        mean = valid_data.mean()
        std = valid_data.std() + 1e-8
    
    return mean, std


def apply_znorm_to_epoch(
    epoch_data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Apply z-normalization to a single epoch.
    
    Parameters
    ----------
    epoch_data : np.ndarray, shape (n_channels, n_samples)
        Single epoch data.
    mean, std : np.ndarray
        Normalization statistics from compute_subject_normalization_stats.
    config : PreprocessingConfig
        Config (to check if normalization is enabled).
    
    Returns
    -------
    np.ndarray, shape (n_channels, n_samples)
        Normalized epoch.
    """
    if not config.apply_znorm or mean is None:
        return epoch_data
    
    if config.znorm_per_channel:
        # mean, std: (n_channels,)
        mean = mean[:, np.newaxis]  # (n_channels, 1)
        std = std[:, np.newaxis]    # (n_channels, 1)
    
    return (epoch_data - mean) / std


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
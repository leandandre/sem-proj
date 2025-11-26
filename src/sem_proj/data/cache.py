"""
Cache preprocessed subjects to disk to speed up loading.

Caching Strategy:
- Cache key = subject_id + hash of preprocessing config
- Cached data = preprocessed raw + bounds + labels + valid masks
- Cache invalidates automatically when preprocessing config changes
"""
from pathlib import Path
import pickle
import hashlib
import numpy as np
import mne
from typing import Optional, Dict, Any
from sem_proj.data.preprocessing import PreprocessingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def get_cache_key(subject: str, config: PreprocessingConfig, mode: str = "headband") -> str:
    """
    Generate unique cache key for subject + preprocessing config + mode.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-1')
    config : PreprocessingConfig
        Preprocessing configuration
    mode : str
        'headband' or 'psg'
    
    Returns
    -------
    str
        Unique cache key (e.g., 'sub-1_headband_a3f2b9c1')
    """
    # Convert config to sorted string for consistent hashing
    config_str = str(sorted(config.to_dict().items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    key = f"{subject}_{mode}_{config_hash}"
    return key


def get_cache_path(subject: str, config: PreprocessingConfig, mode: str = "headband") -> Path:
    """Get full path to cache file."""
    cache_file = CACHE_DIR / f"{get_cache_key(subject, config, mode)}.pkl"
    return cache_file


def load_from_cache(subject: str, config: PreprocessingConfig, mode: str = "headband") -> Optional[Dict[str, Any]]:
    """
    Load preprocessed subject from cache if exists.
    
    Returns
    -------
    dict or None
        Dictionary containing:
        - 'raw_data': np.ndarray, preprocessed signal data
        - 'sfreq': float, sampling frequency
        - 'ch_names': list[str], channel names
        - 'bounds': np.ndarray, epoch sample bounds
        - 'labels': np.ndarray, sleep stage labels
        - 'valid_mask': np.ndarray, boolean mask of valid epochs
        - 'norm_mean': np.ndarray, normalization mean
        - 'norm_std': np.ndarray, normalization std
    """
    cache_file = get_cache_path(subject, config, mode)
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  ✓ Loaded {subject} ({mode}) from cache")
            return data
        except Exception as e:
            print(f"  ⚠️  Cache corrupted for {subject} ({mode}): {e}")
            cache_file.unlink()  # Delete corrupted cache
            return None
    
    return None


def save_to_cache(
    subject: str,
    config: PreprocessingConfig,
    mode: str,
    raw_data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    bounds: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
):
    """
    Save preprocessed subject to cache.
    
    Parameters
    ----------
    raw_data : np.ndarray
        Preprocessed signal data (channels × samples)
    sfreq : float
        Sampling frequency after preprocessing
    ch_names : list[str]
        Channel names
    bounds : np.ndarray
        Epoch sample bounds (N_epochs × 2)
    labels : np.ndarray
        Sleep stage labels (N_epochs,)
    valid_mask : np.ndarray
        Boolean mask of valid epochs (N_epochs,)
    norm_mean : np.ndarray
        Normalization mean (for selected channels)
    norm_std : np.ndarray
        Normalization std (for selected channels)
    """
    cache_file = get_cache_path(subject, config, mode)
    
    data = {
        'raw_data': raw_data,
        'sfreq': sfreq,
        'ch_names': ch_names,
        'bounds': bounds,
        'labels': labels,
        'valid_mask': valid_mask,
        'norm_mean': norm_mean,  # may be None for epoch-wise
        'norm_std': norm_std,    # may be None for epoch-wise
        'config_hash': get_cache_key(subject, config, mode),
    }
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Cached {subject} ({mode})")
    except Exception as e:
        print(f"  ⚠️  Failed to cache {subject} ({mode}): {e}")


def clear_cache(subject: Optional[str] = None):
    """
    Clear cache files.
    
    Parameters
    ----------
    subject : str or None
        If provided, only clear cache for this subject.
        If None, clear all cache files.
    """
    if subject is not None:
        # Clear only this subject
        pattern = f"{subject}_*.pkl"
        files = list(CACHE_DIR.glob(pattern))
    else:
        # Clear all cache
        files = list(CACHE_DIR.glob("*.pkl"))
    
    for file in files:
        file.unlink()
    
    print(f"Cleared {len(files)} cache file(s)")


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about current cache.
    
    Returns
    -------
    dict
        - 'num_files': int
        - 'total_size_mb': float
        - 'subjects': list[str]
    """
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    
    total_size = sum(f.stat().st_size for f in cache_files)
    total_size_mb = total_size / (1024 * 1024)
    
    # Extract unique subjects
    subjects = set()
    for f in cache_files:
        subject = f.stem.split('_')[0]  # Extract 'sub-X' from filename
        subjects.add(subject)
    
    return {
        'num_files': len(cache_files),
        'total_size_mb': total_size_mb,
        'subjects': sorted(subjects),
    }
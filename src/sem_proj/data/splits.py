"""
Fixed data splits for BOAS dataset.
Creates train/val/test splits once and saves them to disk.
"""
from pathlib import Path
import json
from typing import Dict, List
from sem_proj.data.boa_loader import build_pid_mappings
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS_FILE = PROJECT_ROOT / "data" / "processed" / "data_splits.json"


def create_fixed_splits(
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,         # NEVER CHANGE THIS SEED
    force_recreate: bool = False,
) -> Dict[str, List[str]]:
    """
    Create fixed train/val/test splits and save to disk.
    
    Parameters
    ----------
    train_frac : float
        Fraction of data for training
    val_frac : float
        Fraction of data for validation
    test_frac : float
        Fraction of data for testing
    seed : int
        Random seed for reproducibility
    force_recreate : bool
        If True, recreate splits even if they exist
    
    Returns
    -------
    dict
        Dictionary with keys 'train_subjects', 'val_subjects', 'test_subjects'
    """
    
    # Check if splits already exist
    if SPLITS_FILE.exists() and not force_recreate:
        print(f"Loading existing splits from {SPLITS_FILE}")
        with open(SPLITS_FILE, 'r') as f:
            splits = json.load(f)
        return splits
    
    print(f"Creating new data splits with seed={seed}")
    
    # Get all participant IDs - build_pid_mappings returns (pid_to_sub, sub_to_pid)
    pid_to_sub, sub_to_pid = build_pid_mappings()
    all_pids = sorted(pid_to_sub.keys())
    
    # Shuffle PIDs
    rng = np.random.RandomState(seed)
    rng.shuffle(all_pids)
    
    # Calculate split sizes
    n_total = len(all_pids)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    # Rest goes to test (handles rounding)
    
    # Split PIDs
    train_pids = all_pids[:n_train]
    val_pids = all_pids[n_train:n_train + n_val]
    test_pids = all_pids[n_train + n_val:]
    
    # Map PIDs to subject IDs
    train_subjects = []
    val_subjects = []
    test_subjects = []
    
    for pid in train_pids:
        train_subjects.extend(pid_to_sub[pid])
    for pid in val_pids:
        val_subjects.extend(pid_to_sub[pid])
    for pid in test_pids:
        test_subjects.extend(pid_to_sub[pid])
    
    splits = {
        'train_subjects': sorted(train_subjects),
        'val_subjects': sorted(val_subjects),
        'test_subjects': sorted(test_subjects),
        'train_pids': sorted(train_pids),
        'val_pids': sorted(val_pids),
        'test_pids': sorted(test_pids),
        'metadata': {
            'seed': seed,
            'train_frac': train_frac,
            'val_frac': val_frac,
            'test_frac': test_frac,
            'n_train_subjects': len(train_subjects),
            'n_val_subjects': len(val_subjects),
            'n_test_subjects': len(test_subjects),
            'n_train_pids': len(train_pids),
            'n_val_pids': len(val_pids),
            'n_test_pids': len(test_pids),
        }
    }
    
    # Save to disk
    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_FILE, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✓ Created and saved splits to {SPLITS_FILE}")
    print(f"  Train: {len(train_subjects)} subjects ({len(train_pids)} PIDs)")
    print(f"  Val:   {len(val_subjects)} subjects ({len(val_pids)} PIDs)")
    print(f"  Test:  {len(test_subjects)} subjects ({len(test_pids)} PIDs)")
    
    return splits

### always use load_splits() to get the FULL split, never create_fixed_splits() ###
def load_splits() -> Dict[str, List[str]]:
    """
    Load fixed train/val/test splits from disk.
    Creates them if they don't exist.
    
    Returns
    -------
    dict
        Dictionary with keys 'train_subjects', 'val_subjects', 'test_subjects'
    """
    if not SPLITS_FILE.exists():
        print("No existing splits found. Creating new splits...")
        return create_fixed_splits()
    
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)
    
    return splits

### use these helper functions to get the different pid lists (training/validation/test) ###
def get_train_subjects() -> List[str]:
    """Get training subject IDs."""
    return load_splits()['train_subjects']


def get_val_subjects() -> List[str]:
    """Get validation subject IDs."""
    return load_splits()['val_subjects']


def get_test_subjects() -> List[str]:
    """Get test subject IDs (use only for final evaluation!)."""
    return load_splits()['test_subjects']


def print_split_info():
    """Print information about the current data splits."""
    splits = load_splits()
    metadata = splits['metadata']
    
    print("\n" + "="*60)
    print("BOAS Dataset Splits")
    print("="*60)
    print(f"Split file: {SPLITS_FILE}")
    print(f"Random seed: {metadata['seed']}")
    print(f"\nSplit ratios: {metadata['train_frac']:.0%} / {metadata['val_frac']:.0%} / {metadata['test_frac']:.0%}")
    print(f"\nSubjects:")
    print(f"  Train: {metadata['n_train_subjects']} subjects ({metadata['n_train_pids']} unique patients)")
    print(f"  Val:   {metadata['n_val_subjects']} subjects ({metadata['n_val_pids']} unique patients)")
    print(f"  Test:  {metadata['n_test_subjects']} subjects ({metadata['n_test_pids']} unique patients)")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Create splits if they don't exist
    splits = create_fixed_splits(force_recreate=False)
    print_split_info()
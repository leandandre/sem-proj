"""
K-fold cross-validation splits for BOAS dataset.

Creates k-fold splits on the development set (80% of data),
with a fixed test set (20% of data) held out for final evaluation only.

Split is PID-wise (all recordings from one participant stay together).
NOT USED YET
"""

from pathlib import Path
import json
from typing import Dict, List, Tuple
from sem_proj.data.boa_loader import build_pid_mappings
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS_CV_FILE = PROJECT_ROOT / "data" / "processed" / "data_splits_cv.json"


def create_cv_splits(
    dev_frac: float = 0.8,
    test_frac: float = 0.2,
    n_folds: int = 5,
    seed: int = 42,
    force_recreate: bool = False,
) -> Dict:
    """
    Create k-fold CV splits on development set + fixed test set.
    
    Splits data into:
    - Development set (80%): further split into k folds for cross-validation
    - Test set (20%): held out for final evaluation only
    
    All splits are PID-wise (participant-level).
    
    Parameters
    ----------
    dev_frac : float
        Fraction of data for development (CV). Default 0.8.
    test_frac : float
        Fraction of data for final test. Default 0.2.
    n_folds : int
        Number of folds for cross-validation.
    seed : int
        Random seed for reproducibility. NEVER CHANGE THIS.
    force_recreate : bool
        If True, recreate splits even if they exist.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'test_subjects': subjects in final test set
        - 'test_pids': PIDs in final test set
        - 'folds': list of dicts, each with 'train_subjects', 'val_subjects',
                   'train_pids', 'val_pids'
        - 'metadata': information about the splits
    """
    
    # Check if splits already exist
    if SPLITS_CV_FILE.exists() and not force_recreate:
        print(f"Loading existing CV splits from {SPLITS_CV_FILE}")
        with open(SPLITS_CV_FILE, 'r') as f:
            splits = json.load(f)
        return splits
    
    print(f"Creating new k-fold CV splits with seed={seed}, k={n_folds}")
    
    # Verify fractions sum to 1
    assert abs(dev_frac + test_frac - 1.0) < 1e-6, "dev_frac + test_frac must equal 1.0"
    
    # Get all participant IDs
    pid_to_sub, sub_to_pid = build_pid_mappings()
    all_pids = sorted(pid_to_sub.keys())
    
    # Shuffle PIDs
    rng = np.random.RandomState(seed)
    rng.shuffle(all_pids)
    
    # Split into dev and test
    n_total = len(all_pids)
    n_dev = int(dev_frac * n_total)
    
    dev_pids = all_pids[:n_dev]
    test_pids = all_pids[n_dev:]
    
    # Convert PIDs to subjects
    test_subjects = []
    for pid in test_pids:
        test_subjects.extend(pid_to_sub[pid])
    
    # Create k-fold splits on development set
    folds = []
    fold_size = n_dev // n_folds
    
    for fold_idx in range(n_folds):
        # Determine validation PIDs for this fold
        val_start = fold_idx * fold_size
        if fold_idx == n_folds - 1:
            # Last fold gets remaining PIDs (handles rounding)
            val_end = n_dev
        else:
            val_end = (fold_idx + 1) * fold_size
        
        val_pids = dev_pids[val_start:val_end]
        train_pids = list(dev_pids[:val_start]) + list(dev_pids[val_end:])
        
        # Convert to subjects
        train_subjects = []
        val_subjects = []
        
        for pid in train_pids:
            train_subjects.extend(pid_to_sub[pid])
        for pid in val_pids:
            val_subjects.extend(pid_to_sub[pid])
        
        fold_data = {
            'fold_idx': fold_idx,
            'train_subjects': sorted(train_subjects),
            'val_subjects': sorted(val_subjects),
            'train_pids': sorted(train_pids),
            'val_pids': sorted(val_pids),
            'metadata': {
                'n_train_subjects': len(train_subjects),
                'n_val_subjects': len(val_subjects),
                'n_train_pids': len(train_pids),
                'n_val_pids': len(val_pids),
            }
        }
        folds.append(fold_data)
    
    # Build output dictionary
    splits = {
        'test_subjects': sorted(test_subjects),
        'test_pids': sorted(test_pids),
        'folds': folds,
        'metadata': {
            'seed': seed,
            'dev_frac': dev_frac,
            'test_frac': test_frac,
            'n_folds': n_folds,
            'n_dev_subjects': sum(f['metadata']['n_train_subjects'] + f['metadata']['n_val_subjects'] for f in folds) // n_folds * n_folds,
            'n_test_subjects': len(test_subjects),
            'n_dev_pids': len(dev_pids),
            'n_test_pids': len(test_pids),
        }
    }
    
    # Save to disk
    SPLITS_CV_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_CV_FILE, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✓ Created and saved CV splits to {SPLITS_CV_FILE}")
    print(f"  Development set: {len(dev_pids)} PIDs across {n_folds} folds")
    print(f"  Test set: {len(test_pids)} PIDs ({len(test_subjects)} subjects)")
    for fold_idx, fold in enumerate(folds):
        print(f"    Fold {fold_idx}: train {fold['metadata']['n_train_pids']} PIDs, val {fold['metadata']['n_val_pids']} PIDs")
    
    return splits


def load_cv_splits() -> Dict:
    """
    Load k-fold CV splits from disk.
    Creates them if they don't exist.
    
    Returns
    -------
    dict
        Dictionary with test_subjects, test_pids, and folds.
    """
    if not SPLITS_CV_FILE.exists():
        print("No existing CV splits found. Creating new splits...")
        return create_cv_splits()
    
    with open(SPLITS_CV_FILE, 'r') as f:
        splits = json.load(f)
    
    return splits


def get_fold_splits(fold_idx: int) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Get train and validation subjects/PIDs for a specific fold.
    
    Parameters
    ----------
    fold_idx : int
        Fold index (0 to n_folds-1).
    
    Returns
    -------
    train_subjects, val_subjects, train_pids, val_pids : Tuple[List[str], List[str], List[str], List[str]]
    """
    splits = load_cv_splits()
    fold = splits['folds'][fold_idx]
    
    return (
        fold['train_subjects'],
        fold['val_subjects'],
        fold['train_pids'],
        fold['val_pids'],
    )


def get_test_subjects_cv() -> List[str]:
    """Get test subject IDs (held-out, use only for final evaluation!)."""
    return load_cv_splits()['test_subjects']


def get_test_pids_cv() -> List[str]:
    """Get test PIDs (held-out, use only for final evaluation!)."""
    return load_cv_splits()['test_pids']


def get_n_folds() -> int:
    """Get the number of folds."""
    return load_cv_splits()['metadata']['n_folds']


def print_cv_split_info():
    """Print information about the current CV splits."""
    splits = load_cv_splits()
    metadata = splits['metadata']
    
    print("\n" + "="*70)
    print("BOAS Dataset K-Fold CV Splits")
    print("="*70)
    print(f"Split file: {SPLITS_CV_FILE}")
    print(f"Random seed: {metadata['seed']}")
    print(f"\nSplit configuration:")
    print(f"  Development set: {metadata['dev_frac']:.0%} ({metadata['n_dev_pids']} PIDs)")
    print(f"  Test set: {metadata['test_frac']:.0%} ({metadata['n_test_pids']} PIDs, {len(splits['test_subjects'])} subjects)")
    print(f"  Number of folds: {metadata['n_folds']}")
    
    print(f"\nFold details:")
    for fold in splits['folds']:
        fold_meta = fold['metadata']
        print(f"  Fold {fold['fold_idx']}: "
              f"train {fold_meta['n_train_pids']} PIDs ({fold_meta['n_train_subjects']} subjects) | "
              f"val {fold_meta['n_val_pids']} PIDs ({fold_meta['n_val_subjects']} subjects)")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Create splits if they don't exist
    splits = create_cv_splits(force_recreate=False)
    print_cv_split_info()
    
    # Example: print first fold
    train_subs, val_subs, train_pids, val_pids = get_fold_splits(0)
    print(f"\nExample - Fold 0:")
    print(f"  Train subjects: {len(train_subs)}")
    print(f"  Val subjects: {len(val_subs)}")
    
    # Example: get test set
    test_subs = get_test_subjects_cv()
    print(f"\nTest set (held-out):")
    print(f"  Test subjects: {len(test_subs)}")
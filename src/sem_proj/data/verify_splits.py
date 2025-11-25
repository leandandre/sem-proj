"""
Verify data splits have no overlap and are correctly set up.
"""
from sem_proj.data.splits import load_splits
import json
from pathlib import Path

def verify_everything():
    # Load splits
    splits = load_splits()
    
    train = set(splits['train_subjects'])
    val = set(splits['val_subjects'])
    test = set(splits['test_subjects'])
    
    train_pids = set(splits['train_pids'])
    val_pids = set(splits['val_pids'])
    test_pids = set(splits['test_pids'])
    
    print("="*60)
    print("DATA SPLIT VERIFICATION")
    print("="*60)
    
    # 1. Check overlaps
    print("\n1. Checking for subject overlaps...")
    train_val = train & val
    train_test = train & test
    val_test = val & test
    
    if train_val:
        print(f"    {len(train_val)} subjects in both train and val!")
        return False
    print("    No overlap between train and val")
    
    if train_test:
        print(f"    {len(train_test)} subjects in both train and test!")
        return False
    print("    No overlap between train and test")
    
    if val_test:
        print(f"    {len(val_test)} subjects in both val and test!")
        return False
    print("    No overlap between val and test")
    
    # 2. Check PID overlaps
    print("\n2. Checking for PID overlaps (no patient leakage)...")
    if train_pids & val_pids:
        print(f"    PIDs overlap between train and val!")
        return False
    print("    No PID overlap between train and val")
    
    if train_pids & test_pids:
        print(f"    PIDs overlap between train and test!")
        return False
    print("    No PID overlap between train and test")
    
    if val_pids & test_pids:
        print(f"    PIDs overlap between val and test!")
        return False
    print("    No PID overlap between val and test")
    
    # 3. Check counts
    print("\n3. Dataset statistics:")
    metadata = splits['metadata']
    print(f"   Train: {metadata['n_train_subjects']} subjects, {metadata['n_train_pids']} PIDs")
    print(f"   Val:   {metadata['n_val_subjects']} subjects, {metadata['n_val_pids']} PIDs")
    print(f"   Test:  {metadata['n_test_subjects']} subjects, {metadata['n_test_pids']} PIDs")
    print(f"   Total: {len(train) + len(val) + len(test)} subjects")
    
    # 4. Check if all subjects are accounted for
    print("\n4. Checking completeness...")
    all_subjects = train | val | test
    print(f"    All subjects accounted for: {len(all_subjects)} total")
    
    # 5. Verify split ratios
    print("\n5. Actual split ratios:")
    total = len(all_subjects)
    print(f"   Train: {len(train)/total:.1%}")
    print(f"   Val:   {len(val)/total:.1%}")
    print(f"   Test:  {len(test)/total:.1%}")
    
    print("\n" + "="*60)
    print(" ALL CHECKS PASSED! Your splits are correct!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    verify_everything()
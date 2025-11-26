"""
Check sleep stage class distribution in the BOAS dataset.
"""
from collections import Counter
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sem_proj.data.datasets import BoasDataset
from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.data.splits import get_train_subjects, get_val_subjects, get_test_subjects


def analyze_class_distribution(subjects, split_name, use_cache=True):
    """Analyze class distribution for a given split."""
    
    # Use minimal preprocessing for speed
    config = PreprocessingConfig.no_preprocessing()
    
    print(f"\nLoading {split_name} data ({len(subjects)} subjects)...")
    dataset = BoasDataset(
        subjects=subjects,
        mode="headband",
        preprocess_config=config,
        use_cache=use_cache
    )
    
    # Count all labels
    all_labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        all_labels.append(label.item())
    
    counter = Counter(all_labels)
    total = len(all_labels)
    
    class_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    
    print(f"\n{'='*80}")
    print(f"{split_name.upper()} SET - Sleep Stage Distribution")
    print(f"{'='*80}")
    print(f"Total samples: {total:,}")
    print(f"\n{'Stage':<10} {'Count':<12} {'Percentage':<12} {'Visualization'}")
    print(f"{'-'*80}")
    
    for class_idx in sorted(counter.keys()):
        count = counter[class_idx]
        percentage = (count / total) * 100
        bar_length = int(percentage / 2)  # Scale for console
        bar = '█' * bar_length
        class_name = class_names.get(class_idx, f'Unknown_{class_idx}')
        print(f"{class_name:<10} {count:<12,} {percentage:>6.2f}%      {bar}")
    
    print(f"{'='*80}\n")
    
    # Compute imbalance ratio
    max_count = max(counter.values())
    min_count = min(counter.values())
    imbalance_ratio = max_count / min_count
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}x (max/min)")
    print(f"Most common: {class_names[max(counter, key=counter.get)]} ({max_count:,} samples)")
    print(f"Least common: {class_names[min(counter, key=counter.get)]} ({min_count:,} samples)")
    
    return counter


def main():
    print(f"\n{'='*80}")
    print(f"BOAS DATASET - CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    # Get splits
    train_subs = get_train_subjects()
    val_subs = get_val_subjects()
    test_subs = get_test_subjects()
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_subs)} subjects")
    print(f"  Val:   {len(val_subs)} subjects")
    print(f"  Test:  {len(test_subs)} subjects")
    
    # Analyze each split
    train_dist = analyze_class_distribution(train_subs, "Training", use_cache=True)
    val_dist = analyze_class_distribution(val_subs, "Validation", use_cache=True)
    test_dist = analyze_class_distribution(test_subs, "Test", use_cache=True)
    
    # Combined statistics
    print(f"\n{'='*80}")
    print(f"COMBINED STATISTICS - COUNTS AND PERCENTAGES")
    print(f"{'='*80}")
    
    class_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    
    # Calculate totals
    train_total = sum(train_dist.values())
    val_total = sum(val_dist.values())
    test_total = sum(test_dist.values())
    overall_total = train_total + val_total + test_total
    
    print(f"\n{'Stage':<10} {'Train Count':<15} {'Train %':<12} {'Val Count':<15} {'Val %':<12} {'Test Count':<15} {'Test %':<12}")
    print(f"{'-'*110}")
    
    for class_idx in range(5):
        class_name = class_names[class_idx]
        
        train_count = train_dist.get(class_idx, 0)
        val_count = val_dist.get(class_idx, 0)
        test_count = test_dist.get(class_idx, 0)
        
        train_pct = (train_count / train_total) * 100 if train_total > 0 else 0
        val_pct = (val_count / val_total) * 100 if val_total > 0 else 0
        test_pct = (test_count / test_total) * 100 if test_total > 0 else 0
        
        print(f"{class_name:<10} {train_count:<15,} {train_pct:>6.2f}%      "
              f"{val_count:<15,} {val_pct:>6.2f}%      "
              f"{test_count:<15,} {test_pct:>6.2f}%")
    
    print(f"{'-'*110}")
    print(f"{'TOTAL':<10} {train_total:<15,} {'100.00%':<12} "
          f"{val_total:<15,} {'100.00%':<12} "
          f"{test_total:<15,} {'100.00%':<12}")
    
    # Overall combined statistics
    print(f"\n{'='*80}")
    print(f"OVERALL COMBINED (All Splits)")
    print(f"{'='*80}")
    
    print(f"\n{'Stage':<10} {'Total Count':<15} {'Percentage':<12} {'Visualization'}")
    print(f"{'-'*80}")
    
    for class_idx in range(5):
        class_name = class_names[class_idx]
        
        combined_count = train_dist.get(class_idx, 0) + val_dist.get(class_idx, 0) + test_dist.get(class_idx, 0)
        combined_pct = (combined_count / overall_total) * 100 if overall_total > 0 else 0
        bar_length = int(combined_pct / 2)
        bar = '█' * bar_length
        
        print(f"{class_name:<10} {combined_count:<15,} {combined_pct:>6.2f}%      {bar}")
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<10} {overall_total:<15,} {'100.00%':<12}")
    print(f"{'='*80}\n")
    
    # Calculate overall imbalance
    combined_counts = [
        train_dist.get(i, 0) + val_dist.get(i, 0) + test_dist.get(i, 0)
        for i in range(5)
    ]
    max_combined = max(combined_counts)
    min_combined = min(combined_counts)
    overall_imbalance = max_combined / min_combined if min_combined > 0 else float('inf')
    
    print(f"Overall class imbalance ratio: {overall_imbalance:.2f}x (max/min)")
    print(f"Most common overall: {class_names[combined_counts.index(max_combined)]} ({max_combined:,} samples)")
    print(f"Least common overall: {class_names[combined_counts.index(min_combined)]} ({min_combined:,} samples)")
    
    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*80}")
    print("  ✓ If N1 < 10%: This is NORMAL (N1 is shortest sleep stage)")
    print("  ✓ If N2 > 40%: This is NORMAL (N2 is most common sleep stage)")
    print("  ⚠ If imbalance > 10x: Consider class weights in loss function")
    print("  ✓ Wake should be ~15-25% (including awake periods)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
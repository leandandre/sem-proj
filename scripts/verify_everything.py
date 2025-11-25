"""Run this to verify everything is set up correctly."""
from sem_proj.data.splits import print_split_info, load_splits
from sem_proj.data.preprocessing import PreprocessingConfig, get_expected_seq_length
from pathlib import Path

def verify_setup():
    print("="*60)
    print("SETUP VERIFICATION")
    print("="*60)
    
    # 1. Check splits
    print("\n1. Checking data splits...")
    try:
        splits = load_splits()
        print_split_info()
        
        # Verify no overlap
        train = set(splits['train_subjects'])
        val = set(splits['val_subjects'])
        test = set(splits['test_subjects'])
        
        assert len(train & val) == 0, "Train/val overlap!"
        assert len(train & test) == 0, "Train/test overlap!"
        assert len(val & test) == 0, "Val/test overlap!"
        print("✅ No subject overlap detected")
        
    except Exception as e:
        print(f"❌ Split verification failed: {e}")
        return False
    
    # 2. Check YAML configs - UPDATED names
    print("\n2. Checking YAML configs...")
    CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "preprocess"
    
    configs_to_check = [
        "no_preprocess.yaml",
        "notch_bandpass.yaml",
        "notch_bandpass_znorm.yaml",
        "notch_bandpass_resample.yaml",
        "notch_bandpass_resample_znorm.yaml",  # UPDATED
        "only_znorm.yaml"
    ]
    
    for config_name in configs_to_check:
        config_path = CONFIG_DIR / config_name
        if not config_path.exists():
            print(f"❌ Missing config: {config_path}")
            return False
        
        try:
            config = PreprocessingConfig.from_yaml(config_path)
            seq_len = get_expected_seq_length(config)
            print(f"✅ {config_name}: seq_length={seq_len}")
        except Exception as e:
            print(f"❌ Error loading {config_name}: {e}")
            return False
    
    # 3. Check one subject can load
    print("\n3. Testing data loading...")
    try:
        from sem_proj.data.datasets import BoasDataset
        
        test_config = PreprocessingConfig.from_yaml(CONFIG_DIR / "notch_bandpass_resample_znorm.yaml")
        test_ds = BoasDataset(
            subjects=splits['train_subjects'][:1],  # Just one subject
            mode="headband",
            preprocess_config=test_config
        )
        
        x, y = test_ds[0]
        print(f"✅ Successfully loaded test epoch: shape={x.shape}, label={y}")
        
        expected_seq_len = get_expected_seq_length(test_config)
        assert x.shape[1] == expected_seq_len, f"Sequence length mismatch! Got {x.shape[1]}, expected {expected_seq_len}"
        print(f"✅ Sequence length matches expected: {expected_seq_len}")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED! Ready to train!")
    print("="*60)
    print("\nYou can now run:")
    print("  python -m sem_proj.training.epoch_models")
    print("Or train all configs:")
    print("  python scripts/train_all_configs.py")
    return True

if __name__ == "__main__":
    success = verify_setup()
    exit(0 if success else 1)
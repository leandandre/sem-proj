"""
Train model with all preprocessing configurations.
Simply runs the training script multiple times by editing CONFIG_NAME.
"""
import subprocess
from pathlib import Path
import re

# CONFIGS to train
CONFIGS = [
    'no_preprocess',
    'notch_bandpass',
    'notch_bandpass_znorm',
    'notch_bandpass_resample',
    'notch_bandpass_resample_znorm',
    'only_znorm'
]

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-3

# Path to the training script
TRAINING_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "sem_proj" / "training" / "epoch_models.py"

def update_config_in_script(config_name: str):
    """Temporarily modify the training script to use specified config."""
    
    # Read the script
    with open(TRAINING_SCRIPT, 'r') as f:
        content = f.read()
    
    # Replace the CONFIG_NAME line
    pattern = r'CONFIG_NAME = "[^"]*"'
    replacement = f'CONFIG_NAME = "{config_name}"'
    
    new_content = re.sub(pattern, replacement, content)
    
    # Also update hyperparameters if needed
    new_content = re.sub(r'NUM_EPOCHS = \d+', f'NUM_EPOCHS = {EPOCHS}', new_content)
    new_content = re.sub(r'BATCH_SIZE = \d+', f'BATCH_SIZE = {BATCH_SIZE}', new_content)
    new_content = re.sub(r'LEARNING_RATE = [0-9.e-]+', f'LEARNING_RATE = {LR}', new_content)
    
    # Write back
    with open(TRAINING_SCRIPT, 'w') as f:
        f.write(new_content)

def main():
    print("="*60)
    print("TRAINING ALL PREPROCESSING CONFIGURATIONS")
    print("="*60)
    print(f"Configs to train: {len(CONFIGS)}")
    print(f"Epochs per config: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print("="*60)
    print("\nConfigurations:")
    for i, config in enumerate(CONFIGS, 1):
        print(f"  {i}. {config}")
    print("="*60)
    
    # Save original script content
    with open(TRAINING_SCRIPT, 'r') as f:
        original_content = f.read()
    
    try:
        for i, config in enumerate(CONFIGS, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(CONFIGS)}] Training with config: {config}")
            print(f"{'='*60}\n")
            
            # Update the config in the training script
            update_config_in_script(config)
            
            # Run the training script
            cmd = ['python', '-m', 'sem_proj.training.epoch_models']
            
            try:
                subprocess.run(cmd, check=True)
                print(f"\n Successfully trained with {config}")
            except subprocess.CalledProcessError as e:
                print(f"\n Failed to train with {config}: {e}")
                user_input = input("Continue with remaining configs? (y/n): ")
                if user_input.lower() != 'y':
                    print("Training interrupted by user.")
                    break
                continue
    
    finally:
        # Restore original script content
        print("\nRestoring original training script...")
        with open(TRAINING_SCRIPT, 'w') as f:
            f.write(original_content)
        print("✓ Restored")
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE!")
    print("="*60)
    print("\nView results in TensorBoard:")
    print("  tensorboard --logdir logs")
    print("\nCheckpoint locations:")
    for config in CONFIGS:
        print(f"  checkpoints/transformer_{config}_v1/")

if __name__ == "__main__":
    main()
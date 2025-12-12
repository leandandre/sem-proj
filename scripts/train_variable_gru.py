"""
Entry point script for training variable-length GRU sequence classifier.
TRAINING ON FULL TRAINING SET. DID NOT REALLY USE THAT SCRIPT.
finetune_variable_gru.py HANDLES THE ALL CASES (also fraction=1.0).

This script implements:
- Extract sequences from continuous recordings until artifact/disconnection
- Train GRU on variable-length sequences
- Optionally use pretrained SSL encoder

Usage:
    python scripts/train_variable_gru.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.training.sequence_models_variable import train_variable_gru

# Path to configs
CONFIG_DIR = PROJECT_ROOT / "configs" / "preprocess"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def main():
    """
    Main training script for variable-length GRU.
    
    Adjust parameters below for your experiments.
    """
    
    # Load preprocessing config
    config = PreprocessingConfig.from_yaml(
        CONFIG_DIR / "notch_bandpass_resample_znorm.yaml"
    )
    
    # Optional: Path to pretrained SSL encoder checkpoint
    # Set to None to train from scratch
    ssl_encoder_checkpoint = None
    # Example: ssl_encoder_checkpoint = CHECKPOINT_DIR / "ssl_transformer_v1" / "best_model_hb.pt"
    
    print("="*80)
    print("Variable-Length GRU Training")
    print("="*80)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Config: {config.to_dict()}")
    print(f"SSL checkpoint: {ssl_encoder_checkpoint}")
    print("="*80 + "\n")
    
    # Train variable-length GRU
    best_f1 = train_variable_gru(
        # Training params
        num_epochs=100,
        batch_size=1,                    # Small batch for variable-length
        min_seq_len=5,                   # Minimum sequence length
        lr_encoder=1e-5,                 # Low LR if fine-tuning encoder
        lr_gru=1e-4,                     # Higher LR for GRU; if fully_supervised=True this lr is used for all
        mode="headband",                 # "headband" or "psg"
        experiment_name="variable_gru_headband_v1",
        preprocess_config=config,
        use_cache=True,
        
        # Encoder params
        encoder_checkpoint=ssl_encoder_checkpoint,
        freeze_encoder=False,            # Set True to freeze encoder
        fully_supervised=True,           # True if no pretrained encoder, use lr_gru as learning rate for all
        d_model=128,             # Must match pretrained encoder, if used ({96, 128, 160})
        nhead=4,
        num_layers_encoder=3,
        dim_feedforward=512,
        dropout_encoder=0.2,
        target_tokens=240,
        
        # GRU params
        gru_hidden=128,
        gru_layers=2,
        gru_bidirectional=False,
        gru_dropout=0.2,
        use_attention=False,             # Set True for attention-augmented GRU
        
        # Training settings
        class_weighted_loss=True,        # Balance class distribution
        gradient_clip=5.0,
        early_stopping_patience=12,
        num_classes=5,
    )
    
    print(f"\n{'='*80}")
    print(f"Training complete! Best validation F1: {best_f1:.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
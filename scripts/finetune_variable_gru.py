"""
Fine-tuning script for variable-length GRU with controlled labeled data fractions.

This script implements a two-stage fine-tuning strategy:
    
    Stage 1: Hyperparameter Tuning
    -------------------------------
    - Train on fraction p of train_subjects
    - Validate on full val_subjects
    - Purpose: Find optimal hyperparameters with limited labeled data
    
    Stage 2: Final Evaluation
    --------------------------
    - Train on fraction p of (train_subjects + val_subjects)
    - Test on full test_subjects
    - Purpose: Final model evaluation using all available training data
        
Example workflow:
    - Set fraction = 0.1 (10% of labeled data)
    - Run Stage 1 to tune hyperparameters
    - Run Stage 2 with best hyperparameters for final evaluation
    - Repeat with different fractions: 0.25, 0.5, 1.0, etc.
"""

import sys
from pathlib import Path
import random
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.data.splits import get_train_subjects, get_val_subjects, get_test_subjects
from sem_proj.training.sequence_models_variable import train_variable_gru

# Path to configs
CONFIG_DIR = PROJECT_ROOT / "configs" / "preprocess"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def sample_subjects(subjects: List[str], fraction: float, seed: int = 42) -> List[str]:
    """
    Randomly sample a fraction of subjects.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject IDs to sample from.
    fraction : float
        Fraction of subjects to sample (0.0 to 1.0).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    List[str]
        Sampled subject IDs, sorted for consistency.
    """
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be between 0.0 and 1.0, got {fraction}")
    
    if fraction == 1.0:
        return subjects
    
    # Set seed for reproducibility
    rng = random.Random(seed)
    
    # Calculate number of subjects to sample
    n_sample = max(1, int(len(subjects) * fraction))  # At least 1 subject
    
    # Sample subjects
    sampled = rng.sample(subjects, n_sample)
    
    return sorted(sampled)


def run_stage1_hyperparameter_tuning(
    fraction: float = 0.1,
    seed: int = 42,
    ssl_checkpoint: Path = None,
    num_epochs: int = 100,
    batch_size: int = 1,
    lr_gru: float = 1e-4,
    experiment_name: str = "finetune_stage1_p0.1",
):
    """
    Stage 1: Train on fraction p of train_subjects, validate on full val_subjects.
    
    This stage is for hyperparameter tuning with limited labeled data.
    
    Parameters
    ----------
    fraction : float
        Fraction of training data to use (e.g., 0.1 = 10%).
    seed : int
        Random seed for subject sampling.
    ssl_checkpoint : Path, optional
        Path to pretrained SSL encoder checkpoint.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    lr_gru : float
        Learning rate for GRU layers.
    experiment_name : str
        Name for this experiment.
    
    Returns
    -------
    float
        Best validation F1 score.
    """
    print("\n" + "="*80)
    print("STAGE 1: HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Strategy: Train on {fraction*100:.0f}% of train_subjects, validate on full val_subjects")
    print("="*80 + "\n")
    
    # Load full splits
    train_subjects = get_train_subjects()
    val_subjects = get_val_subjects()
    
    # Sample fraction of training subjects
    sampled_train = sample_subjects(train_subjects, fraction, seed)
    
    print(f"Subject allocation:")
    print(f"  Total train subjects: {len(train_subjects)}")
    print(f"  Sampled train subjects ({fraction*100:.0f}%): {len(sampled_train)}")
    print(f"  Validation subjects (100%): {len(val_subjects)}")
    print(f"  Random seed: {seed}")
    print(f"\nSampled training subjects: {sampled_train[:5]}..." if len(sampled_train) > 5 else f"\nSampled training subjects: {sampled_train}")
    print()
    
    # Load preprocessing config (always this one since it showed best results)
    config = PreprocessingConfig.from_yaml(
        CONFIG_DIR / "notch_bandpass_resample_znorm.yaml"
    )
    
    # Determine if using SSL checkpoint or fully supervised
    fully_supervised = (ssl_checkpoint is None)
    freeze_encoder = False  # For fine-tuning, we typically don't freeze
    
    print(f"Training configuration:")
    print(f"  Mode: {'Fully supervised' if fully_supervised else 'Fine-tuning from SSL'}")
    print(f"  SSL checkpoint: {ssl_checkpoint}")
    print(f"  Freeze encoder: {freeze_encoder}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate (GRU): {lr_gru}")
    print()
    
    # Train model
    best_f1 = train_variable_gru(
        # Training params
        num_epochs=num_epochs,
        batch_size=batch_size,
        min_seq_len=5,
        lr_encoder=1e-5 if not fully_supervised else lr_gru,
        lr_gru=lr_gru,
        mode="headband",
        experiment_name=experiment_name,
        preprocess_config=config,
        use_cache=True,
        
        # Encoder params
        encoder_checkpoint=ssl_checkpoint,
        freeze_encoder=freeze_encoder,
        fully_supervised=fully_supervised,
        d_model=128,
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
        use_attention=False,
        
        # Training settings
        class_weighted_loss=True,
        gradient_clip=5.0,
        early_stopping_patience=12,
        num_classes=5,
        
        # Custom data splits - THIS IS THE KEY PART
        train_subjects=sampled_train,
        val_subjects=val_subjects,
    )
    
    print("\n" + "="*80)
    print(f"STAGE 1 COMPLETE - Best Validation F1: {best_f1:.4f}")
    print("="*80 + "\n")
    
    return best_f1


def run_stage2_final_evaluation(
    fraction: float = 0.1,
    seed: int = 42,
    ssl_checkpoint: Path = None,
    num_epochs: int = 100,
    batch_size: int = 1,
    lr_gru: float = 1e-4,
    experiment_name: str = "finetune_stage2_p0.1",
):
    """
    Stage 2: Train on fraction p of (train + val), test on full test_subjects.
    
    This stage is for final evaluation using all available training data.
    
    Parameters
    ----------
    fraction : float
        Fraction of combined train+val data to use.
    seed : int
        Random seed for subject sampling.
    ssl_checkpoint : Path, optional
        Path to pretrained SSL encoder checkpoint.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    lr_gru : float
        Learning rate for GRU layers.
    experiment_name : str
        Name for this experiment.
    
    Returns
    -------
    float
        Best validation F1 score (using test set as validation).
    """
    print("\n" + "="*80)
    print("STAGE 2: FINAL EVALUATION")
    print("="*80)
    print(f"Strategy: Train on {fraction*100:.0f}% of (train + val), test on full test_subjects")
    print("="*80 + "\n")
    
    # Load full splits
    train_subjects = get_train_subjects()
    val_subjects = get_val_subjects()
    test_subjects = get_test_subjects()
    
    # Combine train and val for training
    combined_subjects = train_subjects + val_subjects
    
    # Sample fraction of combined subjects
    sampled_train = sample_subjects(combined_subjects, fraction, seed)
    
    print(f"Subject allocation:")
    print(f"  Total train subjects: {len(train_subjects)}")
    print(f"  Total val subjects: {len(val_subjects)}")
    print(f"  Combined subjects: {len(combined_subjects)}")
    print(f"  Sampled for training ({fraction*100:.0f}%): {len(sampled_train)}")
    print(f"  Test subjects (100%): {len(test_subjects)}")
    print(f"  Random seed: {seed}")
    print(f"\nSampled training subjects: {sampled_train[:5]}..." if len(sampled_train) > 5 else f"\nSampled training subjects: {sampled_train}")
    print()
    
    # Load preprocessing config
    config = PreprocessingConfig.from_yaml(
        CONFIG_DIR / "notch_bandpass_resample_znorm.yaml"
    )
    
    # Determine if using SSL checkpoint or fully supervised
    fully_supervised = (ssl_checkpoint is None)
    freeze_encoder = False
    
    print(f"Training configuration:")
    print(f"  Mode: {'Fully supervised' if fully_supervised else 'Fine-tuning from SSL'}")
    print(f"  SSL checkpoint: {ssl_checkpoint}")
    print(f"  Freeze encoder: {freeze_encoder}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate (GRU): {lr_gru}")
    print()
    
    # Train model - NOTE: We use test_subjects as "val_subjects" for final evaluation
    best_f1 = train_variable_gru(
        # Training params
        num_epochs=num_epochs,
        batch_size=batch_size,
        min_seq_len=5,
        lr_encoder=1e-5 if not fully_supervised else lr_gru,
        lr_gru=lr_gru,
        mode="headband",
        experiment_name=experiment_name,
        preprocess_config=config,
        use_cache=True,
        
        # Encoder params
        encoder_checkpoint=ssl_checkpoint,
        freeze_encoder=freeze_encoder,
        fully_supervised=fully_supervised,
        d_model=128,
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
        use_attention=False,
        
        # Training settings
        class_weighted_loss=True,
        gradient_clip=5.0,
        early_stopping_patience=12,
        num_classes=5,
        
        # Custom data splits - THIS IS THE KEY PART
        # Train on sampled (train+val), "validate" on test
        train_subjects=sampled_train,
        val_subjects=test_subjects,  # Using test as validation for final eval
    )
    
    print("\n" + "="*80)
    print(f"STAGE 2 COMPLETE - Test F1: {best_f1:.4f}")
    print("="*80 + "\n")
    
    return best_f1


def main():
    """
    Main entry point for fine-tuning experiments.
    
    Demonstrates both Stage 1 and Stage 2 approaches.
    Uncomment the stage you want to run.
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Data fraction to use (0.1 = 10%, 0.25 = 25%, 0.5 = 50%, 1.0 = 100%)
    FRACTION = 1.0
    
    # Random seed for reproducibility
    SEED = 42
    
    # Optional: Path to pretrained SSL encoder
    # Set to None for fully supervised training from scratch
    SSL_CHECKPOINT = None
    # Example: SSL_CHECKPOINT = CHECKPOINT_DIR / "ssl_transformer_v1" / "best_model_hb.pt"
    
    # Training hyperparameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 1
    LR_GRU = 1e-4
        
    print("\n" + "="*80)
    print("FINE-TUNING WITH CONTROLLED LABELED DATA FRACTIONS")
    print("="*80)
    print(f"Configuration:")
    print(f"  Data fraction: {FRACTION*100:.0f}%")
    print(f"  Random seed: {SEED}")
    print(f"  SSL checkpoint: {SSL_CHECKPOINT}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR_GRU}")
    print("="*80)
    

    # Uncomment to run Stage 1

    stage1_f1 = run_stage1_hyperparameter_tuning(
        fraction=FRACTION,
        seed=SEED,
        ssl_checkpoint=SSL_CHECKPOINT,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr_gru=LR_GRU,
        experiment_name=f"finetune_stage1_p{FRACTION}",
    )
    

    # Uncomment to run Stage 2 (after completing Stage 1 and tuning hyperparameters)

    # stage2_f1 = run_stage2_final_evaluation(
    #     fraction=FRACTION,
    #     seed=SEED,
    #     ssl_checkpoint=SSL_CHECKPOINT,
    #     num_epochs=NUM_EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     lr_gru=LR_GRU,
    #     experiment_name=f"finetune_stage2_p{FRACTION}",
    # )
        
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE")
    print("="*80)
    print(f"Data fraction: {FRACTION*100:.0f}%")
    print(f"Stage 1 (Hyperparameter Tuning) - Best Val F1: {stage1_f1:.4f}")
    # print(f"Stage 2 (Final Evaluation) - Test F1: {stage2_f1:.4f}")
    print("="*80 + "\n")
    


if __name__ == "__main__":
    main()

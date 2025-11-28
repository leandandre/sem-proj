"""
Analyze trained model checkpoint.
Shows configuration, performance metrics, and visualizations.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from pprint import pprint

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sem_proj.models.model_factory import EpochTransformer

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def load_checkpoint(experiment_name: str, checkpoint_name: str = "best_model.pt"):
    """Load a saved checkpoint."""
    checkpoint_path = CHECKPOINT_DIR / experiment_name / checkpoint_name
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}\n")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    return checkpoint


def print_checkpoint_info(checkpoint: dict, experiment_name: str):
    """Print detailed information about the checkpoint."""
    print("=" * 80)
    print(f"CHECKPOINT ANALYSIS: {experiment_name}")
    print("=" * 80)
    
    # Training progress
    print(f"\n TRAINING PROGRESS")
    print("-" * 80)
    print(f"Stopped at epoch: {checkpoint['epoch'] + 1}")
    
    # Performance metrics
    print(f"\n VALIDATION PERFORMANCE")
    print("-" * 80)
    print(f"Validation Loss:     {checkpoint['val_loss']:.4f}")
    print(f"Validation Accuracy: {checkpoint['val_accuracy']:.4f} ({checkpoint['val_accuracy']*100:.2f}%)")
    print(f"Macro F1 Score:      {checkpoint['val_macro_f1']:.4f}")
    
    # Per-class F1 scores
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    per_class_f1 = checkpoint['val_per_class_f1']
    
    print(f"\n PER-CLASS F1 SCORES")
    print("-" * 80)
    print(f"{'Class':<10} {'F1 Score':<12} {'Bar Chart'}")
    print("-" * 80)
    
    for class_name, f1 in zip(class_names, per_class_f1):
        bar_length = int(f1 * 50)
        bar = '█' * bar_length
        print(f"{class_name:<10} {f1:.4f}       {bar}")
    
    # Model architecture
    print(f"\n MODEL ARCHITECTURE")
    print("-" * 80)
    hyperparams = checkpoint['hyperparameters']
    train_cfg = checkpoint.get('training_config', {})  # Get early for token info
    
    print(f"Model type:          {train_cfg.get('model_type', 'Unknown')}")
    print(f"Input channels:      {hyperparams['input_channels']}")
    print(f"Sequence length:     {hyperparams['seq_length']}")
    
    # ADD THESE LINES:
    print(f"Max tokens:          {train_cfg.get('max_tokens', hyperparams.get('max_tokens', 'N/A'))}")
    print(f"Patch size:          {train_cfg.get('patch_size', 'N/A')}")
    print(f"Final seq length:    {train_cfg.get('final_seq_length', 'N/A')} (tokens fed to transformer)")
    
    print(f"Embedding dim:       {hyperparams['d_model']}")
    print(f"Attention heads:     {hyperparams['nhead']}")
    print(f"Transformer layers:  {hyperparams['num_layers']}")
    print(f"Feedforward dim:     {hyperparams['dim_feedforward']}")
    print(f"Dropout:             {hyperparams['dropout']}")
    print(f"Number of classes:   {hyperparams['num_classes']}")
    
    # Training configuration
    if 'training_config' in checkpoint:
        print(f"\n TRAINING CONFIGURATION")
        print("-" * 80)
        print(f"Batch size:          {train_cfg.get('batch_size', 'N/A')}")
        print(f"Learning rate:       {train_cfg.get('learning_rate', 'N/A')}")
        print(f"Optimizer:           {train_cfg.get('optimizer', 'N/A')}")
        print(f"LR Scheduler:        {train_cfg.get('scheduler', 'N/A')}")
        print(f"  - Patience:        {train_cfg.get('scheduler_patience', 'N/A')}")
        print(f"  - Factor:          {train_cfg.get('scheduler_factor', 'N/A')}")
        print(f"Early stop patience: {train_cfg.get('early_stop_patience', 'N/A')}")
        print(f"Class weighted loss: {train_cfg.get('class_weighted_loss', 'N/A')}")
        print(f"Used cache:          {train_cfg.get('use_cache', 'N/A')}")
        print(f"Trainable params:    {train_cfg.get('num_trainable_params', 'N/A'):,}")
        print(f"Training samples:    {train_cfg.get('train_samples', 'N/A'):,}")
        print(f"Validation samples:  {train_cfg.get('val_samples', 'N/A'):,}")
    
    # Class weights (if used)
    if checkpoint.get('class_weights') is not None:
        print(f"\n  CLASS WEIGHTS")
        print("-" * 80)
        class_weights = checkpoint['class_weights']
        for class_name, weight in zip(class_names, class_weights):
            print(f"{class_name:<10} {weight:.4f}")
    
    # Preprocessing configuration
    print(f"\n PREPROCESSING CONFIGURATION")
    print("-" * 80)
    preprocess = checkpoint['preprocessing']
    for key, value in preprocess.items():
        print(f"{key:<25} {value}")
    
    print("=" * 80)


def plot_per_class_f1(checkpoint: dict, experiment_name: str, save_path: Path = None):
    """Create bar plot of per-class F1 scores."""
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    per_class_f1 = checkpoint['val_per_class_f1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    bars = ax.bar(class_names, per_class_f1, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, f1 in zip(bars, per_class_f1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sleep Stage', fontsize=14, fontweight='bold')
    ax.set_title(f'Per-Class F1 Scores\n{experiment_name}', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=checkpoint['val_macro_f1'], color='red', linestyle='--', 
               linewidth=2, label=f'Macro F1: {checkpoint["val_macro_f1"]:.3f}')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to: {save_path}")
    
    plt.show()


def plot_confusion_style_summary(checkpoint: dict, experiment_name: str, save_path: Path = None):
    """Create a summary visualization with multiple subplots."""
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    per_class_f1 = checkpoint['val_per_class_f1']
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Per-class F1 bar chart
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    bars = ax1.bar(class_names, per_class_f1, color=colors, alpha=0.8, edgecolor='black')
    for bar, f1 in zip(bars, per_class_f1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Per-Class Performance - {experiment_name}', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=checkpoint['val_macro_f1'], color='red', linestyle='--', 
                linewidth=2, label=f'Macro F1: {checkpoint["val_macro_f1"]:.3f}')
    ax1.legend()
    
    # 2. Overall metrics (text box)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    metrics_text = f"""
    OVERALL METRICS
    ───────────────────────
    Validation Accuracy: {checkpoint['val_accuracy']:.4f}
    Macro F1 Score:      {checkpoint['val_macro_f1']:.4f}
    Validation Loss:     {checkpoint['val_loss']:.4f}
    
    Best Epoch:          {checkpoint['epoch'] + 1}
    """
    ax2.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    # 3. Model architecture (text box) - ADD TOKEN INFO
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    hp = checkpoint['hyperparameters']
    train_cfg = checkpoint.get('training_config', {})
    
    max_tokens = train_cfg.get('max_tokens', hp.get('max_tokens', 'N/A'))
    final_seq_len = train_cfg.get('final_seq_length', 'N/A')
    patch_size = train_cfg.get('patch_size', 'N/A')
    
    arch_text = f"""
    MODEL ARCHITECTURE
    ───────────────────────
    Input Seq Length:    {hp['seq_length']}
    Patch Size:          {patch_size}
    Final Tokens:        {final_seq_len}
    Max Tokens:          {max_tokens}
    
    Embedding Dim:       {hp['d_model']}
    Attention Heads:     {hp['nhead']}
    Transformer Layers:  {hp['num_layers']}
    Feedforward Dim:     {hp['dim_feedforward']}
    Dropout:             {hp['dropout']}
    """
    ax3.text(0.1, 0.5, arch_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f'Model Checkpoint Analysis\n{experiment_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved summary plot to: {save_path}")
    
    plt.show()


def compare_checkpoints(experiment_names: list[str]):
    """Compare multiple checkpoints side by side."""
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    fig, axes = plt.subplots(1, len(experiment_names), 
                             figsize=(6*len(experiment_names), 6), 
                             sharey=True)
    
    if len(experiment_names) == 1:
        axes = [axes]
    
    for ax, exp_name in zip(axes, experiment_names):
        try:
            checkpoint = load_checkpoint(exp_name)
            per_class_f1 = checkpoint['val_per_class_f1']
            
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
            bars = ax.bar(class_names, per_class_f1, color=colors, alpha=0.8, edgecolor='black')
            
            for bar, f1 in zip(bars, per_class_f1):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f'{exp_name}\nMacro F1: {checkpoint["val_macro_f1"]:.3f}', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=checkpoint['val_macro_f1'], color='red', 
                      linestyle='--', linewidth=2, alpha=0.7)
            
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Checkpoint not found:\n{exp_name}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(exp_name, fontsize=12)
    
    axes[0].set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze model checkpoint')
    parser.add_argument('experiment', type=str, nargs='?', default=None,
                       help='Experiment name (e.g., epochtransformer_notch_bandpass_resample_znorm_v1)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Checkpoint file name (default: best_model.pt)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to disk')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Compare multiple experiments')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple experiments
        print(f"\nComparing {len(args.compare)} experiments...\n")
        compare_checkpoints(args.compare)
    elif args.experiment:
        # Analyze single experiment
        checkpoint = load_checkpoint(args.experiment, args.checkpoint)
        print_checkpoint_info(checkpoint, args.experiment)
        
        # Generate plots
        if args.save_plots:
            plots_dir = PROJECT_ROOT / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            plot_path_1 = plots_dir / f"{args.experiment}_per_class_f1.png"
            plot_path_2 = plots_dir / f"{args.experiment}_summary.png"
            
            plot_per_class_f1(checkpoint, args.experiment, save_path=plot_path_1)
            plot_confusion_style_summary(checkpoint, args.experiment, save_path=plot_path_2)
        else:
            plot_per_class_f1(checkpoint, args.experiment)
            plot_confusion_style_summary(checkpoint, args.experiment)
    else:
        parser.print_help()
        print("\nAvailable checkpoints:")
        for exp_dir in CHECKPOINT_DIR.iterdir():
            if exp_dir.is_dir():
                print(f"  - {exp_dir.name}")


if __name__ == "__main__":
    # Quick test mode (no command line args)
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("  python scripts/analyze_checkpoint.py epochtransformer_notch_bandpass_resample_znorm_v1")
        print("  python scripts/analyze_checkpoint.py epochtransformer_notch_bandpass_resample_znorm_v1 --save-plots")
        print("  python scripts/analyze_checkpoint.py --compare epochtransformer_notch_bandpass_resample_znorm_v1 epochtransformer_notch_bandpass_resample_znorm_v2")
        print("\nAvailable checkpoints:")
        for exp_dir in CHECKPOINT_DIR.iterdir():
            if exp_dir.is_dir():
                print(f"  - {exp_dir.name}")
    else:
        main()
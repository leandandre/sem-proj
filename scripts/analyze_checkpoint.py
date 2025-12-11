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
# CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_leomed"


def load_checkpoint(experiment_name: str, checkpoint_name: str = "best_model.pt"):
    """Load a saved checkpoint."""
    checkpoint_path = CHECKPOINT_DIR / experiment_name / checkpoint_name
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}\n")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    return checkpoint


def scan_all_checkpoints(checkpoint_dir: Path = None, name_filter: str = None):
    """
    Scan all experiments in checkpoint directory and extract macro F1 scores.
    
    Parameters
    ----------
    checkpoint_dir : Path, optional
        Directory to scan (default: CHECKPOINT_DIR).
    name_filter : str, optional
        Only include experiments whose name contains this substring (e.g., "conv1d_v2").
    
    Returns
    -------
    results : list[dict]
        List of dicts with keys: 'experiment', 'macro_f1', 'accuracy', 'loss', 'per_class_f1'
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    results = []
    
    for exp_dir in checkpoint_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Apply name filter
        if name_filter and name_filter not in exp_dir.name:
            continue
        
        best_model_path = exp_dir / "best_model.pt"
        if not best_model_path.exists():
            continue
        
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            
            results.append({
                'experiment': exp_dir.name,
                'macro_f1': checkpoint.get('val_macro_f1', 0.0),
                'accuracy': checkpoint.get('val_accuracy', 0.0),
                'loss': checkpoint.get('val_loss', float('inf')),
                'per_class_f1': checkpoint.get('val_per_class_f1', [0]*5),
                'epoch': checkpoint.get('epoch', -1) + 1,
            })
        except Exception as e:
            print(f"Warning: Failed to load {exp_dir.name}: {e}")
            continue
    
    return results


def print_top_k_models(k: int = 4, checkpoint_dir: Path = None, name_filter: str = None):
    """
    Print top-K models by macro F1 score.
    
    Parameters
    ----------
    k : int
        Number of top models to show.
    checkpoint_dir : Path
        Directory to scan (default: CHECKPOINT_DIR).
    name_filter : str, optional
        Only include experiments whose name contains this substring (e.g., "conv1d_v2").
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    print(f"\n{'='*80}")
    print(f"SCANNING CHECKPOINT DIRECTORY: {checkpoint_dir.name}")
    if name_filter:
        print(f"FILTER: Only experiments containing '{name_filter}'")
    print(f"{'='*80}\n")
    
    results = scan_all_checkpoints(checkpoint_dir, name_filter=name_filter)
    
    if not results:
        print(f"No valid checkpoints found{' matching filter' if name_filter else ''}.")
        return []
    
    print(f"Found {len(results)} matching experiments.")
    
    # Sort by macro F1 (descending)
    results_sorted = sorted(results, key=lambda x: x['macro_f1'], reverse=True)
    
    top_k = results_sorted[:k]
    
    print(f"\n{'='*80}")
    print(f"TOP {k} MODELS BY MACRO F1")
    print(f"{'='*80}\n")
    
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    for rank, result in enumerate(top_k, start=1):
        print(f"#{rank} - {result['experiment']}")
        print("-" * 80)
        print(f"  Macro F1:     {result['macro_f1']:.4f}")
        print(f"  Accuracy:     {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"  Val Loss:     {result['loss']:.4f}")
        print(f"  Best Epoch:   {result['epoch']}")
        print(f"\n  Per-Class F1:")
        for cls_name, f1 in zip(class_names, result['per_class_f1']):
            bar_length = int(f1 * 40)
            bar = '█' * bar_length
            print(f"    {cls_name:<6} {f1:.4f}  {bar}")
        print()
    
    return top_k


def shorten_experiment_name(name: str, max_len: int = 60) -> str:
    """
    Shorten long experiment names intelligently.
    
    Examples
    --------
    'ablate_conv1d_v2_notch_bandpass_resample_znorm_d128_tok480_h8_L6'
    -> 'ablate_conv1d_v2_..._d128_tok480_h8_L6'
    """
    if len(name) <= max_len:
        return name
    
    # Try to keep prefix and important suffix (architecture params)
    parts = name.split('_')
    
    # Keep first 3 parts (e.g., 'ablate', 'conv1d', 'v2')
    prefix = '_'.join(parts[:3])
    
    # Keep last 4 parts if they look like architecture params (e.g., d128, tok480, h8, L6)
    suffix_parts = []
    for part in reversed(parts[-4:]):
        if any(part.startswith(p) for p in ['d', 'tok', 'h', 'L', 'v']):
            suffix_parts.insert(0, part)
        else:
            break
    
    suffix = '_'.join(suffix_parts) if suffix_parts else '_'.join(parts[-2:])
    
    shortened = f"{prefix}_..._{suffix}"
    
    # If still too long, just truncate with ellipsis
    if len(shortened) > max_len:
        return name[:max_len-3] + "..."
    
    return shortened


def plot_top_k_comparison(top_k: list[dict], checkpoint_dir: Path = None, name_filter: str = None, save_path: Path = None):
    """
    Create comparison plot for top-K models.
    
    Parameters
    ----------
    top_k : list[dict]
        List of top models from print_top_k_models().
    checkpoint_dir : Path
        Checkpoint directory (for title).
    name_filter : str, optional
        Filter string (for title).
    save_path : Path, optional
        Path to save plot.
    """
    if not top_k:
        print("No models to plot.")
        return
    
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    k = len(top_k)
    
    # Determine grid layout
    if k <= 2:
        nrows, ncols = 1, k
        figsize = (10*k, 7)
    else:
        nrows, ncols = 2, 2
        figsize = (20, 14)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes).flatten()  # Handle single subplot case
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    
    for idx, (ax, result) in enumerate(zip(axes, top_k)):
        per_class_f1 = result['per_class_f1']
        
        bars = ax.bar(class_names, per_class_f1, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, f1 in zip(bars, per_class_f1):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{f1:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Macro F1 line
        ax.axhline(y=result['macro_f1'], color='red', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Macro F1: {result["macro_f1"]:.3f}')
        
        # Title with rank and shortened name
        exp_name_short = shorten_experiment_name(result['experiment'], max_len=60)
        
        title = f"#{idx+1}: {exp_name_short}\n"
        title += f"Macro F1: {result['macro_f1']:.4f} | Acc: {result['accuracy']:.4f}"
        ax.set_title(title, fontsize=11, fontweight='bold', wrap=True)
        
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='upper right')
        
        # Only show y-label on leftmost plots
        if idx % ncols == 0:
            ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
        
        # Only show x-label on bottom plots
        if idx >= k - ncols:
            ax.set_xlabel('Sleep Stage', fontsize=13, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(k, len(axes)):
        axes[idx].axis('off')
    
    dir_name = checkpoint_dir.name if checkpoint_dir else CHECKPOINT_DIR.name
    title_str = f'Top {k} Models by Macro F1\nCheckpoint Directory: {dir_name}'
    if name_filter:
        title_str += f'\nFilter: "{name_filter}"'
    plt.suptitle(title_str, fontsize=17, fontweight='bold', y=0.985)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved top-{k} comparison plot to: {save_path}")
    
    plt.show()


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
    train_cfg = checkpoint.get('training_config', {})
    
    method = train_cfg.get('method', 'unknown')
    model_type = train_cfg.get('model_type', 'Unknown')
    print(f"Model type:          {method or model_type}")
    
    # Handle different model types
    if 'multichannel_sleepnet' in method.lower():
        # MultiChannelSleepNet hyperparameters
        print(f"Sampling frequency:  {hyperparams.get('fs', 128)} Hz")
        print(f"Epoch length:        {hyperparams.get('epoch_len_sec', 30)} sec")
        print(f"FFT size:            {hyperparams.get('n_fft', 256)}")
        print(f"Frequency bins:      {hyperparams.get('freq_bins', 128)}")
        print(f"Time frames:         {hyperparams.get('pad_size', 'N/A')}")
        print(f"\n  SINGLE-CHANNEL TRANSFORMER")
        print(f"  Attention heads:     {hyperparams.get('num_head', 'N/A')}")
        print(f"  Transformer layers:  {hyperparams.get('num_encoder', 'N/A')}")
        print(f"  Feedforward dim:     {hyperparams.get('forward_hidden', 'N/A')}")
        print(f"\n  MULTI-CHANNEL FUSION TRANSFORMER")
        print(f"  Attention heads:     {hyperparams.get('num_head', 'N/A')}")
        print(f"  Transformer layers:  {hyperparams.get('num_encoder_multi', 'N/A')}")
        print(f"  Feedforward dim:     {hyperparams.get('forward_hidden', 'N/A')}")
        print(f"\n  CLASSIFIER")
        print(f"  FC hidden dim:       {hyperparams.get('fc_hidden', 'N/A')}")
        print(f"  Dropout (fusion):    {hyperparams.get('dropout_tf', 'N/A')}")
        print(f"  Dropout (internal):  {hyperparams.get('dropout_tr', 'N/A')}")
    elif 'Sequence' in model_type:
        # Sequence model hyperparameters
        print(f"Sequence length:     {train_cfg.get('seq_len', 'N/A')} epochs")
        print(f"Stride:              {train_cfg.get('stride', 'N/A')} epochs")
        print(f"GRU hidden size:     {hyperparams.get('gru_hidden', 'N/A')}")
        print(f"GRU layers:          {hyperparams.get('gru_layers', 'N/A')}")
        if 'gru_bidirectional' in hyperparams:
            print(f"GRU bidirectional:   {hyperparams.get('gru_bidirectional', False)}")
        
        # Transformer hyperparams for sequence if applicable
        if 'Transformer' in model_type:
            print(f"Transformer d_model: {hyperparams.get('d_model_seq', 'N/A')}")
            print(f"Transformer heads:   {hyperparams.get('nhead_seq', 'N/A')}")
            print(f"Transformer layers:  {hyperparams.get('num_layers_seq', 'N/A')}")
        
        # Epoch encoder hyperparameters
        print(f"\n  EPOCH ENCODER (EpochTransformerConv1D_v2)")
        print(f"  Embedding dim:       {hyperparams.get('d_model', 'N/A')}")
        print(f"  Attention heads:     {hyperparams.get('nhead', 'N/A')}")
        print(f"  Transformer layers:  {hyperparams.get('num_layers', 'N/A')}")
        print(f"  Feedforward dim:     {hyperparams.get('dim_feedforward', 'N/A')}")
        print(f"  Target tokens:       {hyperparams.get('target_tokens', 'N/A')}")
    else:
        # Epoch model hyperparameters (Conv1D or mean-pool)
        print(f"Input channels:      {hyperparams.get('input_channels', 'N/A')}")
        print(f"Sequence length:     {hyperparams.get('seq_length', 'N/A')}")
        print(f"Max tokens:          {train_cfg.get('max_tokens', hyperparams.get('max_tokens', 'N/A'))}")
        print(f"Patch size:          {train_cfg.get('patch_size', 'N/A')}")
        print(f"Final seq length:    {train_cfg.get('final_seq_length', 'N/A')} (tokens fed to transformer)")
        print(f"Embedding dim:       {hyperparams.get('d_model', 'N/A')}")
        print(f"Attention heads:     {hyperparams.get('nhead', 'N/A')}")
        print(f"Transformer layers:  {hyperparams.get('num_layers', 'N/A')}")
        print(f"Feedforward dim:     {hyperparams.get('dim_feedforward', 'N/A')}")
        print(f"Dropout:             {hyperparams.get('dropout', 'N/A')}")
    
    print(f"Number of classes:   {hyperparams.get('num_classes', 'N/A')}")
    
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
        print(f"\n CLASS WEIGHTS")
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
    
    # 3. Model architecture (text box)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    hp = checkpoint['hyperparameters']
    train_cfg = checkpoint.get('training_config', {})
    
    method = train_cfg.get('method', 'unknown')
    model_type = train_cfg.get('model_type', 'Unknown')
    
    # Generate architecture text based on model type
    if 'multichannel_sleepnet' in method.lower():
        arch_text = f"""
    MODEL ARCHITECTURE
    ───────────────────────
    Model Type:          MultiChannelSleepNet
    Sampling Frequency:  {hp.get('fs', 128)} Hz
    FFT Size:            {hp.get('n_fft', 256)}
    
    Single-Ch Layers:    {hp.get('num_encoder', 'N/A')}
    Fusion Layers:       {hp.get('num_encoder_multi', 'N/A')}
    Attn Heads:          {hp.get('num_head', 'N/A')}
    
    FC Hidden Dim:       {hp.get('fc_hidden', 'N/A')}
    Dropout:             {hp.get('dropout_tf', 'N/A')}
    """
    elif 'Sequence' in model_type:
        arch_text = f"""
    MODEL ARCHITECTURE
    ───────────────────────
    Model Type:          {model_type}
    Sequence Length:     {train_cfg.get('seq_len', 'N/A')} epochs
    Stride:              {train_cfg.get('stride', 'N/A')} epochs
    
    Epoch Encoder d_model: {hp.get('d_model', 'N/A')}
    Epoch Transformer Layers: {hp.get('num_layers', 'N/A')}
    Target Tokens:       {hp.get('target_tokens', 'N/A')}
    
    GRU Hidden:          {hp.get('gru_hidden', 'N/A')}
    GRU Layers:          {hp.get('gru_layers', 'N/A')}
    """
    else:
        max_tokens = train_cfg.get('max_tokens', hp.get('max_tokens', 'N/A'))
        final_seq_len = train_cfg.get('final_seq_length', 'N/A')
        patch_size = train_cfg.get('patch_size', 'N/A')
        
        arch_text = f"""
    MODEL ARCHITECTURE
    ───────────────────────
    Input Seq Length:    {hp.get('seq_length', 'N/A')}
    Patch Size:          {patch_size}
    Final Tokens:        {final_seq_len}
    Max Tokens:          {max_tokens}
    
    Embedding Dim:       {hp.get('d_model', 'N/A')}
    Attention Heads:     {hp.get('nhead', 'N/A')}
    Transformer Layers:  {hp.get('num_layers', 'N/A')}
    Feedforward Dim:     {hp.get('dim_feedforward', 'N/A')}
    Dropout:             {hp.get('dropout', 'N/A')}
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
            
            # Updated title with both Macro F1 and Accuracy
            ax.set_title(f'{exp_name}\nMacro F1: {checkpoint["val_macro_f1"]:.3f} | Acc: {checkpoint["val_accuracy"]:.3f}', 
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
    parser.add_argument('--top-k', type=int, default=None,
                       help='Show top-K models by macro F1 (e.g., --top-k 4)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--filter', type=str, default=None,
                       help='Only include experiments containing this string (e.g., "conv1d_v2")')
    
    args = parser.parse_args()
    
    # Resolve checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    else:
        checkpoint_dir = CHECKPOINT_DIR
    
    if args.top_k:
        # Show top-K models
        top_k = print_top_k_models(k=args.top_k, checkpoint_dir=checkpoint_dir, name_filter=args.filter)
        
        if top_k and args.save_plots:
            plots_dir = PROJECT_ROOT / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            filter_suffix = f"_{args.filter}" if args.filter else ""
            plot_path = plots_dir / f"top_{args.top_k}_models_{checkpoint_dir.name}{filter_suffix}.png"
            plot_top_k_comparison(top_k, checkpoint_dir=checkpoint_dir, name_filter=args.filter, save_path=plot_path)
        elif top_k:
            plot_top_k_comparison(top_k, checkpoint_dir=checkpoint_dir, name_filter=args.filter)
    
    elif args.compare:
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
        print(f"\nScanning: {checkpoint_dir}")
        print("\nAvailable checkpoints:")
        for exp_dir in sorted(checkpoint_dir.iterdir()):
            if exp_dir.is_dir():
                print(f"  - {exp_dir.name}")


if __name__ == "__main__":
    # Quick test mode (no command line args)
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("  python scripts/analyze_checkpoint.py epochtransformer_notch_bandpass_resample_znorm_v1")
        print("  python scripts/analyze_checkpoint.py epochlevel_multichannel_sleepnet_notch_bandpass_resample_v1")
        print("  python scripts/analyze_checkpoint.py epochlevel_multichannel_sleepnet_notch_bandpass_resample_v1 --save-plots")
        print("  python scripts/analyze_checkpoint.py --compare exp1 exp2")
        print("  python scripts/analyze_checkpoint.py --top-k 4")
        print("  python scripts/analyze_checkpoint.py --top-k 4 --save-plots")
        print("  python scripts/analyze_checkpoint.py --top-k 4 --checkpoint-dir checkpoints")
        print("  python scripts/analyze_checkpoint.py --top-k 4 --filter multichannel_sleepnet")
        print("  python scripts/analyze_checkpoint.py --top-k 4 --filter multichannel_sleepnet --save-plots")
        print("\nAvailable checkpoint directories:")
        print("  - checkpoints")
        print("  - checkpoints_leomed")
    else:
        main()
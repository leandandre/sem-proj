"""
Test time-shift and amplitude-scale data augmentation.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import matplotlib modules in EXACT same order as analyze_checkpoint.py
import torch
import numpy as np
import matplotlib.pyplot as plt  # No backend configuration needed - use system default

from sem_proj.data.transforms import RandomTimeShift, RandomAmplitudeScale, Compose
from sem_proj.data.datasets import BoasDataset
from sem_proj.data.preprocessing import PreprocessingConfig

config = PreprocessingConfig.from_yaml("configs/preprocess/notch_bandpass_resample_znorm.yaml")

# Without augmentation
print("Loading original dataset...")
ds_orig = BoasDataset(subjects=["sub-1"], mode="headband", preprocess_config=config, use_cache=True)
x_orig, y = ds_orig[0]
print(f"✓ Loaded epoch with label: {y}")

# Test individual augmentations
print("\nGenerating augmented versions...")

# Time-shift only
transform_shift = RandomTimeShift(max_shift_ratio=0.1)
x_shift1 = transform_shift(x_orig.clone())
x_shift2 = transform_shift(x_orig.clone())

# Amplitude scale only
transform_amp = RandomAmplitudeScale(scale_range=(0.8, 1.2))
x_amp1 = transform_amp(x_orig.clone())
x_amp2 = transform_amp(x_orig.clone())

# Both combined
transform_both = Compose([
    RandomTimeShift(max_shift_ratio=0.1),
    RandomAmplitudeScale(scale_range=(0.8, 1.2))
])
x_both1 = transform_both(x_orig.clone())
x_both2 = transform_both(x_orig.clone())

print(f"✓ Generated augmented versions")
print(f"  - Original amplitude range: [{x_orig[0].min():.3f}, {x_orig[0].max():.3f}]")
print(f"  - Amplitude-scaled #1 range: [{x_amp1[0].min():.3f}, {x_amp1[0].max():.3f}]")
print(f"  - Amplitude-scaled #2 range: [{x_amp2[0].min():.3f}, {x_amp2[0].max():.3f}]")

# Create figure with multiple subplots
print("\nCreating plot...")
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)

time = np.arange(x_orig.shape[1]) / 128  # Time in seconds (assuming 128 Hz)

# Column 1: Time-Shift Only
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time, x_orig[0].numpy(), 'b-', linewidth=0.8, alpha=0.7, label='Original')
ax1.plot(time, x_shift1[0].numpy(), 'r-', linewidth=0.8, alpha=0.7, label='Shifted')
ax1.set_title("Time-Shift Only #1", fontweight='bold')
ax1.set_ylabel("Amplitude (Ch 0)")
ax1.grid(alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time, x_orig[0].numpy(), 'b-', linewidth=0.8, alpha=0.7, label='Original')
ax2.plot(time, x_shift2[0].numpy(), 'g-', linewidth=0.8, alpha=0.7, label='Shifted')
ax2.set_title("Time-Shift Only #2", fontweight='bold')
ax2.set_ylabel("Amplitude (Ch 0)")
ax2.grid(alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

# Column 2: Amplitude Scale Only
ax3 = fig.add_subplot(gs[0, 1])
ax3.plot(time, x_orig[0].numpy(), 'b-', linewidth=0.8, alpha=0.7, label='Original')
ax3.plot(time, x_amp1[0].numpy(), 'r-', linewidth=0.8, alpha=0.7, label='Scaled')
ax3.set_title("Amplitude Scale Only #1", fontweight='bold')
ax3.set_ylabel("Amplitude (Ch 0)")
ax3.grid(alpha=0.3)
ax3.legend(loc='upper right', fontsize=9)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time, x_orig[0].numpy(), 'b-', linewidth=0.8, alpha=0.7, label='Original')
ax4.plot(time, x_amp2[0].numpy(), 'g-', linewidth=0.8, alpha=0.7, label='Scaled')
ax4.set_title("Amplitude Scale Only #2", fontweight='bold')
ax4.set_ylabel("Amplitude (Ch 0)")
ax4.grid(alpha=0.3)
ax4.legend(loc='upper right', fontsize=9)

# Column 3: Both Combined
ax5 = fig.add_subplot(gs[0, 2])
ax5.plot(time, x_orig[0].numpy(), 'b-', linewidth=0.8, alpha=0.7, label='Original')
ax5.plot(time, x_both1[0].numpy(), 'r-', linewidth=0.8, alpha=0.7, label='Both')
ax5.set_title("Time-Shift + Amplitude #1", fontweight='bold')
ax5.set_ylabel("Amplitude (Ch 0)")
ax5.grid(alpha=0.3)
ax5.legend(loc='upper right', fontsize=9)

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(time, x_orig[0].numpy(), 'b-', linewidth=0.8, alpha=0.7, label='Original')
ax6.plot(time, x_both2[0].numpy(), 'g-', linewidth=0.8, alpha=0.7, label='Both')
ax6.set_title("Time-Shift + Amplitude #2", fontweight='bold')
ax6.set_ylabel("Amplitude (Ch 0)")
ax6.grid(alpha=0.3)
ax6.legend(loc='upper right', fontsize=9)

# Bottom row: Zoomed-in comparison (5-second window)
zoom_start = 10  # seconds
zoom_end = 15
zoom_idx_start = int(zoom_start * 128)
zoom_idx_end = int(zoom_end * 128)
time_zoom = time[zoom_idx_start:zoom_idx_end]

ax7 = fig.add_subplot(gs[2, :])
ax7.plot(time_zoom, x_orig[0, zoom_idx_start:zoom_idx_end].numpy(), 'b-', linewidth=1.2, label='Original')
ax7.plot(time_zoom, x_amp1[0, zoom_idx_start:zoom_idx_end].numpy(), 'r--', linewidth=1.2, label='Amplitude Scaled')
ax7.set_title(f"Zoomed Comparison: Amplitude Scaling (seconds {zoom_start}-{zoom_end})", fontweight='bold', fontsize=12)
ax7.set_ylabel("Amplitude (Ch 0)")
ax7.set_xlabel("Time (seconds)")
ax7.grid(alpha=0.3)
ax7.legend(loc='upper right')

# Bottom: Statistics comparison
ax8 = fig.add_subplot(gs[3, :])
ax8.axis('off')

stats_text = f"""
AUGMENTATION STATISTICS (Channel 0)
{'─'*80}

Original Signal:
  Mean: {x_orig[0].mean():.6f}    Std: {x_orig[0].std():.6f}    Min: {x_orig[0].min():.3f}    Max: {x_orig[0].max():.3f}

Time-Shifted #1:
  Mean: {x_shift1[0].mean():.6f}    Std: {x_shift1[0].std():.6f}    Min: {x_shift1[0].min():.3f}    Max: {x_shift1[0].max():.3f}
  → Mean/Std unchanged (only position shifts)

Amplitude-Scaled #1:
  Mean: {x_amp1[0].mean():.6f}    Std: {x_amp1[0].std():.6f}    Min: {x_amp1[0].min():.3f}    Max: {x_amp1[0].max():.3f}
  → Mean/Std/Range scaled by random factor

Amplitude-Scaled #2:
  Mean: {x_amp2[0].mean():.6f}    Std: {x_amp2[0].std():.6f}    Min: {x_amp2[0].min():.3f}    Max: {x_amp2[0].max():.3f}
  → Different random scaling factor

Combined (Shift + Amplitude) #1:
  Mean: {x_both1[0].mean():.6f}    Std: {x_both1[0].std():.6f}    Min: {x_both1[0].min():.3f}    Max: {x_both1[0].max():.3f}
  → Both effects applied
"""

ax8.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.3))

plt.suptitle(f'Data Augmentation Test - Epoch Label: {y}\nTime-Shift (±10%) and Amplitude Scale (0.8-1.2×)', 
             fontsize=16, fontweight='bold')

# Save first (same as analyze_checkpoint.py)
save_path = PROJECT_ROOT / "plots" / "augmentation_test_full.png"
save_path.parent.mkdir(exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved augmentation test plot to: {save_path}")

print(f"\nSignal info:")
print(f"  Shape: {x_orig.shape}")
print(f"  Duration: {x_orig.shape[1] / 128:.1f} seconds")
print(f"  Time-shift range: ±{int(x_orig.shape[1] * 0.1)} samples = ±{x_orig.shape[1] * 0.1 / 128:.2f} seconds")
print(f"  Amplitude scale range: 0.8× to 1.2× (±20%)")
print(f"  Epoch label: {y}")

# Show interactively (same as analyze_checkpoint.py)
plt.show()

print("\n✓ Done! Check the saved plot at:", save_path)
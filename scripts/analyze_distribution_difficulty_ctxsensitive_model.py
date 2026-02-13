"""
This script scores the validation and the test nights individually to see what nights are hard to classify and if vaidation is more difficutl than test set.
"""
import numpy as np
import json
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sem_proj.data.datasets import BoasSequenceDataset
from sem_proj.data.preprocessing import PreprocessingConfig
from sem_proj.models.model_factory import SSLEpochTransformerConv1D_v2, SequenceGRUClassifier
from sem_proj.data.boa_loader import build_pid_mappings

CONFIG_DIR = PROJECT_ROOT / "configs" / "preprocess"
CHECKPOINT_LEOMED_DIR = PROJECT_ROOT / "checkpoints_leomed"
SPLITS_FILE = PROJECT_ROOT / "data" / "processed" / "data_splits_70_15_15.json"
TARGET_DIR_METRICS = PROJECT_ROOT / "reports" / "metrics"
TARGET_DIR_METRICS.mkdir(parents=True, exist_ok=True)
TARGET_DIR_PLOTS = PROJECT_ROOT / "plots"
TARGET_DIR_PLOTS.mkdir(parents=True, exist_ok=True)

def evaluate_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)  # (B, L, C, T)
            y = y.to(device)  # (B, L)

            output = model(x)  # (B, L, num_classes)

            preds = output.argmax(dim=-1)  # (B, L)
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()

            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(y.cpu().numpy().flatten())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return accuracy, macro_f1, per_class_f1

def main():
    ### these are the hyperparameters and the architecture for my best sequence model (same architecture for Supervised from Scratch)###
    batch_size = 64
    seq_len = 20  # length of input sequences
    stride = 5    # stride for creating sequences
    lr_encoder = 1e-5 # for supervised from scratch it is 1e-4
    lr_gru = 1e-4 # same for both
    mode = "headband"
    experiment_name = "ctxsensitive_stage1_p1.0_ssl_finetuning" # change accordingly
    preprocess_config = PreprocessingConfig.from_yaml(CONFIG_DIR / "notch_bandpass_resample_znorm.yaml")
    use_cache = True
    ssl_checkpoint = CHECKPOINT_LEOMED_DIR / "ssl_cross_modal_notch_bandpass_resample_znorm_v1" / "best_model.pt"
    freeze_encoder=False
    d_model = 128 # embedding dimension in all models (SSL and supervised)
    nhead = 4 # 4 attention heads in the transformer
    num_layers_encoder = 2 # 2 transformer layers
    dim_feedforward = 512 # feedforward dimension in the transformer, always kept it at 4*d_model
    dropout_encoder = 0.2
    dropout_gru = 0.2 # never in use since gru only has one layer
    target_tokens = 240 # number of input tokens to the transformer
    class_weighted_loss = True # always used it
    gradient_clip = 5.0 # for my final models always used 5.0
    early_stopping_patience = 12
    num_classes = 5 # 5 sleep stages to classify
    hidden_size = 128 # gru hidden size
    num_layers = 1 # gru number of layers
    bidirectional = True # gru bidirectionality
    ### end of architecture and hyperparameters ### 

    # NOTE: the leq_len and stride were not saved in the checkpoints, so add them manually (L=20, stride=5)
    
    ### loading and evluating model finetuned with p=1.0 during STAGE 1 ###
    finetuned_checkpoint = CHECKPOINT_LEOMED_DIR / experiment_name / "best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)
    val_nights = splits['val_subjects']
    test_nights = splits['test_subjects']

    val_nights_metrics = {}
    test_nights_metrics = {}


    checkpoint_dict = torch.load(finetuned_checkpoint, map_location='cpu')
    encoder = SSLEpochTransformerConv1D_v2(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers_encoder,
        dim_feedforward=dim_feedforward,
        dropout=dropout_encoder,
        target_tokens=target_tokens,
    )
    context_model = SequenceGRUClassifier(
        epoch_model = encoder,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        bidirectional=bidirectional,
    )
    context_model.load_state_dict(checkpoint_dict['model_state_dict'])
    context_model = context_model.to(device)

    val_metrics_file = TARGET_DIR_METRICS / "ctxsensitive_val_set_per_night_metrics.json"
    test_metrics_file = TARGET_DIR_METRICS / "ctxsensitive_test_set_per_night_metrics.json"

    # Check if metrics already exist, if so load them
    if val_metrics_file.exists() and test_metrics_file.exists():
        with open(val_metrics_file, 'r') as f:
            val_nights_metrics = json.load(f)
        with open(test_metrics_file, 'r') as f:
            test_nights_metrics = json.load(f)
    else:
        for night_id in val_nights:
            night_ds = BoasSequenceDataset(
                subjects=[night_id],
                mode=mode,
                seq_len=seq_len,
                stride=stride,      # to be 100% correct, stride should have been = seq_length, but does not matter much for qualitative analysis (13.02)
                transform_hb=None, # no data augmentation during evaluation!
                preprocess_config=preprocess_config,
                use_cache=use_cache,
            )
            night_dl = DataLoader(
                night_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
            )
            night_acc, night_mf1, night_perclass_f1 = evaluate_model(context_model, night_dl, device)
            val_nights_metrics[night_id] = {
                "night_acc": night_acc,
                "night_mf1": night_mf1,
                "night_perclass_f1": night_perclass_f1.tolist(),
            }
        
        for night_id in test_nights:
            night_ds = BoasSequenceDataset(
                subjects=[night_id],
                mode=mode,
                seq_len=seq_len,
                stride=stride,      # to be 100% correct, stride should have been = seq_length, but does not matter much for qualitative analysis (13.02)
                transform_hb=None, # no data augmentation during evaluation!
                preprocess_config=preprocess_config,
                use_cache=use_cache,
            )
            night_dl = DataLoader(
                night_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
            )
            night_acc, night_mf1, night_perclass_f1 = evaluate_model(context_model, night_dl, device)
            test_nights_metrics[night_id] = {
                "night_acc": night_acc,
                "night_mf1": night_mf1,
                "night_perclass_f1": night_perclass_f1.tolist(),
            }

        with open(val_metrics_file, 'w') as f:
            json.dump(val_nights_metrics, f, indent=4)
        with open(test_metrics_file, 'w') as f:
            json.dump(test_nights_metrics, f, indent=4)
        print("Metrics computed and saved.")

    
    val_nights_mf1s = []
    test_nights_mf1s = []
    for night_id in val_nights:
        nested = val_nights_metrics[night_id]
        val_nights_mf1s.append(nested['night_mf1'])
    for night_id in test_nights:
        nested = test_nights_metrics[night_id]
        test_nights_mf1s.append(nested['night_mf1'])

    # ## what would the statistics be if we exclude the 3 bad nights in the val set? ##
    # val_nights_mf1s = [mf1 for mf1 in val_nights_mf1s if mf1 >= 0.5]

    print("Validation Nights Macro F1 Score Distribution:")
    print(f"MF1 Distr over VAL nights: {np.mean(val_nights_mf1s):.4f} ± {np.std(val_nights_mf1s):.4f}, median: {np.median(val_nights_mf1s):.4f}")
    print("\nTest Nights Macro F1 Score Distribution:")
    print(f"MF1 Distr over TEST nights: {np.mean(test_nights_mf1s):.4f} ± {np.std(test_nights_mf1s):.4f}, median: {np.median(test_nights_mf1s):.4f}")

    color_val = 'orange'
    color_test = 'blue'
    plt.figure(figsize=(10,6))
    plt.hist(val_nights_mf1s, bins=14, alpha=0.7, label='Validation Nights', color=color_val)
    plt.xlabel('Macro F1 Score', fontsize=18)
    plt.ylabel('Number of Nights', fontsize=18)
    plt.title('Distribution of Macro F1 Scores Across Validation Nights', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=16)
    # plt.show()
    plt.savefig(TARGET_DIR_PLOTS / "ctxsensitive_val_set_per_night_mf1_distribution.pdf", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(10,6))
    plt.hist(test_nights_mf1s, bins=14, alpha=0.7, label='Test Nights', color=color_test)
    plt.xlabel('Macro F1 Score', fontsize=18)
    plt.ylabel('Number of Nights', fontsize=18)
    plt.title('Distribution of Macro F1 Scores Across Test Nights', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=16)
    # plt.show()
    plt.savefig(TARGET_DIR_PLOTS / "ctxsensitive_test_set_per_night_mf1_distribution.pdf", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10,6))
    plt.hist(val_nights_mf1s, bins=14, alpha=0.7, label='Validation Nights', color=color_val)
    plt.hist(test_nights_mf1s, bins=14, alpha=0.7, label='Test Nights', color=color_test)
    plt.xlabel('Macro F1 Score', fontsize=18)
    plt.ylabel('Number of Nights', fontsize=18)
    plt.title('Distribution of Macro F1 Scores Across Validation and Test Nights', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=16)
    # plt.show()
    plt.savefig(TARGET_DIR_PLOTS / "ctxsensitive_val_and_test_set_per_night_mf1_distributions.pdf", dpi=300, bbox_inches='tight')
    


    # Violin plots for clearer distribution comparison, 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Side-by-side violin plots
    data_for_violin = {
        'Validation': val_nights_mf1s,
        'Test': test_nights_mf1s
    }
    positions = [0, 1]
    parts = axes[0].violinplot([val_nights_mf1s, test_nights_mf1s], positions=positions, showmeans=True, showmedians=True)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(['Validation', 'Test'], fontsize=16)
    axes[0].set_ylabel('Macro F1 Score', fontsize=18)
    axes[0].set_title('Violin Plot: Macro F1 Score Distribution', fontsize=18)
    axes[0].tick_params(axis='y', labelsize=16)
    axes[0].grid(axis='y', alpha=0.3)
    
    # KDE plot on the second subplot
    axes[1].hist(val_nights_mf1s, bins=14, alpha=0.5, label='Validation Nights', color=color_val, density=True)
    axes[1].hist(test_nights_mf1s, bins=14, alpha=0.5, label='Test Nights', color=color_test, density=True)
    
    # Add KDE curves
    from scipy import stats
    kde_val = stats.gaussian_kde(val_nights_mf1s)
    kde_test = stats.gaussian_kde(test_nights_mf1s)
    x_range = np.linspace(min(min(val_nights_mf1s), min(test_nights_mf1s)), 
                           max(max(val_nights_mf1s), max(test_nights_mf1s)), 200)
    axes[1].plot(x_range, kde_val(x_range), color=color_val, linewidth=2.5, label='Validation KDE')
    axes[1].plot(x_range, kde_test(x_range), color=color_test, linewidth=2.5, label='Test KDE')
    axes[1].set_xlabel('Macro F1 Score', fontsize=18)
    axes[1].set_ylabel('Density', fontsize=18)
    axes[1].set_title('Histogram with KDE: Macro F1 Score Distribution', fontsize=18)
    axes[1].legend(fontsize=12)
    axes[1].tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(TARGET_DIR_PLOTS / "ctxsensitive_val_and_test_set_per_night_mf1_violin_and_kde.pdf", dpi=300, bbox_inches='tight')
    


    # I see in the histogram that 3 nights in val set have mf1 < 0.5
    nights_low_mf1 = []
    for night_id in val_nights:
        nested = val_nights_metrics[night_id]
        if nested['night_mf1'] < 0.5:
            nights_low_mf1.append((night_id, nested['night_mf1']))
    print("\nValidation Nights with Macro F1 Score below 0.5:")
    for night_id, mf1 in nights_low_mf1:
        print(f"Night ID: {night_id}, Macro F1 Score: {mf1:.4f}")
    
    # which participents are these nights?
    val_pids = splits['val_pids']
    nights_low_mf1_ids = [night_id for night_id, _ in nights_low_mf1]
    sub_pid_tuples = []
    pid_to_subs, sub_to_pid = build_pid_mappings() # get the whole mapping
    for night_id in nights_low_mf1_ids:
        pid = sub_to_pid[night_id]
        sub_pid_tuples.append((night_id, pid))
    print("\nCorresponding Participant IDs for Low MF1 Nights:")
    for night_id, pid in sub_pid_tuples:
        print(f"Night ID: {night_id}, Participant ID: {pid}")
    



if __name__ == "__main__":
    main()
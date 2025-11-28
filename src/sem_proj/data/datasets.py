from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .boa_loader import (
    list_boas_subjects,
    load_headband_raw,
    load_psg_raw,
    load_labels,
    compute_epoch_sample_bounds,
)

from .preprocessing import (
    PreprocessingConfig,
    preprocess_raw_basic,
    compute_subject_normalization_stats,
    apply_znorm_to_epoch,
    get_expected_seq_length,
)

from .cache import load_from_cache, save_to_cache

N_HB_CHANNELS = 2
N_PSG_CHANNELS = 6


class BoasDataset(Dataset):
    """
    BOAS dataset with flexible preprocessing including per-subject z-normalization.
    
    Supports caching for faster loading on subsequent runs.
    """

    def __init__(
        self,
        subjects: Optional[Iterable[str]] = None,
        mode: str = "headband",
        transform_hb=None,
        transform_psg=None,
        target_transform=None,
        preprocess_config: Optional[PreprocessingConfig] = None,
        use_cache: bool = True,
    ):
        super().__init__()
        if subjects is None:
            subjects = list_boas_subjects()
        self.subjects: List[str] = list(subjects)

        assert mode in {"headband", "psg", "cross"}
        self.mode = mode
        
        self.preprocess_config = preprocess_config or PreprocessingConfig.no_preprocessing()
        self.use_cache = use_cache  # Store cache preference

        self.transform_hb = transform_hb
        self.transform_psg = transform_psg
        self.target_transform = target_transform

        # Per-subject caches (now storing np.ndarray instead of mne.Raw)
        self.raw_data_hb = {}      # np.ndarray: preprocessed data
        self.raw_data_psg = {}
        self.sfreq_hb = {}         # sampling frequencies
        self.sfreq_psg = {}
        
        self.labels_psg = {}
        self.bounds_hb = {}
        self.bounds_psg = {}
        self.valid_psg = {}
        self.valid_hb_ai = {}
        
        # Per-subject normalization statistics
        self.norm_stats_hb = {}
        self.norm_stats_psg = {}

        # Global list of (subject, epoch_idx)
        self.indices: List[Tuple[str, int]] = []

        self._build_index()

    def _build_index(self):
        """Build index and compute per-subject normalization statistics."""
        for subj in self.subjects:
            print(f"Loading {subj}...")
            
            # TRY TO LOAD FROM CACHE FIRST
            if self.use_cache:
                cache_data_hb = None
                cache_data_psg = None
                
                if self.mode in {"headband", "cross"}:
                    cache_data_hb = load_from_cache(subj, self.preprocess_config, mode="headband")
                
                # ALWAYS try to load PSG cache (needed for labels/valid_mask)
                cache_data_psg = load_from_cache(subj, self.preprocess_config, mode="psg")
                
                # Determine cache hit status
                cache_hit_hb = cache_data_hb is not None if self.mode in {"headband", "cross"} else True
                cache_hit_psg = cache_data_psg is not None  # Always need PSG for labels
                
                if cache_hit_hb and cache_hit_psg:
                    # Load headband data if needed
                    if self.mode in {"headband", "cross"}:
                        self.raw_data_hb[subj] = cache_data_hb['raw_data']
                        self.sfreq_hb[subj] = cache_data_hb['sfreq']
                        self.bounds_hb[subj] = cache_data_hb['bounds']
                        self.norm_stats_hb[subj] = (cache_data_hb['norm_mean'], cache_data_hb['norm_std'])
                        self.valid_hb_ai[subj] = cache_data_hb.get('valid_mask', np.ones(len(cache_data_hb['labels']), dtype=bool))
                    
                    # ALWAYS load PSG labels and valid mask
                    self.labels_psg[subj] = cache_data_psg['labels']
                    self.valid_psg[subj] = cache_data_psg['valid_mask']
                    
                    # Load PSG data if in PSG or cross mode
                    if self.mode in {"psg", "cross"}:
                        self.raw_data_psg[subj] = cache_data_psg['raw_data']
                        self.sfreq_psg[subj] = cache_data_psg['sfreq']
                        self.bounds_psg[subj] = cache_data_psg['bounds']
                        self.norm_stats_psg[subj] = (cache_data_psg['norm_mean'], cache_data_psg['norm_std'])
                    
                    # Add to indices
                    if self.mode == "headband":
                        mask = self.valid_psg[subj] & self.valid_hb_ai[subj]
                    elif self.mode == "psg":
                        mask = self.valid_psg[subj]
                    else:  # cross
                        mask = self.valid_psg[subj] & self.valid_hb_ai[subj]
                    
                    epoch_indices = np.where(mask)[0]
                    for ei in epoch_indices:
                        self.indices.append((subj, int(ei)))
                    
                    continue  # Skip to next subject
            
            # CACHE MISS - Process from scratch
            # --- Load and preprocess raw data ---
            if self.mode in {"headband", "cross"}:
                raw_hb = load_headband_raw(subj, preprocess_config=None)
                raw_hb = preprocess_raw_basic(raw_hb, self.preprocess_config)
                
            if self.mode in {"psg", "cross"}:
                raw_psg = load_psg_raw(subj, preprocess_config=None)
                raw_psg = preprocess_raw_basic(raw_psg, self.preprocess_config)

            # --- Load labels ---
            labels_psg, onset_psg, duration_psg, is_valid_psg, _ = load_labels(
                subj, source="psg", label_type="hum",
            )
            self.labels_psg[subj] = labels_psg
            self.valid_psg[subj] = is_valid_psg

            _, onset_hb_ai, duration_hb_ai, is_valid_hb_ai, _ = load_labels(
                subj, source="headband", label_type="ai",
            )
            self.valid_hb_ai[subj] = is_valid_hb_ai

            # --- Compute epoch bounds ---
            if self.mode in {"headband", "cross"}:
                self.bounds_hb[subj] = compute_epoch_sample_bounds(
                    raw_hb, onset_psg, duration_psg
                )
            if self.mode in {"psg", "cross"}:
                self.bounds_psg[subj] = compute_epoch_sample_bounds(
                    raw_psg, onset_psg, duration_psg
                )

            # --- Channel selection and normalization ---
            if self.mode in {"headband", "cross"}:
                raw_hb_selected = raw_hb.copy()
                raw_hb_selected.pick(raw_hb_selected.ch_names[:N_HB_CHANNELS])
                
                valid_for_norm_hb = self.valid_psg[subj] & self.valid_hb_ai[subj]
                # Only compute night stats if flag enabled
                mean_hb, std_hb = compute_subject_normalization_stats(
                    raw_hb_selected,
                    valid_for_norm_hb,
                    self.bounds_hb[subj],
                    self.preprocess_config,
                )
                self.norm_stats_hb[subj] = (mean_hb, std_hb)  # may be (None, None)
                
                self.raw_data_hb[subj] = raw_hb_selected.get_data()
                self.sfreq_hb[subj] = raw_hb_selected.info['sfreq']
                
                if self.use_cache:
                    save_to_cache(
                        subject=subj,
                        config=self.preprocess_config,
                        mode="headband",
                        raw_data=self.raw_data_hb[subj],
                        sfreq=self.sfreq_hb[subj],
                        ch_names=raw_hb_selected.ch_names,
                        bounds=self.bounds_hb[subj],
                        labels=labels_psg,
                        valid_mask=valid_for_norm_hb,
                        norm_mean=mean_hb,   # None if epoch-wise
                        norm_std=std_hb,     # None if epoch-wise
                    )

            if self.mode in {"psg", "cross"}:
                raw_psg_selected = raw_psg.copy()
                raw_psg_selected.pick(raw_psg_selected.ch_names[:N_PSG_CHANNELS])
                
                valid_for_norm_psg = self.valid_psg[subj]
                mean_psg, std_psg = compute_subject_normalization_stats(
                    raw_psg_selected,
                    valid_for_norm_psg,
                    self.bounds_psg[subj],
                    self.preprocess_config,
                )
                self.norm_stats_psg[subj] = (mean_psg, std_psg)
                
                self.raw_data_psg[subj] = raw_psg_selected.get_data()
                self.sfreq_psg[subj] = raw_psg_selected.info['sfreq']
                
                if self.use_cache:
                    save_to_cache(
                        subject=subj,
                        config=self.preprocess_config,
                        mode="psg",
                        raw_data=self.raw_data_psg[subj],
                        sfreq=self.sfreq_psg[subj],
                        ch_names=raw_psg_selected.ch_names,
                        bounds=self.bounds_psg[subj],
                        labels=labels_psg,
                        valid_mask=valid_for_norm_psg,
                        norm_mean=mean_psg,
                        norm_std=std_psg,
                    )

            # --- Decide which epochs to include ---
            if self.mode == "headband":
                mask = self.valid_psg[subj] & self.valid_hb_ai[subj]
            elif self.mode == "psg":
                mask = self.valid_psg[subj]
            else:  # cross
                mask = self.valid_psg[subj] & self.valid_hb_ai[subj]

            epoch_indices = np.where(mask)[0]
            for ei in epoch_indices:
                self.indices.append((subj, int(ei)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        subj, epoch_idx = self.indices[idx]

        # Label
        y = self.labels_psg[subj][epoch_idx]
        y = torch.tensor(y, dtype=torch.long)
        if self.target_transform is not None:
            y = self.target_transform(y)

        if self.mode == "headband":
            start, stop = self.bounds_hb[subj][epoch_idx]
            x_hb = self.raw_data_hb[subj][:, start:stop]
            mean_hb, std_hb = self.norm_stats_hb[subj]  # (None, None) if epoch-wise
            x_hb = apply_znorm_to_epoch(x_hb, mean_hb, std_hb, self.preprocess_config)
            
            x_hb = torch.from_numpy(x_hb).float()   # Shape: (2, 3840)
            if self.transform_hb is not None:
                x_hb = self.transform_hb(x_hb)
            return x_hb, y

        if self.mode == "psg":
            start, stop = self.bounds_psg[subj][epoch_idx]
            x_psg = self.raw_data_psg[subj][:, start:stop]
            mean_psg, std_psg = self.norm_stats_psg[subj]  # (None, None) if epoch-wise
            x_psg = apply_znorm_to_epoch(x_psg, mean_psg, std_psg, self.preprocess_config)
            
            x_psg = torch.from_numpy(x_psg).float()
            if self.transform_psg is not None:
                x_psg = self.transform_psg(x_psg)
            return x_psg, y

        # mode == "cross"
        start_hb, stop_hb = self.bounds_hb[subj][epoch_idx]
        start_psg, stop_psg = self.bounds_psg[subj][epoch_idx]

        x_hb = self.raw_data_hb[subj][:, start_hb:stop_hb]
        x_psg = self.raw_data_psg[subj][:, start_psg:stop_psg]
        
        mean_hb, std_hb = self.norm_stats_hb[subj]
        mean_psg, std_psg = self.norm_stats_psg[subj]
        x_hb = apply_znorm_to_epoch(x_hb, mean_hb, std_hb, self.preprocess_config)
        x_psg = apply_znorm_to_epoch(x_psg, mean_psg, std_psg, self.preprocess_config)

        x_hb = torch.from_numpy(x_hb).float()
        x_psg = torch.from_numpy(x_psg).float()

        if self.transform_hb is not None:
            x_hb = self.transform_hb(x_hb)
        if self.transform_psg is not None:
            x_psg = self.transform_psg(x_psg)

        return x_hb, x_psg, y
    

### not in usage yet, probably needs an update to match new BoasDataset above (25_11_25) ###
class BoasSequenceDataset(Dataset):
    """
    Sequence-level BOAS dataset that wraps BoasDataset.
    
    Returns sequences of consecutive epochs from the same recording (subject/night).
    Sequences never cross recording boundaries to maintain causality for real-time application.
    
    Returns:
        - x_seq: (seq_len, C, T) for single modality or (seq_len, 1, C, T) if add_channel_dim=True
        - y_seq: (seq_len,) labels for each epoch in sequence
    """
    
    def __init__(
        self,
        subjects: Optional[Iterable[str]] = None,
        mode: str = "headband",
        seq_len: int = 20,
        stride: int = 1,
        transform_hb=None,
        transform_psg=None,
        target_transform=None,
        add_channel_dim: bool = True,
        preprocess_config: Optional[PreprocessingConfig] = None,
    ):
        """
        Parameters
        ----------
        subjects : Iterable[str], optional
            Subject IDs to include.
        mode : str
            "headband", "psg", or "cross".
        seq_len : int
            Number of consecutive epochs per sequence.
        stride : int
            Stride between sequence start positions (within same recording).
        transform_hb, transform_psg, target_transform
            Transforms passed to underlying BoasDataset.
        add_channel_dim : bool
            If True, add singleton channel dimension: (seq_len, C, T) -> (seq_len, 1, C, T)
        """
        super().__init__()
        
        # Create underlying epoch-level dataset
        self.epoch_dataset = BoasDataset(
            subjects=subjects,
            mode=mode,
            transform_hb=transform_hb,
            transform_psg=transform_psg,
            target_transform=target_transform,
            preprocess_config=preprocess_config,  # PASS THROUGH
        )
        
        self.mode = mode
        self.seq_len = seq_len
        self.stride = stride
        self.add_channel_dim = add_channel_dim

        # FAST lookup from (subj, epoch_idx) -> dataset index fast
        self.idx_map = {
            key: i for i, key in enumerate(self.epoch_dataset.indices)
        }
        
        # Build sequence index: list of (subject, start_epoch_idx)
        self.sequences: List[Tuple[str, int]] = []
        self._build_sequence_index()
    
    def _build_sequence_index(self):
        """
        Build list of valid sequence start positions.
        Each sequence is (subject, start_epoch_idx) where start_epoch_idx + seq_len
        stays within the same recording.
        """
        # Group epoch indices by subject
        subject_to_indices = {}
        for idx, (subj, epoch_idx) in enumerate(self.epoch_dataset.indices):
            if subj not in subject_to_indices:
                subject_to_indices[subj] = []
            subject_to_indices[subj].append(epoch_idx)
        
        # For each subject, find valid sequence start positions
        for subj, epoch_indices in subject_to_indices.items():
            # Sort epoch indices to ensure consecutive epochs
            epoch_indices = sorted(epoch_indices)
            
            # Find consecutive runs of epochs
            consecutive_runs = []
            if len(epoch_indices) == 0:
                continue
            
            current_run = [epoch_indices[0]]
            for i in range(1, len(epoch_indices)):
                if epoch_indices[i] == current_run[-1] + 1:
                    # Continue current run
                    current_run.append(epoch_indices[i])
                else:
                    # Start new run
                    consecutive_runs.append(current_run)
                    current_run = [epoch_indices[i]]
            consecutive_runs.append(current_run)
            
            # Generate sequences from each consecutive run
            for run in consecutive_runs:
                if len(run) < self.seq_len:
                    continue  # Skip runs shorter than seq_len
                
                # Create sequences with stride
                for start_pos in range(0, len(run) - self.seq_len + 1, self.stride):
                    start_epoch_idx = run[start_pos]
                    self.sequences.append((subj, start_epoch_idx))
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        subj, start_epoch_idx = self.sequences[idx]
        
        # Collect seq_len consecutive epochs
        x_list = []
        y_list = []
        
        for offset in range(self.seq_len):
            epoch_idx = start_epoch_idx + offset
            
            # Find this epoch in the underlying dataset
            # (we know it exists from _build_sequence_index)
            dataset_idx = self.idx_map[(subj, epoch_idx)]
            
            if self.mode == "cross":
                x_hb, x_psg, y = self.epoch_dataset[dataset_idx]
                # For cross mode, return both modalities
                # Stack them or handle as needed
                x_list.append((x_hb, x_psg))
            else:
                x, y = self.epoch_dataset[dataset_idx]
                x_list.append(x)
            
            y_list.append(int(y))
        
        # Stack into sequences
        if self.mode == "cross":
            # Separate stacking for each modality
            x_hb_seq = torch.stack([x[0] for x in x_list], dim=0)  # (seq_len, C, T)
            x_psg_seq = torch.stack([x[1] for x in x_list], dim=0)  # (seq_len, C, T)
            
            if self.add_channel_dim:
                x_hb_seq = x_hb_seq.unsqueeze(1)  # (seq_len, 1, C, T)
                x_psg_seq = x_psg_seq.unsqueeze(1)  # (seq_len, 1, C, T)
            
            y_seq = torch.as_tensor(y_list, dtype=torch.long)  # (seq_len,)
            return x_hb_seq, x_psg_seq, y_seq
        else:
            x_seq = torch.stack(x_list, dim=0)  # (seq_len, C, T)
            
            if self.add_channel_dim:
                x_seq = x_seq.unsqueeze(1)  # (seq_len, 1, C, T)
            
            y_seq = torch.as_tensor(y_list, dtype=torch.long)  # (seq_len,)
            return x_seq, y_seq





if __name__ == "__main__":
    # ds_hb = BoasDataset(subjects=["sub-1", "sub-2"], mode="headband")
    # print("len headband:", len(ds_hb))
    # x_hb, y = ds_hb[0]
    # print("HB epoch shape:", x_hb.shape, "label:", y)

    # ds_psg = BoasDataset(subjects=["sub-1"], mode="psg")
    # print("len psg:", len(ds_psg))
    # x_psg, y = ds_psg[0]
    # print("PSG epoch shape:", x_psg.shape, "label:", y)

    # ds_cross = BoasDataset(subjects=["sub-1"], mode="cross")
    # print("len cross:", len(ds_cross))
    # x_hb, x_psg, y = ds_cross[0]
    # print("cross HB shape:", x_hb.shape, "PSG shape:", x_psg.shape, "label:", y)


    # splits = split_by_pid(seed=42)
    # print("train_subjects:", splits["train_subjects"][:10])
    # print("val_subjects:", splits["val_subjects"][:10])
    # print("test_subjects:", splits["test_subjects"][:10])

    # print("num train pids:", len(splits["train_pids"]))
    # print("num val pids:", len(splits["val_pids"]))
    # print("num test pids:", len(splits["test_pids"]))

    # splits = split_by_pid(seed=42)

    # train_subs = set(splits["train_subjects"])
    # val_subs   = set(splits["val_subjects"])
    # test_subs  = set(splits["test_subjects"])

    # print("train ∩ val:", train_subs & val_subs)
    # print("train ∩ test:", train_subs & test_subs)
    # print("val ∩ test:", val_subs & test_subs)

    # # Check pid separation explicitly
    # _, sub_to_pid = build_pid_mappings()

    # train_pids = {sub_to_pid[s] for s in train_subs}
    # val_pids   = {sub_to_pid[s] for s in val_subs}
    # test_pids  = {sub_to_pid[s] for s in test_subs}
    # print("pid train ∩ val:", train_pids & val_pids)
    # print("pid train ∩ test:", train_pids & test_pids)
    # print("pid val ∩ test:", val_pids & test_pids)


    # from .boa_loader import load_participants_table

    # df = load_participants_table()
    # print("number of rows:", len(df))
    # print("unique participant_id:", df["participant_id"].nunique())
    # print("unique pid:", df["pid"].nunique())
    pass

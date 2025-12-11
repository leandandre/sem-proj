"""
Variable-length sequence dataset for BOAS sleep stage classification.

Extracts consecutive valid epochs until artifacts/disconnections occur,
creating naturally-bounded sequences for GRU-based classification.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from .boa_loader import (
    LABEL_ARTIFACT,
    LABEL_DISCONNECT,
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
)

from .cache import load_from_cache, save_to_cache

N_HB_CHANNELS = 2
N_PSG_CHANNELS = 6


class BoasVariableLengthSequenceDataset(Dataset):
    """
    Variable-length sequence dataset for BOAS.
    USED IT FOR THE FINAL MODEL, CHANGED FROM FIXED LENGTH TO NOW VARIABLE LENGTH
    
    Extracts consecutive valid epochs (labels 0-4) until encountering:
    - Headband: AI artifact (label -2)
    - PSG: Disconnection (label 8)
    
    Each sequence represents a continuous recording segment without interruptions.
    
    Returns:
        For mode="headband" or "psg":
            - x_seq: (L, C, T) where L varies per sequence
            - y_seq: (L,) labels
            
        For mode="cross":
            - x_hb_seq: (L, C_hb, T)
            - x_psg_seq: (L, C_psg, T)
            - y_seq: (L,)
    """
    
    def __init__(
        self,
        subjects: Optional[Iterable[str]] = None,
        mode: str = "headband",
        min_seq_len: int = 5,
        transform_hb=None,
        transform_psg=None,
        target_transform=None,
        preprocess_config: Optional[PreprocessingConfig] = None,
        use_cache: bool = True,
    ):
        """
        Parameters
        ----------
        subjects : Iterable[str], optional
            Subject IDs to include.
        mode : str
            "headband", "psg", or "cross".
        min_seq_len : int
            Minimum number of consecutive valid epochs to form a sequence.
            Sequences shorter than this are discarded.
        transform_hb, transform_psg, target_transform
            Transforms to apply to data.
        preprocess_config : PreprocessingConfig, optional
            Preprocessing configuration.
        use_cache : bool
            Whether to use cached preprocessed data.
        """
        super().__init__()
        
        if subjects is None:
            subjects = list_boas_subjects()
        self.subjects: List[str] = list(subjects)
        
        assert mode in {"headband", "psg", "cross"}
        self.mode = mode
        self.min_seq_len = min_seq_len
        
        self.preprocess_config = preprocess_config or PreprocessingConfig.no_preprocessing()
        self.use_cache = use_cache
        
        self.transform_hb = transform_hb
        self.transform_psg = transform_psg
        self.target_transform = target_transform
        
        # Per-subject caches
        self.raw_data_hb = {}
        self.raw_data_psg = {}
        self.sfreq_hb = {}
        self.sfreq_psg = {}
        
        self.labels_psg = {}
        self.labels_hb_ai = {}  # Need headband AI labels for artifact detection
        self.bounds_hb = {}
        self.bounds_psg = {}
        
        # Normalization statistics
        self.norm_stats_hb = {}
        self.norm_stats_psg = {}
        
        # List of sequences: (subject, start_epoch_idx, end_epoch_idx)
        self.sequences: List[Tuple[str, int, int]] = []
        
        self._build_sequences()
    
    def _build_sequences(self):
        """
        Build variable-length sequences by finding continuous runs of valid epochs.
        
        For each subject:
        1. Load all epoch labels (including invalid ones)
        2. Identify runs of consecutive valid epochs
        3. Each run becomes one sequence
        """
        for subj in self.subjects:
            print(f"Building sequences for {subj}...")
            
            # Load PSG labels (ground truth)
            labels_psg, onset_psg, duration_psg, is_valid_psg, is_special_psg = load_labels(
                subj, source="psg", label_type="hum"
            )
            self.labels_psg[subj] = labels_psg
            
            # Load headband AI labels (for artifact detection)
            labels_hb_ai, onset_hb, duration_hb, is_valid_hb, is_special_hb = load_labels(
                subj, source="headband", label_type="ai"
            )
            self.labels_hb_ai[subj] = labels_hb_ai
            
            # Try to load from cache
            cache_loaded_hb = False
            cache_loaded_psg = False
            
            if self.use_cache:
                if self.mode in {"headband", "cross"}:
                    cache_data_hb = load_from_cache(subj, self.preprocess_config, mode="headband")
                    if cache_data_hb is not None:
                        self.raw_data_hb[subj] = cache_data_hb['raw_data']
                        self.sfreq_hb[subj] = cache_data_hb['sfreq']
                        self.bounds_hb[subj] = cache_data_hb['bounds']
                        self.norm_stats_hb[subj] = (cache_data_hb['norm_mean'], cache_data_hb['norm_std'])
                        cache_loaded_hb = True
                        print(f"  Loaded headband from cache")
                
                if self.mode in {"psg", "cross"}:
                    cache_data_psg = load_from_cache(subj, self.preprocess_config, mode="psg")
                    if cache_data_psg is not None:
                        self.raw_data_psg[subj] = cache_data_psg['raw_data']
                        self.sfreq_psg[subj] = cache_data_psg['sfreq']
                        self.bounds_psg[subj] = cache_data_psg['bounds']
                        self.norm_stats_psg[subj] = (cache_data_psg['norm_mean'], cache_data_psg['norm_std'])
                        cache_loaded_psg = True
                        print(f"  Loaded PSG from cache")
            
            # Load and preprocess if not cached
            if self.mode in {"headband", "cross"} and not cache_loaded_hb:
                raw_hb = load_headband_raw(subj)
                raw_hb = preprocess_raw_basic(raw_hb, self.preprocess_config)
                self.bounds_hb[subj] = compute_epoch_sample_bounds(raw_hb, onset_hb, duration_hb)
                
                raw_hb_selected = raw_hb.copy()
                raw_hb_selected.pick(raw_hb_selected.ch_names[:N_HB_CHANNELS])
                
                # Compute normalization stats using only valid epochs
                valid_for_norm_hb = is_valid_psg & is_valid_hb
                mean_hb, std_hb = compute_subject_normalization_stats(
                    raw_hb_selected,
                    valid_for_norm_hb,
                    self.bounds_hb[subj],
                    self.preprocess_config,
                )
                self.norm_stats_hb[subj] = (mean_hb, std_hb)
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
                        valid_mask=is_valid_psg,
                        valid_hb_mask=valid_for_norm_hb,
                        norm_mean=mean_hb,
                        norm_std=std_hb,
                    )
                    print(f"  Saved headband to cache")
            
            if self.mode in {"psg", "cross"} and not cache_loaded_psg:
                raw_psg = load_psg_raw(subj)
                raw_psg = preprocess_raw_basic(raw_psg, self.preprocess_config)
                self.bounds_psg[subj] = compute_epoch_sample_bounds(raw_psg, onset_psg, duration_psg)
                
                raw_psg_selected = raw_psg.copy()
                raw_psg_selected.pick(raw_psg_selected.ch_names[:N_PSG_CHANNELS])
                
                valid_for_norm_psg = is_valid_psg
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
                        valid_hb_mask=None,
                        norm_mean=mean_psg,
                        norm_std=std_psg,
                    )
                    print(f"  Saved PSG to cache")
            
            # Extract sequences based on mode
            sequences = self._extract_sequences_for_subject(
                subj, labels_psg, labels_hb_ai, is_valid_psg, is_valid_hb
            )
            
            print(f"  Found {len(sequences)} sequences (min_len={self.min_seq_len})")
            self.sequences.extend(sequences)
        
        print(f"\nTotal sequences across all subjects: {len(self.sequences)}")
        
        # Print sequence length statistics
        if len(self.sequences) > 0:
            seq_lengths = [end - start for _, start, end in self.sequences]
            print(f"Sequence length stats:")
            print(f"  Min: {np.min(seq_lengths)}")
            print(f"  Max: {np.max(seq_lengths)}")
            print(f"  Mean: {np.mean(seq_lengths):.1f}")
            print(f"  Median: {np.median(seq_lengths):.1f}")
    
    def _extract_sequences_for_subject(
        self,
        subj: str,
        labels_psg: np.ndarray,
        labels_hb_ai: np.ndarray,
        is_valid_psg: np.ndarray,
        is_valid_hb: np.ndarray,
    ) -> List[Tuple[str, int, int]]:
        """
        Extract variable-length sequences for one subject.
        
        A sequence is a continuous run of epochs where:
        - PSG label is valid (0-4)
        - Headband AI label is valid (not -2 artifact) if using headband mode
        
        Returns list of (subject, start_epoch_idx, end_epoch_idx) tuples.
        end_epoch_idx is exclusive (Python convention).
        """
        sequences = []
        
        # Determine which epochs are valid based on mode
        if self.mode == "headband":
            # Need both PSG valid AND headband not artifact
            valid_mask = is_valid_psg & is_valid_hb
        elif self.mode == "psg":
            # Only need PSG valid (not disconnection)
            valid_mask = is_valid_psg
        else:  # cross
            # Need both modalities valid
            valid_mask = is_valid_psg & is_valid_hb
        
        # Find continuous runs of valid epochs
        n_epochs = len(labels_psg)
        
        if n_epochs == 0:
            return sequences
        
        # Find run boundaries
        in_run = False
        run_start = None
        
        for i in range(n_epochs):
            if valid_mask[i]:
                if not in_run:
                    # Start new run
                    run_start = i
                    in_run = True
            else:
                if in_run:
                    # End current run
                    run_end = i  # Exclusive
                    run_length = run_end - run_start
                    
                    if run_length >= self.min_seq_len:
                        sequences.append((subj, run_start, run_end))
                    
                    in_run = False
        
        # Handle run extending to end of recording
        if in_run:
            run_end = n_epochs
            run_length = run_end - run_start
            
            if run_length >= self.min_seq_len:
                sequences.append((subj, run_start, run_end))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        """
        Get one variable-length sequence.
        
        Returns:
            For mode="headband" or "psg":
                x_seq: (L, C, T) tensor
                y_seq: (L,) tensor
                
            For mode="cross":
                x_hb_seq: (L, C_hb, T) tensor
                x_psg_seq: (L, C_psg, T) tensor
                y_seq: (L,) tensor
        """
        subj, start_idx, end_idx = self.sequences[idx]
        seq_len = end_idx - start_idx
        
        # Collect all epochs in this sequence
        if self.mode == "headband":
            x_list = []
            y_list = []
            
            mean_hb, std_hb = self.norm_stats_hb[subj]
            
            for epoch_idx in range(start_idx, end_idx):
                start, stop = self.bounds_hb[subj][epoch_idx]
                x_epoch = self.raw_data_hb[subj][:, start:stop]
                x_epoch = apply_znorm_to_epoch(x_epoch, mean_hb, std_hb, self.preprocess_config)
                x_epoch = torch.from_numpy(x_epoch).float()
                
                if self.transform_hb is not None:
                    x_epoch = self.transform_hb(x_epoch)
                
                x_list.append(x_epoch)
                
                y = self.labels_psg[subj][epoch_idx]
                y_list.append(int(y))
            
            x_seq = torch.stack(x_list, dim=0)  # (L, C, T)
            y_seq = torch.tensor(y_list, dtype=torch.long)  # (L,)
            
            if self.target_transform is not None:
                y_seq = self.target_transform(y_seq)
            
            return x_seq, y_seq
        
        elif self.mode == "psg":
            x_list = []
            y_list = []
            
            mean_psg, std_psg = self.norm_stats_psg[subj]
            
            for epoch_idx in range(start_idx, end_idx):
                start, stop = self.bounds_psg[subj][epoch_idx]
                x_epoch = self.raw_data_psg[subj][:, start:stop]
                x_epoch = apply_znorm_to_epoch(x_epoch, mean_psg, std_psg, self.preprocess_config)
                x_epoch = torch.from_numpy(x_epoch).float()
                
                if self.transform_psg is not None:
                    x_epoch = self.transform_psg(x_epoch)
                
                x_list.append(x_epoch)
                
                y = self.labels_psg[subj][epoch_idx]
                y_list.append(int(y))
            
            x_seq = torch.stack(x_list, dim=0)  # (L, C, T)
            y_seq = torch.tensor(y_list, dtype=torch.long)  # (L,)
            
            if self.target_transform is not None:
                y_seq = self.target_transform(y_seq)
            
            return x_seq, y_seq
        
        else:  # mode == "cross"
            x_hb_list = []
            x_psg_list = []
            y_list = []
            
            mean_hb, std_hb = self.norm_stats_hb[subj]
            mean_psg, std_psg = self.norm_stats_psg[subj]
            
            for epoch_idx in range(start_idx, end_idx):
                # Headband
                start_hb, stop_hb = self.bounds_hb[subj][epoch_idx]
                x_hb_epoch = self.raw_data_hb[subj][:, start_hb:stop_hb]
                x_hb_epoch = apply_znorm_to_epoch(x_hb_epoch, mean_hb, std_hb, self.preprocess_config)
                x_hb_epoch = torch.from_numpy(x_hb_epoch).float()
                
                if self.transform_hb is not None:
                    x_hb_epoch = self.transform_hb(x_hb_epoch)
                
                # PSG
                start_psg, stop_psg = self.bounds_psg[subj][epoch_idx]
                x_psg_epoch = self.raw_data_psg[subj][:, start_psg:stop_psg]
                x_psg_epoch = apply_znorm_to_epoch(x_psg_epoch, mean_psg, std_psg, self.preprocess_config)
                x_psg_epoch = torch.from_numpy(x_psg_epoch).float()
                
                if self.transform_psg is not None:
                    x_psg_epoch = self.transform_psg(x_psg_epoch)
                
                x_hb_list.append(x_hb_epoch)
                x_psg_list.append(x_psg_epoch)
                
                y = self.labels_psg[subj][epoch_idx]
                y_list.append(int(y))
            
            x_hb_seq = torch.stack(x_hb_list, dim=0)  # (L, C_hb, T)
            x_psg_seq = torch.stack(x_psg_list, dim=0)  # (L, C_psg, T)
            y_seq = torch.tensor(y_list, dtype=torch.long)  # (L,)
            
            if self.target_transform is not None:
                y_seq = self.target_transform(y_seq)
            
            return x_hb_seq, x_psg_seq, y_seq


def variable_length_collate_fn(batch, mode="headband"):
    """
    Collate function for variable-length sequences.
    
    Pads sequences to max length in batch and creates attention masks.
    
    Parameters
    ----------
    batch : list
        List of dataset items (variable-length sequences).
    mode : str
        Dataset mode ("headband", "psg", or "cross").
    
    Returns
    -------
    For mode="headband" or "psg":
        padded_seqs: (B, L_max, C, T) tensor
        padded_labels: (B, L_max) tensor with -100 for padding
        lengths: (B,) tensor of actual sequence lengths
        
    For mode="cross":
        padded_hb: (B, L_max, C_hb, T) tensor
        padded_psg: (B, L_max, C_psg, T) tensor
        padded_labels: (B, L_max) tensor
        lengths: (B,) tensor
    """
    # Sort by length (descending) for pack_padded_sequence compatibility
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    
    if mode in {"headband", "psg"}:
        sequences = [item[0] for item in batch]  # List of (L_i, C, T)
        labels = [item[1] for item in batch]      # List of (L_i,)
        
        lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
        max_len = lengths[0].item()  # Already sorted
        
        C = sequences[0].shape[1]
        T = sequences[0].shape[2]
        B = len(batch)
        
        # Pad sequences
        padded_seqs = torch.zeros(B, max_len, C, T, dtype=sequences[0].dtype)
        padded_labels = torch.full((B, max_len), -100, dtype=torch.long)
        
        for i, (seq, label, length) in enumerate(zip(sequences, labels, lengths)):
            padded_seqs[i, :length] = seq
            padded_labels[i, :length] = label
        
        return padded_seqs, padded_labels, lengths
    
    else:  # mode == "cross"
        hb_seqs = [item[0] for item in batch]     # List of (L_i, C_hb, T)
        psg_seqs = [item[1] for item in batch]    # List of (L_i, C_psg, T)
        labels = [item[2] for item in batch]      # List of (L_i,)
        
        lengths = torch.tensor([seq.shape[0] for seq in hb_seqs], dtype=torch.long)
        max_len = lengths[0].item()
        
        C_hb = hb_seqs[0].shape[1]
        C_psg = psg_seqs[0].shape[1]
        T = hb_seqs[0].shape[2]
        B = len(batch)
        
        padded_hb = torch.zeros(B, max_len, C_hb, T, dtype=hb_seqs[0].dtype)
        padded_psg = torch.zeros(B, max_len, C_psg, T, dtype=psg_seqs[0].dtype)
        padded_labels = torch.full((B, max_len), -100, dtype=torch.long)
        
        for i, (hb_seq, psg_seq, label, length) in enumerate(zip(hb_seqs, psg_seqs, labels, lengths)):
            padded_hb[i, :length] = hb_seq
            padded_psg[i, :length] = psg_seq
            padded_labels[i, :length] = label
        
        return padded_hb, padded_psg, padded_labels, lengths
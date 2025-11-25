import pandas as pd
import numpy as np
from pathlib import Path
# import PyQt5
import matplotlib
matplotlib.use("Qt5Agg")
import mne
from typing import Optional, Tuple
from sem_proj.data.preprocessing import PreprocessingConfig


# Project root = .../sem-proj
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BOAS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "boas"


def list_boas_subjects() -> list[str]:
    """Return all BOAS subject IDs, e.g. ['sub-1', 'sub-2', ...]."""
    return sorted(
        p.name for p in BOAS_RAW_DIR.glob("sub-*") if p.is_dir()
    )

def get_headband_edf(subject: str) -> Path:
    subj_dir = BOAS_RAW_DIR / subject / "eeg"
    matches = list(subj_dir.glob("*_acq-headband_eeg.edf"))
    if not matches:
        raise FileNotFoundError(f"No headband EDF for {subject}")
    return matches[0]

def get_psg_edf(subject: str) -> Path:
    subj_dir = BOAS_RAW_DIR / subject / "eeg"
    matches = list(subj_dir.glob("*_acq-psg_eeg.edf"))
    if not matches:
        raise FileNotFoundError(f"No PSG EDF for {subject}")
    return matches[0]

# FIXED: Removed preprocessing from these functions - it's done in datasets.py
def load_headband_raw(subject: str, preprocess_config: Optional[PreprocessingConfig] = None) -> mne.io.BaseRaw:
    """Load raw headband EDF. Preprocessing is done separately in datasets.py."""
    raw = mne.io.read_raw_edf(get_headband_edf(subject), preload=True)
    return raw

def load_psg_raw(subject: str, preprocess_config: Optional[PreprocessingConfig] = None) -> mne.io.BaseRaw:
    """Load raw PSG EDF. Preprocessing is done separately in datasets.py."""
    raw = mne.io.read_raw_edf(get_psg_edf(subject), preload=True)
    return raw

def get_psg_events_tsv(subject: str) -> Path:
    """Return path to the PSG events.tsv for a subject."""
    eeg_dir = BOAS_RAW_DIR / subject / "eeg"
    path = eeg_dir / f"{subject}_task-Sleep_acq-psg_events.tsv"
    if not path.exists():
        raise FileNotFoundError(f"PSG events file not found for {subject}: {path}")
    return path

def get_headband_events_tsv(subject: str) -> Path:
    """Return path to the headband events.tsv for a subject."""
    eeg_dir = BOAS_RAW_DIR / subject / "eeg"
    path = eeg_dir / f"{subject}_task-Sleep_acq-headband_events.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Headband events file not found for {subject}: {path}")
    return path


LABEL_WAKE = 0
LABEL_N1 = 1
LABEL_N2 = 2
LABEL_N3 = 3
LABEL_REM = 4
LABEL_DISCONNECT = 8   # human only
LABEL_ARTIFACT = -2    # AI only

def load_labels(
    subject: str,
    source: str = "psg",     # "psg" or "headband"
    label_type: str = "hum", # "hum" or "ai"
):
    """
    Load sleep stage labels for one subject from BOAS event files.

    Parameters
    ----------
    subject : str
        Subject ID, e.g. "sub-1".
    source : {"psg", "headband"}
        Which recording the events file belongs to.
    label_type : {"hum", "ai"}
        Human consensus ("hum") or AI-generated labels ("ai").

    Returns
    -------
    labels : np.ndarray, shape (N_epochs,)
        Integer sleep stage labels per epoch.
    onset : np.ndarray, shape (N_epochs,)
        Onset (in seconds) of each epoch.
    duration : np.ndarray, shape (N_epochs,)
        Duration (in seconds) of each epoch.
    is_valid : np.ndarray[bool], shape (N_epochs,)
        True for epochs with labels in {0,1,2,3,4}.
    is_special : np.ndarray[bool], shape (N_epochs,)
        True for epochs with labels in {8, -2}.
    """
    if source == "psg":
        events_path = get_psg_events_tsv(subject)
    elif source == "headband":
        events_path = get_headband_events_tsv(subject)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    df = pd.read_csv(events_path, sep="\t")

    if label_type == "hum":
        col = "stage_hum"
    elif label_type == "ai":
        col = "stage_ai"
    else:
        raise ValueError(f"Unknown label_type: {label_type!r}")

    if col not in df.columns:
        raise KeyError(f"Column {col!r} not found in {events_path}")

    labels = df[col].to_numpy()
    onset = df["onset"].to_numpy() if "onset" in df.columns else None
    duration = df["duration"].to_numpy() if "duration" in df.columns else None

    # Masks
    special_codes = np.array([LABEL_DISCONNECT, LABEL_ARTIFACT])
    is_special = np.isin(labels, special_codes)
    is_valid = ~is_special

    # Optional debugging: check epoch duration distribution
    if duration is not None:
        unique_durs = np.unique(np.round(duration, 1))
        print(f"[{subject}] unique epoch durations (rounded): {unique_durs}")

    return labels, onset, duration, is_valid, is_special


def compute_epoch_sample_bounds(
    raw: mne.io.BaseRaw,
    onset: np.ndarray,
    duration: np.ndarray,
) -> np.ndarray:
    """
    Compute [start, stop) sample indices for each epoch.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded raw signal (headband or PSG).
    onset : np.ndarray, shape (N_epochs,)
        Epoch onset times in seconds (from events.tsv).
    duration : np.ndarray, shape (N_epochs,)
        Epoch duration in seconds (should be ~30).

    Returns
    -------
    bounds : np.ndarray, shape (N_epochs, 2)
        Each row is [start_sample, stop_sample] (Python-style, stop exclusive).
    """
    sfreq: float = raw.info["sfreq"]          # sampling frequency in Hz
    first_samp: int = raw.first_samp         # usually 0 for EDF
    print(f"Sampling frequency: {sfreq} Hz, first_samp: {first_samp}")

    starts = (onset * sfreq).astype(int) + first_samp
    stops = ((onset + duration) * sfreq).astype(int) + first_samp

    bounds = np.stack([starts, stops], axis=1)
    return bounds


def load_participants_table() -> pd.DataFrame:
    """
    Load participants.tsv from BOAS root.

    Returns
    -------
    df : pd.DataFrame
        Must contain columns 'participant_id' (e.g. 'sub-1') and 'pid' (unique person ID).
    """
    path = BOAS_RAW_DIR / "participants.tsv"
    if not path.exists():
        raise FileNotFoundError(f"participants.tsv not found at {path}")
    df = pd.read_csv(path, sep="\t")
    if "participant_id" not in df.columns or "pid" not in df.columns:
        raise KeyError("participants.tsv must contain 'participant_id' and 'pid' columns")
    return df

def build_pid_mappings(df: pd.DataFrame | None = None):
    """
    Build mapping between pids and subject IDs.

    Returns
    -------
    pid_to_subs : dict[pid, list[str]]
        Maps each unique pid to its list of 'sub-*' IDs.
    sub_to_pid : dict[str, pid]
        Maps 'sub-*' -> pid.
    """
    if df is None:
        df = load_participants_table()

    # group by pid, collect participant_id
    grouped = df.groupby("pid")["participant_id"].apply(list)

    pid_to_subs: dict = {}
    for pid, subs in grouped.items():
        # sort to keep deterministic order per pid
        pid_to_subs[pid] = sorted(subs)

    sub_to_pid: dict = {
        sub: pid
        for pid, subs in pid_to_subs.items()
        for sub in subs
    }

    return pid_to_subs, sub_to_pid














# General EDF loading utilities (used for initial exploration only)
def find_edf_files(subject: str) -> list[Path]:
    """
    Return all EDF files for a given subject.

    Parameters
    ----------
    subject : str
        Subject ID like 'sub-1'.

    Returns
    -------
    list[Path]
        List of EDF file paths.
    """
    subj_dir = BOAS_RAW_DIR / subject
    return sorted(subj_dir.rglob("*.edf"))


def load_raw_edf(edf_path: Path) -> mne.io.BaseRaw:
    """
    Load a single EDF file with MNE.

    Parameters
    ----------
    edf_path : Path
        Path to an EDF file.

    Returns
    -------
    mne.io.BaseRaw
        Loaded raw object.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    return raw



if __name__ == "__main__":
    ### Some small tests for myself ###
    # import matplotlib.pyplot as plt
    # print("BOAS_RAW_DIR:", BOAS_RAW_DIR)

    # subs = list_boas_subjects()
    # print("Subjects:", subs)

    # raw = None
    # if subs:
    #     edf_files = find_edf_files(subs[1 - 1])
    #     print("EDF files for first subject:", edf_files)

    #     if edf_files:
    #         raw = load_raw_edf(edf_files[0])
    #         print(raw)
    #         print(raw.info)
    #         tensor = raw.get_data()
    #         print("Data tensor shape:", tensor.shape)
    # if raw is None:
    #     raise SystemExit("No EDF loaded for demo plotting.")

    # sfreq = raw.info.get('sfreq')           # sampling frequency (Hz)
    # ch_names = raw.info.get('ch_names')     # list of channel names
    # nchans = len(ch_names)
    # nchans_info = raw.info.get('nchan')     # number of channels
    # bads = raw.info.get('bads')             # channels marked bad
    # meas_date = raw.info.get('meas_date')   # may be tuple or timestamp
    # print(sfreq, ch_names, nchans, nchans_info, bads, meas_date)

    # raw.plot(duration=10, n_channels=nchans, scalings='auto')
    # plt.show(block=True)
    
    
    # label loading test
    subs = list_boas_subjects()
    subj = subs[1 - 1]
    raw_headband = load_headband_raw(subj)
    labels, onset, duration, is_valid, is_special = load_labels(
        subj,
        source="headband",
        label_type="ai",
    )

    bounds_headband = compute_epoch_sample_bounds(raw_headband, onset, duration)
    print("bounds_headband shape:", bounds_headband.shape)

    # Check that each epoch has correct length in samples
    epoch_lengths = bounds_headband[:, 1] - bounds_headband[:, 0]
    print("unique epoch lengths (samples):", np.unique(epoch_lengths))

    # Inspect one epoch
    i = 14
    start_i, stop_i = bounds_headband[i]
    x_headband = raw_headband.get_data(start=start_i, stop=stop_i)
    print("one epoch shape:", x_headband.shape)  # (n_channels_headband, 7680)
    print(f"label[{i}]:", labels[i], f"is_valid[{i}]:", is_valid[i])

    
        

    
    
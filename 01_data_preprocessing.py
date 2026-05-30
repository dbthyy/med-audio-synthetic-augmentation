"""
01_data_preprocessing.py
========================
Load Coswara dataset, extract mel-spectrograms, split theo GroupShuffleSplit,
và lưu dataset.pt để các file sau dùng chung.

"""

import os
import json
import time
import numpy as np
import pandas as pd
import librosa
import torch
from sklearn.model_selection import GroupShuffleSplit

# CONFIG 
BASE_DIR    = "/content/Coswara-Data/Extracted_data"
TARGET_SR   = 16000
FIX_LEN     = 3 * TARGET_SR  
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
FMIN        = 0
FMAX        = 8000
SAVE_PATH   = "dataset.pt"
RANDOM_SEED = 42


# 1. BUILD METADATA DATAFRAME 
def build_dataframe(base_dir: str) -> pd.DataFrame:
    """collect cough-*.wav paths + metadata."""
    rows = []
    for root, _, files in os.walk(base_dir):
        if "metadata.json" not in files:
            continue
        with open(os.path.join(root, "metadata.json")) as f:
            meta = json.load(f)
        meta_flat = pd.json_normalize(meta).to_dict(orient="records")[0]
        for file in files:
            if file.startswith("cough-") and file.endswith(".wav"):
                row = meta_flat.copy()
                row["path"]      = os.path.join(root, file)
                row["file_name"] = file
                row["user_id"]   = root.split("/")[-1]
                rows.append(row)
    return pd.DataFrame(rows)


# 2. FILTER & LABEL 
def filter_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only healthy / COVID-positive samples."""
    df = df[
        (df["covid_status"] == "healthy") |
        (df["covid_status"].str.startswith("positive"))
    ].copy()
    df["label"] = df["covid_status"].apply(lambda x: 0 if x == "healthy" else 1)
    print(f"[Data] Total samples after filter: {len(df)}")
    print(df["covid_status"].value_counts())
    return df.reset_index(drop=True)


# 3. AUDIO PROCESSING 
def process_audio(path: str) -> np.ndarray:
    """Load WAV, resample to 16 kHz mono, center-crop/zero-pad to 3 s."""
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    if len(audio) > FIX_LEN:
        start = (len(audio) - FIX_LEN) // 2
        audio = audio[start : start + FIX_LEN]
    else:
        audio = np.pad(audio, (0, FIX_LEN - len(audio)))
    return audio


def to_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Convert waveform → log-mel spectrogram (128 × T)."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        window="hann", n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max)


# 4. BUILD RAW WAVEFORM ARRAYS 
def load_waveforms(df: pd.DataFrame):
    """Return X (N, 48000) and y (N,), filtering out silent files."""
    X, y = [], []
    for _, row in df.iterrows():
        audio = process_audio(row["path"])
        X.append(audio)
        y.append(row["label"])
    X = np.array(X)
    y = np.array(y)

    # Remove all-zero (silent) files
    valid = [i for i in range(len(X)) if not np.all(X[i] == 0)]
    print(f"[Data] Removed {len(X) - len(valid)} silent files.")
    return X[valid], y[valid], df.iloc[valid].reset_index(drop=True)


# 5. TRAIN / VAL / TEST SPLIT
def group_split(X: np.ndarray, y: np.ndarray, df: pd.DataFrame):
    """
    GroupShuffleSplit so that a single user never appears in multiple splits.
    Prevents data leakage that StratifiedShuffleSplit would cause.
    Ratio: 70 / 15 / 15
    """
    groups = df["user_id"]
    gss = GroupShuffleSplit(test_size=0.30, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(gss.split(X, y, groups))

    gss2 = GroupShuffleSplit(test_size=0.50, random_state=RANDOM_SEED)
    val_sub, test_sub = next(
        gss2.split(X[temp_idx], y[temp_idx], groups.iloc[temp_idx])
    )
    val_idx  = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]

    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        u, c = np.unique(y[idx], return_counts=True)
        print(f"[Split] {name}: {dict(zip(u, c))}")
    return train_idx, val_idx, test_idx


# 6. MEL-SPECTROGRAM ARRAYS 
def build_mel_arrays(X: np.ndarray, train_idx, val_idx, test_idx):
    """
    Convert waveforms → mel-spectrograms → (H, W, 1) shape.
    Z-score normalization computed on training set only.
    """
    def _mel_batch(indices):
        return np.array([to_mel_spectrogram(X[i]) for i in indices])[..., np.newaxis]

    X_train = _mel_batch(train_idx)
    X_val   = _mel_batch(val_idx)
    X_test  = _mel_batch(test_idx)

    # Per-channel z-score (computed on train)
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std  = X_train.std(axis=(0, 1, 2), keepdims=True)

    X_train = (X_train - mean) / (std + 1e-8)
    X_val   = (X_val   - mean) / (std + 1e-8)
    X_test  = (X_test  - mean) / (std + 1e-8)

    print(f"[Mel] Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")
    return X_train, X_val, X_test, mean, std


# 7. SAVE 
def save_dataset(X_train, X_val, X_test, y, train_idx, val_idx, test_idx, mean, std, path):
    torch.save({
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "X_val":   torch.tensor(X_val,   dtype=torch.float32),
        "X_test":  torch.tensor(X_test,  dtype=torch.float32),
        "y_train": torch.tensor(y[train_idx], dtype=torch.long),
        "y_val":   torch.tensor(y[val_idx],   dtype=torch.long),
        "y_test":  torch.tensor(y[test_idx],  dtype=torch.long),
        "norm_mean": mean,
        "norm_std":  std,
    }, path)
    print(f"[Save] Dataset saved → {path}")


#  MAIN 
if __name__ == "__main__":
    t0 = time.time()

    df  = build_dataframe(BASE_DIR)
    df  = filter_and_label(df)
    X, y, df = load_waveforms(df)
    train_idx, val_idx, test_idx = group_split(X, y, df)
    X_train, X_val, X_test, mean, std = build_mel_arrays(X, train_idx, val_idx, test_idx)
    save_dataset(X_train, X_val, X_test, y, train_idx, val_idx, test_idx, mean, std, SAVE_PATH)

    print(f"\n[Done] Preprocessing completed in {(time.time()-t0)/60:.1f} min")
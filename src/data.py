"""RAVDESS discovery and label utilities."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


def discover_ravdess(dataset_dir: str | Path) -> tuple[list[Path], list[str]]:
    """Find RAVDESS speech WAVs and decode emotion from its standard filename."""
    paths = sorted(Path(dataset_dir).rglob("*.wav"))
    labels, valid_paths = [], []
    for path in paths:
        parts = path.stem.split("-")
        # Modality 03 is audio-only speech; field 3 is emotion.
        if len(parts) == 7 and parts[0] == "03" and parts[2] in EMOTIONS:
            valid_paths.append(path)
            labels.append(EMOTIONS[parts[2]])
    if not valid_paths:
        raise FileNotFoundError(
            "No RAVDESS speech WAV files found. Expected files such as 03-01-05-01-01-01-01.wav."
        )
    return valid_paths, labels


def split_indices(labels: list[str], seed: int = 42):
    """Create stratified 70/15/15 train, validation, and test index arrays."""
    all_idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.30, random_state=seed, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    return train_idx, val_idx, test_idx

"""Feature extraction shared by training and the Streamlit application."""
from __future__ import annotations

from pathlib import Path
import librosa
import numpy as np

SAMPLE_RATE = 22_050
MAX_FRAMES = 180
N_MFCC = 40
N_MELS = 64


def extract_features(audio_path: str | Path) -> np.ndarray:
    """Return a fixed-length (frames, 116) MFCC/chroma/mel feature sequence."""
    signal, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    # Trimming reduces leading/trailing silence and makes utterances more comparable.
    signal, _ = librosa.effects.trim(signal, top_db=30)
    if signal.size == 0:
        raise ValueError("The audio file contains no usable signal.")

    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=signal, sr=SAMPLE_RATE, n_chroma=12)
    mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    sequence = np.vstack([mfcc, chroma, mel_db]).T.astype(np.float32)

    # Models need equal-length inputs. Pad with the final frame (not zeros) for short clips.
    if len(sequence) >= MAX_FRAMES:
        return sequence[:MAX_FRAMES]
    return np.pad(sequence, ((0, MAX_FRAMES - len(sequence)), (0, 0)), mode="edge")

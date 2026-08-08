"""
Shared audio loading + feature primitives used by both the classical-ML
feature extraction pipeline and the deep-learning (CNN) spectrogram pipeline.
"""
import numpy as np
import librosa

SR = 22050
CLIP_SECONDS = 3.5          # fixed clip length everything gets padded/truncated to
N_MELS = 128
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
SPEC_FRAMES = int(np.ceil(CLIP_SECONDS * SR / HOP_LENGTH))  # fixed width for CNN input


def load_fixed_length(path: str, sr: int = SR, seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Load audio, trim leading/trailing silence, pad or truncate to a fixed length."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    target_len = int(seconds * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def augment_audio(y: np.ndarray, sr: int = SR, rng: np.random.Generator = None) -> np.ndarray:
    """Random pitch shift / time stretch / background noise. Training data only."""
    if rng is None:
        rng = np.random.default_rng()
    choice = rng.integers(0, 3)
    if choice == 0:
        steps = rng.uniform(-2.0, 2.0)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    elif choice == 1:
        rate = rng.uniform(0.85, 1.15)
        stretched = librosa.effects.time_stretch(y, rate=rate)
        target_len = len(y)
        if len(stretched) < target_len:
            stretched = np.pad(stretched, (0, target_len - len(stretched)))
        else:
            stretched = stretched[:target_len]
        y = stretched
    else:
        noise_amp = rng.uniform(0.001, 0.01) * np.max(np.abs(y) + 1e-9)
        y = y + noise_amp * rng.standard_normal(len(y))
    return y.astype(np.float32)


def extract_classical_features(y: np.ndarray, sr: int = SR) -> dict:
    """MFCCs, Chroma STFT, Mel spectrogram, ZCR, RMS -> mean+std scalar feature dict."""
    feats = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(chroma.shape[0]):
        feats[f"chroma{i+1}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma{i+1}_std"] = float(np.std(chroma[i]))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel)
    feats["mel_mean"] = float(np.mean(mel_db))
    feats["mel_std"] = float(np.std(mel_db))
    # a handful of per-band summary stats (full 128-band mean/std would over-inflate the
    # classical feature vector relative to MFCC/chroma; band-reduced summary keeps signal)
    band_means = mel_db.mean(axis=1)
    for i in range(0, N_MELS, 8):  # 16 coarse bands
        feats[f"melband{i}_mean"] = float(band_means[i])

    zcr = librosa.feature.zero_crossing_rate(y)
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))

    rms = librosa.feature.rms(y=y)
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))

    return feats


def extract_mel_spectrogram_image(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Fixed-shape (N_MELS, SPEC_FRAMES) log-mel spectrogram, min-max normalized to [0,1]."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < SPEC_FRAMES:
        pad = SPEC_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant", constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :SPEC_FRAMES]
    # normalize to [0, 1] per-clip
    mn, mx = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - mn) / (mx - mn + 1e-9)
    return mel_norm.astype(np.float32)

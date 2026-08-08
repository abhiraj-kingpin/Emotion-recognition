"""
Shared inference module — loads trained artifacts once and exposes a single
`predict(path_or_array)` used by both the FastAPI backend and any local
sanity-testing script (Part 5's "sanity test on your own recorded clips").

Prefers the CNN (best accuracy) when available, falls back to Random Forest,
so the backend / CLI keep working even if only the baseline was trained.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from audio_features import extract_classical_features, extract_mel_spectrogram_image, load_fixed_length
from utils import EMOTIONS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

MIN_RMS_ENERGY = 0.0025   # below this, treat clip as effectively silent
MIN_DURATION_SEC = 0.3    # after trim, clips shorter than this are rejected


class EmotionPredictor:
    def __init__(self, prefer: str = "cnn"):
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.rf = None
        self.svm = None
        self.cnn = None
        self.backend = None
        self._load(prefer)

    def _load(self, prefer: str):
        import joblib

        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
            self.feature_columns = json.load(open(os.path.join(MODEL_DIR, "feature_columns.json")))
            rf_path = os.path.join(MODEL_DIR, "rf_model.joblib")
            svm_path = os.path.join(MODEL_DIR, "svm_model.joblib")
            if os.path.exists(rf_path):
                self.rf = joblib.load(rf_path)
            if os.path.exists(svm_path):
                self.svm = joblib.load(svm_path)

        cnn_path = os.path.join(MODEL_DIR, "cnn_model.keras")
        if os.path.exists(cnn_path):
            import tensorflow as tf
            self.cnn = tf.keras.models.load_model(cnn_path)

        if prefer == "cnn" and self.cnn is not None:
            self.backend = "cnn"
        elif self.rf is not None:
            self.backend = "rf"
        elif self.svm is not None:
            self.backend = "svm"
        elif self.cnn is not None:
            self.backend = "cnn"
        else:
            raise RuntimeError(
                f"No trained model artifacts found in {MODEL_DIR}. "
                "Run train_baseline.py and/or train_deep.py first."
            )

    def is_ready(self) -> bool:
        return self.backend is not None

    def _validate(self, y: np.ndarray, sr: int):
        import librosa

        trimmed, _ = librosa.effects.trim(y, top_db=25)
        duration = len(trimmed) / sr
        rms = float(np.sqrt(np.mean(trimmed ** 2))) if len(trimmed) else 0.0
        if duration < MIN_DURATION_SEC or rms < MIN_RMS_ENERGY:
            return False, f"Clip appears silent or too short (voiced duration={duration:.2f}s, rms={rms:.4f})."
        return True, None

    def predict_from_path(self, path: str, model: str = None) -> dict:
        y = load_fixed_length(path)
        import librosa
        y_full, sr = librosa.load(path, sr=None, mono=True)
        ok, msg = self._validate(y_full, sr)
        if not ok:
            return {"error": msg}
        return self._predict(y, model=model)

    def predict_from_array(self, y_raw: np.ndarray, sr: int, model: str = None) -> dict:
        import librosa

        ok, msg = self._validate(y_raw, sr)
        if not ok:
            return {"error": msg}
        if sr != 22050:
            y_raw = librosa.resample(y_raw, orig_sr=sr, target_sr=22050)
        target_len = int(3.5 * 22050)
        y_raw, _ = librosa.effects.trim(y_raw, top_db=25)
        if len(y_raw) < target_len:
            y_raw = np.pad(y_raw, (0, target_len - len(y_raw)))
        else:
            y_raw = y_raw[:target_len]
        return self._predict(y_raw.astype(np.float32), model=model)

    def _predict(self, y: np.ndarray, model: str = None) -> dict:
        use = model or self.backend

        if use == "cnn":
            if self.cnn is None:
                return {"error": "CNN model not available on this server."}
            spec = extract_mel_spectrogram_image(y)[np.newaxis, ..., np.newaxis]
            probs = self.cnn.predict(spec, verbose=0)[0]
            classes = EMOTIONS
        else:
            if self.scaler is None:
                return {"error": f"{use} model not available on this server."}
            feats = extract_classical_features(y)
            row = np.array([[feats[c] for c in self.feature_columns]])
            row_s = self.scaler.transform(row)
            clf = self.rf if use == "rf" else self.svm
            if clf is None:
                return {"error": f"{use} model not available on this server."}
            probs = clf.predict_proba(row_s)[0]
            classes = self.label_encoder.inverse_transform(np.arange(len(probs)))

        order = np.argsort(probs)[::-1]
        top_label = classes[order[0]]
        confidence = float(probs[order[0]])
        breakdown = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
        return {
            "emotion": str(top_label),
            "confidence": confidence,
            "probabilities": breakdown,
            "model_used": use,
        }


_predictor = None


def get_predictor(prefer: str = "cnn") -> EmotionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = EmotionPredictor(prefer=prefer)
    return _predictor


if __name__ == "__main__":
    # quick CLI sanity test: python predict.py path/to/clip.wav
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file> [rf|svm|cnn]")
        sys.exit(1)
    p = get_predictor()
    result = p.predict_from_path(sys.argv[1], model=sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(result, indent=2))

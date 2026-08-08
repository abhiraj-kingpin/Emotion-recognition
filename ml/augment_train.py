"""
Data augmentation (Part 4) — training split only.
Generates one randomly-augmented (pitch shift / time stretch / background noise)
mel-spectrogram per training clip and caches it separately so train_deep.py can
concatenate original + augmented spectrograms without recomputing anything.
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from audio_features import augment_audio, extract_mel_spectrogram_image, load_fixed_length

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_PATH = os.path.join(PROCESSED_DIR, "spectrograms_train_aug.npz")


def main(force: bool = False, copies: int = 1, seed: int = 42):
    if os.path.exists(OUT_PATH) and not force:
        print(f"[skip] {OUT_PATH} already exists (use --force to recompute)")
        return

    train_csv = os.path.join(PROCESSED_DIR, "train.csv")
    df = pd.read_csv(train_csv)
    rng = np.random.default_rng(seed)

    specs, labels, paths = [], [], []
    for _ in range(copies):
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Augmenting (pass {_+1}/{copies})"):
            y = load_fixed_length(r["file_path"])
            y_aug = augment_audio(y, rng=rng)
            spec = extract_mel_spectrogram_image(y_aug)
            specs.append(spec)
            labels.append(r["emotion"])
            paths.append(r["file_path"])

    X = np.stack(specs)[..., np.newaxis]
    y_arr = np.array(labels)
    np.savez_compressed(OUT_PATH, X=X, y=y_arr, file_path=np.array(paths))
    print(f"Saved augmented training spectrograms: X{X.shape} -> {OUT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copies", type=int, default=1, help="augmented copies per training clip")
    args = parser.parse_args()
    main(force=args.force, copies=args.copies)

"""
Part 2: Feature Extraction
---------------------------
Single pass over every clip in data/processed/metadata.csv that produces two
cached artifacts so nothing needs to be recomputed on subsequent runs:

  1. data/processed/features_classical.pkl (+ .csv)
     - one row per clip: MFCC (mean+std x40), Chroma STFT (mean+std x12),
       Mel spectrogram summary, Zero-Crossing Rate (mean+std), RMS Energy (mean+std)
       -> feeds the classical ML baseline (Part 3)

  2. data/processed/spectrograms_<split>.npz
     - fixed-shape log-mel spectrogram "images" (128 mel bands x fixed frames)
       -> feeds the CNN deep-learning model (Part 4)

Re-run any time; if the cache files already exist they are reused unless --force
is passed.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from audio_features import extract_classical_features, extract_mel_spectrogram_image, load_fixed_length

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CLASSICAL_PKL = os.path.join(PROCESSED_DIR, "features_classical.pkl")
CLASSICAL_CSV = os.path.join(PROCESSED_DIR, "features_classical.csv")


def run(force: bool = False, limit: int = None):
    meta_path = os.path.join(PROCESSED_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Run data_prep.py first to generate metadata.csv")
    meta = pd.read_csv(meta_path)
    if limit:
        meta = pd.concat([d.head(limit) for _, d in meta.groupby("emotion")], ignore_index=True)

    if os.path.exists(CLASSICAL_PKL) and not force:
        print(f"[skip] {CLASSICAL_PKL} already exists (use --force to recompute)")
        classical_done = True
    else:
        classical_done = False

    spec_files_needed = [
        os.path.join(PROCESSED_DIR, f"spectrograms_{split}.npz")
        for split in ("train", "val", "test")
    ]
    spec_done = all(os.path.exists(f) for f in spec_files_needed) and not force
    if spec_done:
        print("[skip] spectrogram caches already exist (use --force to recompute)")

    if classical_done and spec_done:
        print("Nothing to do. Delete the cache files or pass --force to recompute.")
        return

    rows = []
    spec_by_split = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
    # each tuple: (specs list, labels list, file_paths list)

    t0 = time.time()
    failed = []
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc="Extracting features"):
        try:
            y = load_fixed_length(r["file_path"])
        except Exception as e:
            failed.append((r["file_path"], str(e)))
            continue

        if not classical_done:
            feats = extract_classical_features(y)
            feats["file_path"] = r["file_path"]
            feats["emotion"] = r["emotion"]
            feats["actor"] = r["actor"]
            feats["sex"] = r["sex"]
            feats["split"] = r["split"]
            rows.append(feats)

        if not spec_done:
            spec = extract_mel_spectrogram_image(y)
            specs, labels, paths = spec_by_split[r["split"]]
            specs.append(spec)
            labels.append(r["emotion"])
            paths.append(r["file_path"])

    if failed:
        print(f"\n{len(failed)} files failed to load, skipped:")
        for p, e in failed[:10]:
            print(f"  {p}: {e}")

    if not classical_done:
        df = pd.DataFrame(rows)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_pickle(CLASSICAL_PKL)
        df.to_csv(CLASSICAL_CSV, index=False)
        print(f"\nSaved classical features: {df.shape} -> {CLASSICAL_PKL}")

    if not spec_done:
        for split, (specs, labels, paths) in spec_by_split.items():
            if not specs:
                continue
            X = np.stack(specs)[..., np.newaxis]  # (N, n_mels, frames, 1)
            y_arr = np.array(labels)
            out_path = os.path.join(PROCESSED_DIR, f"spectrograms_{split}.npz")
            np.savez_compressed(out_path, X=X, y=y_arr, file_path=np.array(paths))
            print(f"Saved {split} spectrograms: X{X.shape} -> {out_path}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap clips per emotion")
    args = parser.parse_args()
    run(force=args.force, limit=args.limit)

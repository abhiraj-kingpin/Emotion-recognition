"""
Part 1: Dataset & Data Handling
---------------------------------
Scans the extracted RAVDESS audio-speech dataset, parses emotion labels
straight out of the RAVDESS filename convention, and produces a stratified
70/15/15 train/val/test split so every emotion is proportionally represented
in every split.

RAVDESS filename convention (7 dash-separated numeric fields):
    modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
    e.g. 03-01-06-01-02-01-12.wav

    modality:        03 = audio-only (the only value present in this dataset)
    vocal channel:   01 = speech
    emotion:         01 neutral, 02 calm, 03 happy, 04 sad,
                      05 angry, 06 fearful, 07 disgust, 08 surprised
    intensity:       01 normal, 02 strong (no "strong" variant for neutral)
    statement:       01 "Kids are talking by the door", 02 "Dogs are sitting by the door"
    repetition:      01 / 02
    actor:           01-24 (odd = male, even = female)
"""
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ravdess")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
INTENSITY_MAP = {"01": "normal", "02": "strong"}
STATEMENT_MAP = {"01": "kids_door", "02": "dogs_door"}


def parse_filename(path: str) -> dict:
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("-")
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
    actor_num = int(actor)
    return {
        "file_path": os.path.abspath(path),
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion_code": emotion,
        "emotion": EMOTION_MAP[emotion],
        "intensity": INTENSITY_MAP[intensity],
        "statement": STATEMENT_MAP[statement],
        "repetition": int(repetition),
        "actor": actor_num,
        "sex": "male" if actor_num % 2 == 1 else "female",
    }


def build_metadata() -> pd.DataFrame:
    wav_files = glob.glob(os.path.join(RAW_DIR, "**", "Actor_*", "*.wav"), recursive=True)
    if not wav_files:
        # some zip layouts extract actors directly under RAW_DIR
        wav_files = glob.glob(os.path.join(RAW_DIR, "Actor_*", "*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No RAVDESS .wav files found under {RAW_DIR}. "
            "Make sure Audio_Speech_Actors_01-24.zip has been downloaded and extracted."
        )
    rows = [parse_filename(p) for p in wav_files]
    df = pd.DataFrame(rows)
    return df


def stratified_split(df: pd.DataFrame, seed: int = 42):
    # 70% train, 15% val, 15% test, stratified on emotion label
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["emotion"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["emotion"], random_state=seed
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return train_df, val_df, test_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_metadata()
    print(f"Found {len(df)} audio clips across {df['actor'].nunique()} actors.")
    print("\nClass balance (all data):")
    counts = df["emotion"].value_counts()
    print(counts.to_string())
    imbalance_ratio = counts.max() / counts.min()
    print(f"\nmax/min class ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 1.5:
        print("Note: noticeable imbalance -> handled downstream via class_weight='balanced' "
              "(classical ML) and class-weighted loss (deep model), rather than naive oversampling, "
              "to avoid duplicating short utterances.")
    else:
        print("Classes are close to balanced (RAVDESS: neutral/calm have half the reps of the rest "
              "at strong intensity, otherwise ~uniform) - accepting as-is.")

    train_df, val_df, test_df = stratified_split(df)

    full = pd.concat([train_df, val_df, test_df]).sort_index()
    full.to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print(f"\nSplit sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print("\nPer-split class balance check:")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pct = (d["emotion"].value_counts(normalize=True) * 100).round(1)
        print(f"  {name}: {pct.to_dict()}")

    print(f"\nSaved metadata + splits to {OUT_DIR}")


if __name__ == "__main__":
    main()

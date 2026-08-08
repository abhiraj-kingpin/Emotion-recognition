"""
Part 5 (sanity-test) + Part 6 (input validation) smoke test.
------------------------------------------------------------------
We don't have a microphone in this environment to record "your own clips",
so this generates the edge cases a real deployment has to survive
(silence, a near-silent whisper-level clip, a very short blip) plus one
genuine held-out RAVDESS clip, and confirms the predictor either returns a
sane label or a graceful validation error - never a crash.
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from predict import get_predictor

SR = 22050
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sanity_clips")


def make_clips():
    os.makedirs(OUT_DIR, exist_ok=True)
    clips = {}

    silence = np.zeros(int(SR * 3.0), dtype=np.float32)
    clips["silence_3s.wav"] = silence

    tiny_blip = (0.2 * np.sin(2 * np.pi * 440 * np.arange(int(SR * 0.15)) / SR)).astype(np.float32)
    clips["tiny_blip_0.15s.wav"] = tiny_blip

    rng = np.random.default_rng(0)
    whisper = (0.003 * rng.standard_normal(int(SR * 2.0))).astype(np.float32)
    clips["near_silent_whisper.wav"] = whisper

    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(SR * 2.5)) / SR)).astype(np.float32)
    clips["pure_tone_2.5s.wav"] = tone

    paths = {}
    for name, y in clips.items():
        path = os.path.join(OUT_DIR, name)
        sf.write(path, y, SR)
        paths[name] = path
    return paths


def main():
    predictor = get_predictor()
    print(f"Loaded predictor, backend={predictor.backend}\n")

    edge_cases = make_clips()
    for name, path in edge_cases.items():
        result = predictor.predict_from_path(path)
        status = "REJECTED (as expected)" if "error" in result else "PREDICTED"
        print(f"[{status}] {name}")
        print(f"  -> {result}\n")

    import pandas as pd
    test_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test.csv")
    if os.path.exists(test_csv):
        df = pd.read_csv(test_csv)
        sample = df.sample(min(5, len(df)), random_state=1)
        print("Held-out RAVDESS clips (ground truth vs. predicted):")
        correct = 0
        for _, row in sample.iterrows():
            result = predictor.predict_from_path(row["file_path"])
            pred = result.get("emotion", "ERROR")
            ok = "correct" if pred == row["emotion"] else "MISS"
            correct += pred == row["emotion"]
            print(f"  true={row['emotion']:<10} pred={pred:<10} conf={result.get('confidence', 0):.2f}  [{ok}]")
        print(f"\n{correct}/{len(sample)} correct on this random sample")


if __name__ == "__main__":
    main()

"""
Part 5: Evaluation
---------------------
Pulls together Random Forest, SVM and CNN test-set performance:
  - confusion matrix per class for the CNN (RF/SVM ones are already written by
    train_baseline.py)
  - precision/recall/F1 per class for all three models
  - a single baseline-vs-deep-learning comparison table (docs/results/comparison_table.md)
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from utils import EMOTIONS

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "results")


def evaluate_cnn():
    import tensorflow as tf

    model_path = os.path.join(MODEL_DIR, "cnn_model.keras")
    if not os.path.exists(model_path):
        print("[skip] no CNN model found, run train_deep.py first")
        return None

    model = tf.keras.models.load_model(model_path)
    d = np.load(os.path.join(PROCESSED_DIR, "spectrograms_test.npz"), allow_pickle=True)
    X_test, y_test = d["X"], d["y"]

    probs = model.predict(X_test, verbose=0)
    pred_idx = probs.argmax(axis=1)
    y_pred = np.array([EMOTIONS[i] for i in pred_idx])

    report_txt = classification_report(y_test, y_pred, labels=EMOTIONS)
    with open(os.path.join(REPORT_DIR, "cnn_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)
    print("\n=== CNN classification report ===")
    print(report_txt)

    save_confusion_matrix(y_test, y_pred, EMOTIONS,
                           os.path.join(REPORT_DIR, "confusion_matrix_cnn.png"),
                           "CNN (mel-spectrogram) — Confusion Matrix (test)")

    report_dict = classification_report(y_test, y_pred, labels=EMOTIONS, output_dict=True)
    acc = report_dict["accuracy"]
    f1_macro = report_dict["macro avg"]["f1-score"]
    return {"test_accuracy": acc, "test_f1_macro": f1_macro}, report_dict


def save_confusion_matrix(y_true, y_pred, labels, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def per_class_table(model_reports: dict) -> pd.DataFrame:
    """model_reports: {model_name: sklearn classification_report(output_dict=True)}"""
    rows = []
    for model_name, report in model_reports.items():
        for emo in EMOTIONS:
            if emo not in report:
                continue
            rows.append({
                "model": model_name,
                "emotion": emo,
                "precision": round(report[emo]["precision"], 3),
                "recall": round(report[emo]["recall"], 3),
                "f1": round(report[emo]["f1-score"], 3),
                "support": int(report[emo]["support"]),
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    baseline_path = os.path.join(REPORT_DIR, "baseline_results.json")
    baseline = json.load(open(baseline_path)) if os.path.exists(baseline_path) else {}

    rf_report = svm_report = cnn_report = None
    if os.path.exists(os.path.join(REPORT_DIR, "rf_classification_report.txt")):
        # regenerate as dict for the per-class table
        from sklearn.preprocessing import LabelEncoder
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        feature_cols = json.load(open(os.path.join(MODEL_DIR, "feature_columns.json")))
        df = pd.read_pickle(os.path.join(PROCESSED_DIR, "features_classical.pkl"))
        test_df = df[df["split"] == "test"]
        X_test = scaler.transform(test_df[feature_cols].values)
        y_test = test_df["emotion"].values

        le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

        rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
        y_pred_rf = le.inverse_transform(rf.predict(X_test))
        rf_report = classification_report(y_test, y_pred_rf, labels=EMOTIONS, output_dict=True)

        svm = joblib.load(os.path.join(MODEL_DIR, "svm_model.joblib"))
        y_pred_svm = le.inverse_transform(svm.predict(X_test))
        svm_report = classification_report(y_test, y_pred_svm, labels=EMOTIONS, output_dict=True)

    cnn_summary = None
    cnn_result = evaluate_cnn()
    if cnn_result:
        cnn_summary, cnn_report = cnn_result

    # ---- comparison table ----
    lines = ["| Model | Test Accuracy | Test Macro-F1 |", "|---|---|---|"]
    if "random_forest" in baseline:
        r = baseline["random_forest"]
        lines.append(f"| Random Forest (baseline) | {r['test_accuracy']:.3f} | {r['test_f1_macro']:.3f} |")
    if "svm" in baseline:
        r = baseline["svm"]
        lines.append(f"| SVM (baseline) | {r['test_accuracy']:.3f} | {r['test_f1_macro']:.3f} |")
    if cnn_summary:
        lines.append(f"| CNN (deep learning) | {cnn_summary['test_accuracy']:.3f} | {cnn_summary['test_f1_macro']:.3f} |")

    table_md = "\n".join(lines)
    with open(os.path.join(REPORT_DIR, "comparison_table.md"), "w", encoding="utf-8") as f:
        f.write("# Baseline vs. Deep Learning - Test Set Results\n\n" + table_md + "\n")
    print("\n" + table_md)

    # ---- per-class table across all available models ----
    reports = {}
    if rf_report: reports["random_forest"] = rf_report
    if svm_report: reports["svm"] = svm_report
    if cnn_report: reports["cnn"] = cnn_report
    if reports:
        pc = per_class_table(reports)
        pc.to_csv(os.path.join(REPORT_DIR, "per_class_metrics.csv"), index=False)
        print(f"\nSaved per-class metrics to {os.path.join(REPORT_DIR, 'per_class_metrics.csv')}")

    print(f"\nAll evaluation artifacts saved under {REPORT_DIR}")


if __name__ == "__main__":
    main()

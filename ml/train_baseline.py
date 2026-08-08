"""
Part 3: Baseline Model (Classical ML)
---------------------------------------
Random Forest and SVM trained on the cached classical feature vectors
(features_classical.pkl), tuned with real GridSearchCV, evaluated on the
held-out val split for model selection and the test split for the final
benchmark numbers used in the baseline-vs-deep-learning comparison table.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from utils import EMOTIONS

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "results")

NON_FEATURE_COLS = {"file_path", "emotion", "actor", "sex", "split"}


def load_splits():
    df = pd.read_pickle(os.path.join(PROCESSED_DIR, "features_classical.pkl"))
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    splits = {}
    for split in ("train", "val", "test"):
        d = df[df["split"] == split]
        splits[split] = (d[feature_cols].values, d["emotion"].values)
    return splits, feature_cols


def evaluate(model, X, y_true_labels, le, name, split_name):
    y_pred = le.inverse_transform(model.predict(X))
    acc = accuracy_score(y_true_labels, y_pred)
    f1_macro = f1_score(y_true_labels, y_pred, average="macro")
    print(f"[{name}] {split_name} accuracy={acc:.4f}  macro-F1={f1_macro:.4f}")
    return acc, f1_macro, y_pred


def save_confusion_matrix(y_true, y_pred, labels, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    splits, feature_cols = load_splits()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits["train"], splits["val"], splits["test"]
    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape}  features={len(feature_cols)}")

    le = LabelEncoder()
    le.fit(EMOTIONS)
    y_train_enc = le.transform(y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}

    # ---------------- Random Forest ----------------
    print("\n=== GridSearchCV: Random Forest ===")
    rf_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 15, 30],
        "min_samples_split": [2, 4],
    }
    t0 = time.time()
    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
        rf_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    rf_search.fit(X_train_s, y_train_enc)
    print(f"RF grid search done in {time.time()-t0:.1f}s. Best params: {rf_search.best_params_}")
    rf_best = rf_search.best_estimator_
    acc_v, f1_v, _ = evaluate(rf_best, X_val_s, y_val, le, "RandomForest", "val")
    acc_t, f1_t, y_pred_rf = evaluate(rf_best, X_test_s, y_test, le, "RandomForest", "test")
    results["random_forest"] = {
        "best_params": rf_search.best_params_,
        "val_accuracy": acc_v, "val_f1_macro": f1_v,
        "test_accuracy": acc_t, "test_f1_macro": f1_t,
    }
    joblib.dump(rf_best, os.path.join(MODEL_DIR, "rf_model.joblib"))
    with open(os.path.join(REPORT_DIR, "rf_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred_rf, labels=EMOTIONS))
    save_confusion_matrix(y_test, y_pred_rf, EMOTIONS,
                           os.path.join(REPORT_DIR, "confusion_matrix_rf.png"),
                           "Random Forest — Confusion Matrix (test)")

    # ---------------- SVM ----------------
    print("\n=== GridSearchCV: SVM ===")
    svm_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    }
    t0 = time.time()
    svm_search = GridSearchCV(
        SVC(probability=True, class_weight="balanced", random_state=42),
        svm_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    svm_search.fit(X_train_s, y_train_enc)
    print(f"SVM grid search done in {time.time()-t0:.1f}s. Best params: {svm_search.best_params_}")
    svm_best = svm_search.best_estimator_
    acc_v, f1_v, _ = evaluate(svm_best, X_val_s, y_val, le, "SVM", "val")
    acc_t, f1_t, y_pred_svm = evaluate(svm_best, X_test_s, y_test, le, "SVM", "test")
    results["svm"] = {
        "best_params": svm_search.best_params_,
        "val_accuracy": acc_v, "val_f1_macro": f1_v,
        "test_accuracy": acc_t, "test_f1_macro": f1_t,
    }
    joblib.dump(svm_best, os.path.join(MODEL_DIR, "svm_model.joblib"))
    with open(os.path.join(REPORT_DIR, "svm_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred_svm, labels=EMOTIONS))
    save_confusion_matrix(y_test, y_pred_svm, EMOTIONS,
                           os.path.join(REPORT_DIR, "confusion_matrix_svm.png"),
                           "SVM — Confusion Matrix (test)")

    # persist shared preprocessing artifacts + feature column order for inference
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    with open(os.path.join(REPORT_DIR, "baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Baseline summary ===")
    print(json.dumps(results, indent=2))
    print(f"\nModels saved to {MODEL_DIR}, reports saved to {REPORT_DIR}")


if __name__ == "__main__":
    main()

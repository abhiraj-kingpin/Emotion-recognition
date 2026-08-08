"""
Part 4: Deep Learning Model — Option A: CNN on mel-spectrograms.
-------------------------------------------------------------------
Treats each clip's fixed-shape log-mel spectrogram as a 2D "image":
3x (Conv2D + BatchNorm + MaxPooling) -> Flatten -> Dense -> Dropout -> softmax.

Trains on original + augmented (pitch shift / time stretch / noise) spectrograms
from the train split only; validates/early-stops on the untouched val split;
final numbers reported on the untouched test split by evaluate.py.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import EMOTIONS

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "results")


def load_spec_split(split):
    path = os.path.join(PROCESSED_DIR, f"spectrograms_{split}.npz")
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"]


def build_model(input_shape, n_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # GlobalAveragePooling2D instead of Flatten: a Flatten here would hand
        # ~36k values to the Dense layer (9M+ params on ~2k training clips -
        # instant overfit, exploding val loss). GAP collapses each of the 128
        # feature maps to one number first, which is both far more
        # parameter-efficient and the standard choice for small-data CNNs.
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(epochs: int = 60, batch_size: int = 32, use_augmented: bool = True):
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow import keras

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    X_train, y_train = load_spec_split("train")
    X_val, y_val = load_spec_split("val")
    X_test, y_test = load_spec_split("test")

    aug_path = os.path.join(PROCESSED_DIR, "spectrograms_train_aug.npz")
    if use_augmented and os.path.exists(aug_path):
        d = np.load(aug_path, allow_pickle=True)
        X_train = np.concatenate([X_train, d["X"]], axis=0)
        y_train = np.concatenate([y_train, d["y"]], axis=0)
        print(f"Training with augmented data: {X_train.shape[0]} clips total")
    else:
        print(f"Training without augmentation: {X_train.shape[0]} clips (run augment_train.py to enable)")

    label_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
    y_train_idx = np.array([label_to_idx[l] for l in y_train])
    y_val_idx = np.array([label_to_idx[l] for l in y_val])
    y_test_idx = np.array([label_to_idx[l] for l in y_test])

    class_weights = compute_class_weight("balanced", classes=np.arange(len(EMOTIONS)), y=y_train_idx)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("Class weights:", {EMOTIONS[i]: round(w, 2) for i, w in class_weight_dict.items()})

    model = build_model(X_train.shape[1:], len(EMOTIONS))
    model.summary()

    ckpt_path = os.path.join(MODEL_DIR, "cnn_model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train_idx,
        validation_data=(X_val, y_val_idx),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )
    print(f"Training finished in {time.time()-t0:.1f}s, {len(history.history['loss'])} epochs run")

    model.save(ckpt_path)  # ensure final/best weights are on disk
    with open(os.path.join(MODEL_DIR, "cnn_labels.json"), "w", encoding="utf-8") as f:
        json.dump(EMOTIONS, f)

    test_loss, test_acc = model.evaluate(X_test, y_test_idx, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

    # training curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "cnn_training_curves.png"), dpi=150)
    plt.close()

    with open(os.path.join(REPORT_DIR, "cnn_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "epochs_run": len(history.history["loss"]),
            "train_samples": int(X_train.shape[0]),
        }, f, indent=2)

    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, use_augmented=not args.no_augment)

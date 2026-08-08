"""
Part 10: Deployment — model size reduction.
-----------------------------------------------
Converts the trained Keras CNN to a dynamic-range-quantized TFLite model.
Cuts the on-disk size roughly 3-4x (float32 -> int8 weights) with a small
accuracy trade-off, which matters when the backend is deployed on a
free-tier host with limited RAM/disk (Render/Railway free tiers are
typically 512MB).

Also re-evaluates the quantized model on the test set so you can see exactly
what the size/accuracy trade-off cost, instead of taking it on faith.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import EMOTIONS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "results")


def main():
    import tensorflow as tf

    keras_path = os.path.join(MODEL_DIR, "cnn_model.keras")
    if not os.path.exists(keras_path):
        print("No CNN model found — run train_deep.py first.")
        return

    model = tf.keras.models.load_model(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantization
    tflite_model = converter.convert()

    tflite_path = os.path.join(MODEL_DIR, "cnn_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    orig_size = os.path.getsize(keras_path)
    quant_size = os.path.getsize(tflite_path)
    print(f"Keras model:  {orig_size/1e6:.2f} MB")
    print(f"TFLite model: {quant_size/1e6:.2f} MB  ({orig_size/quant_size:.1f}x smaller)")

    # sanity-check accuracy on the test set didn't collapse
    test_npz = os.path.join(PROCESSED_DIR, "spectrograms_test.npz")
    if os.path.exists(test_npz):
        d = np.load(test_npz, allow_pickle=True)
        X_test, y_test = d["X"], d["y"]

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]

        correct = 0
        for i in range(len(X_test)):
            interpreter.set_tensor(in_details["index"], X_test[i:i+1].astype(np.float32))
            interpreter.invoke()
            pred = np.argmax(interpreter.get_tensor(out_details["index"])[0])
            if EMOTIONS[pred] == y_test[i]:
                correct += 1
        quant_acc = correct / len(X_test)
        print(f"TFLite test accuracy: {quant_acc:.4f}")

        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(os.path.join(REPORT_DIR, "quantization_results.json"), "w") as f:
            json.dump({
                "keras_size_mb": round(orig_size / 1e6, 2),
                "tflite_size_mb": round(quant_size / 1e6, 2),
                "size_reduction_factor": round(orig_size / quant_size, 2),
                "tflite_test_accuracy": round(quant_acc, 4),
            }, f, indent=2)


if __name__ == "__main__":
    main()

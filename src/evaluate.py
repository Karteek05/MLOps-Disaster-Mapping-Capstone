import os
import sys
import json
import yaml
import tensorflow as tf

sys.path.append(os.getcwd())

from src.metrics import SparseMeanIoU
from src.data_loader import list_train_val_files, build_dataset

MODEL_PATH = os.path.join("models", "unet_model.h5")
DATA_DIR = "data/processed"
EVAL_METRICS_FILE = "eval_metrics.json"

NUM_CHANNELS = 6
NUM_CLASSES = 5


def evaluate_model():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    img_size = params.get("img_size", 1024)
    batch_size = params["batch_size"]

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"SparseMeanIoU": SparseMeanIoU},
        compile=False,
    )
    model.compile(
        loss=params["loss_function"],
        metrics=["accuracy", SparseMeanIoU(NUM_CLASSES)],
    )

    # Same file listing/shuffle/split as train.py, so this scores the
    # model on exactly the held-out slice it never trained on.
    (_, _), (val_images, val_masks) = list_train_val_files(DATA_DIR)
    if not val_images:
        print("FATAL: No validation files found - is data/processed populated?")
        sys.exit(1)

    val_dataset = build_dataset(
        val_images, val_masks, img_size, NUM_CHANNELS, NUM_CLASSES, batch_size
    )

    print(f"Evaluating on {len(val_images)} held-out samples...")
    results = model.evaluate(val_dataset, verbose=1, return_dict=True)

    metrics = {
        "eval_loss": float(results["loss"]),
        "eval_iou": float(results["IoU"]),
        "eval_accuracy": float(results["accuracy"]),
    }

    with open(EVAL_METRICS_FILE, "w") as f:
        json.dump(metrics, f)

    print(f"Metrics written to {EVAL_METRICS_FILE}: {metrics}")
    return metrics


if __name__ == "__main__":
    evaluate_model()

import os
# FIX FOR AMD/MKL CRASH: Set this environment variable AT THE TOP.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import json
import yaml
import numpy as np
import tensorflow as tf
import mlflow

sys.path.append(os.getcwd())

from src.model import get_resnet_unet_model
from src.metrics import SparseMeanIoU
from src.data_loader import load_real_data

# ----------------- CONFIG -----------------
MODELS_DIR = "models"
METRICS_FILE = "metrics.json"
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "unet_model.h5")
DATA_DIR = "data/processed"
SMOKE_TEST = os.environ.get("SMOKE_TEST", "").lower() in ("1", "true", "yes")

# Load parameters from params.yaml
try:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]
except FileNotFoundError:
    print("FATAL: params.yaml not found.")
    sys.exit(1)

NUM_CHANNELS = 6
NUM_CLASSES = 5
BATCH_SIZE = params["batch_size"]
EPOCHS = params["epochs"]
LEARNING_RATE = params["learning_rate"]
PRETRAINED_TRAINABLE = params.get("pretrained_trainable", False)

if SMOKE_TEST:
    # Tiny, offline-friendly settings: no real data, no pretrained-weight
    # download, small enough to run on CPU in CI in a few seconds.
    IMG_SIZE = 64
    EPOCHS = 1
    PRETRAINED = False
else:
    IMG_SIZE = params.get("img_size", 1024)
    PRETRAINED = True
    # Mixed precision cuts activation memory ~in half - important for
    # fitting this model on a 4GB GPU. No speedup on non-Tensor-Core GPUs
    # (e.g. GTX 1650), but the memory savings still matter.
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def load_smoke_test_data():
    print(f"[SMOKE_TEST] Generating tiny synthetic batch ({IMG_SIZE}x{IMG_SIZE}).")
    X = np.random.rand(BATCH_SIZE, IMG_SIZE, IMG_SIZE, NUM_CHANNELS).astype(np.float32)
    Y = np.random.randint(0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE, IMG_SIZE)).astype(np.int32)
    return X, Y


# Per-class pixel weights for the loss, derived from the full training mask
# set (data/processed/*_mask.png: background/no_damage/minor/major/destroyed
# = 94.18% / 4.30% / 0.53% / 0.68% / 0.32%). Without weighting, the model
# minimizes loss by predicting background almost everywhere and never
# learns the rare damage classes. Weights are sqrt-dampened inverse
# frequency (softer than raw inverse frequency, which produced ~300x swings
# and risked destabilizing training), normalized to mean 1 so overall loss
# magnitude - and the learning rate that was tuned for it - stays comparable
# to the unweighted loss.
CLASS_WEIGHTS = tf.constant([0.104, 0.488, 1.383, 1.231, 1.794], dtype=tf.float32)


def weighted_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    per_pixel_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    pixel_weights = tf.gather(CLASS_WEIGHTS, y_true)
    return tf.reduce_mean(per_pixel_loss * pixel_weights)


def train_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    with mlflow.start_run(run_name=f"Run_LR{LEARNING_RATE}_size{IMG_SIZE}"):
        print("-" * 60)
        print("MLflow Run Started. Logging parameters...")
        mlflow.log_params(params)

        model = get_resnet_unet_model(
            img_size=IMG_SIZE,
            num_channels=NUM_CHANNELS,
            num_classes=NUM_CLASSES,
            pretrained=PRETRAINED,
            pretrained_trainable=PRETRAINED_TRAINABLE,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=weighted_sparse_categorical_crossentropy,
            metrics=["accuracy", SparseMeanIoU(NUM_CLASSES)],
        )

        if SMOKE_TEST:
            X_train, Y_train = load_smoke_test_data()
            history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
            final_loss = float(history.history["loss"][-1])
            final_iou = float(history.history["IoU"][-1])
            # DVC tracks this artifact; MLflow only logs params/metrics (not
            # the full model binary, to avoid duplicating hundreds of MB into
            # mlruns/ on every run).
            model.save(MODEL_OUTPUT_PATH)
        else:
            train_dataset, val_dataset = load_real_data(
                DATA_DIR, IMG_SIZE, NUM_CHANNELS, NUM_CLASSES, BATCH_SIZE
            )
            # Keeps only the best-val_IoU epoch's weights on disk. Without
            # save_best_only, a fixed epoch count can end on an overfit
            # epoch (val_IoU peaked mid-run, then degraded by the last
            # epoch) and silently ship the worse checkpoint.
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                MODEL_OUTPUT_PATH, monitor="val_IoU", mode="max",
                save_best_only=True, verbose=1,
            )
            history = model.fit(
                train_dataset,
                epochs=EPOCHS,
                validation_data=val_dataset,
                verbose=1,
                callbacks=[checkpoint_callback],
            )
            best_epoch = int(np.argmax(history.history["val_IoU"]))
            final_loss = float(history.history["val_loss"][best_epoch])
            final_iou = float(history.history["val_IoU"][best_epoch])
            # No explicit model.save() here - the checkpoint callback above
            # already wrote the best epoch's weights to MODEL_OUTPUT_PATH.
            # Saving again here would overwrite it with the *last* epoch's
            # weights, which may be worse.

        mlflow.log_metric("final_loss", final_loss)
        mlflow.log_metric("iou_validation", final_iou)

        with open(METRICS_FILE, "w") as f:
            json.dump({"iou": final_iou, "loss": final_loss}, f)

        print("\n--- Training Run Completed ---")
        print(f"Model saved to {MODEL_OUTPUT_PATH}")
        print(f"Metrics saved to {METRICS_FILE}")
        print("-" * 60)


if __name__ == "__main__":
    train_model()

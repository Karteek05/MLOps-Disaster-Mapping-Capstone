import os
import sys
import json
import yaml
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from PIL import Image
import mlflow
import mlflow.tensorflow

# Add project root to Python path
sys.path.append(os.getcwd())
from src.model import get_resnet_unet_model

# ----------------- CONFIG -----------------
MODELS_DIR = os.path.join(os.getcwd(), "artifacts", "models")
METRICS_FILE = os.path.join(os.getcwd(), "metrics.json")
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "unet_model.h5")
DATA_DIR = os.path.join(os.getcwd(), "data", "processed")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load parameters
try:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]
except FileNotFoundError:
    print("FATAL: params.yaml not found.")
    sys.exit(1)

IMG_SIZE = 1024
NUM_CHANNELS = 6
NUM_CLASSES = 5
BATCH_SIZE = params["batch_size"]
EPOCHS = params["epochs"]
LEARNING_RATE = params["learning_rate"]
LOSS_FUNCTION = params["loss_function"]

# ----------------- CUSTOM METRIC -----------------
class SparseMeanIoU(Metric):
    def __init__(self, num_classes, name="IoU", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_mat = self.add_weight(
            name="conf_mat", shape=(num_classes, num_classes),
            initializer="zeros", dtype=tf.int64
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1])
        y_pred_labels = tf.reshape(y_pred_labels, [-1])
        cm = tf.math.confusion_matrix(
            y_true, y_pred_labels, num_classes=self.num_classes, dtype=tf.int64
        )
        self.conf_mat.assign_add(cm)

    def result(self):
        cm = tf.cast(self.conf_mat, tf.float32)
        inter = tf.linalg.tensor_diag_part(cm)
        union = tf.reduce_sum(cm, 0) + tf.reduce_sum(cm, 1) - inter
        iou = inter / (union + 1e-7)
        return tf.reduce_mean(iou)

    def reset_states(self):
        self.conf_mat.assign(tf.zeros_like(self.conf_mat))


# ----------------- DATA LOADING -----------------
def _collapse_mask_channels(mask):
    c = tf.shape(mask)[-1]

    def squeeze_chan():
        return tf.cast(tf.squeeze(mask, axis=-1), tf.int32)

    def argmax_chan():
        return tf.cast(tf.argmax(mask, axis=-1, output_type=tf.int32), tf.int32)

    return tf.case(
        [
            (tf.equal(c, 1), squeeze_chan),
            (tf.equal(c, NUM_CLASSES), argmax_chan)
        ],
        default=argmax_chan,
        exclusive=True
    )

def load_and_parse(image_path_tensor, mask_path_tensor):
    def _load_npy(path):
        return np.load(path.decode()).astype(np.float32)

    image = tf.numpy_function(_load_npy, [image_path_tensor], tf.float32)
    image.set_shape([IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    mask_bytes = tf.io.read_file(mask_path_tensor)
    mask = tf.io.decode_image(mask_bytes, channels=0, dtype=tf.uint8)

    if mask.shape.rank is None:
        mask.set_shape([IMG_SIZE, IMG_SIZE, None])

    mask = tf.cond(
        tf.equal(tf.rank(mask), 3),
        lambda: _collapse_mask_channels(mask),
        lambda: tf.cast(mask, tf.int32),
    )

    mask.set_shape([IMG_SIZE, IMG_SIZE])
    return image, mask


def load_real_data():
    print(f"Loading data from {DATA_DIR}...")
    all_mask_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_mask.png")))

    if not all_mask_paths:
        print("FATAL: No mask files found.")
        sys.exit(1)

    image_paths_final, mask_paths_final = [], []
    for mask_path in all_mask_paths:
        image_path = mask_path.replace("_mask.png", "_stacked.npy")
        if os.path.exists(image_path):
            image_paths_final.append(image_path)
            mask_paths_final.append(mask_path)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths_final, mask_paths_final))
    DATASET_SIZE = len(image_paths_final)
    TRAIN_SIZE = int(DATASET_SIZE * 0.8)

    dataset = dataset.shuffle(DATASET_SIZE, seed=42)
    train_dataset = dataset.take(TRAIN_SIZE)
    val_dataset = dataset.skip(TRAIN_SIZE)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(load_and_parse, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(load_and_parse, num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    print(f"✅ Data ready: {TRAIN_SIZE} train, {DATASET_SIZE - TRAIN_SIZE} val")
    return train_dataset, val_dataset


# ----------------- TRAIN FUNCTION -----------------
def train_model():
    print("🚀 Starting training...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    with mlflow.start_run(run_name=f"Run_LR{LEARNING_RATE}") as run:
        mlflow.log_params(params)

        train_dataset, val_dataset = load_real_data()

        model = get_resnet_unet_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=LOSS_FUNCTION,
            metrics=["accuracy", SparseMeanIoU(NUM_CLASSES)],
        )

        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            verbose=1,
        )

        final_loss = float(history.history["loss"][-1])
        final_val_iou = float(history.history["val_IoU"][-1])

        mlflow.log_metric("final_loss", final_loss)
        mlflow.log_metric("IoU (validation)", final_val_iou)

        # Save model
        abs_model_path = os.path.abspath(MODEL_OUTPUT_PATH)
        model.save(abs_model_path)
        print(f"✅ Model saved to: {abs_model_path}")

        mlflow.log_artifact(abs_model_path, "model_artifact")

        with open(METRICS_FILE, "w") as f:
            json.dump({"iou": final_val_iou, "loss": final_loss}, f)

        print(f"✅ Metrics written to {METRICS_FILE}")


if __name__ == "__main__":
    train_model()

import os
# FIX FOR AMD/MKL CRASH: Set this environment variable AT THE TOP.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

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
MODELS_DIR = "models"
METRICS_FILE = "metrics.json"
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "unet_model.h5")
DATA_DIR = "data/processed"

# Load parameters from params.yaml
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
EPOCHS = params["epochs"] # This will be '1' from your params.yaml
LEARNING_RATE = params["learning_rate"]
LOSS_FUNCTION = params["loss_function"]

# ----------------- CUSTOM METRIC -----------------
# We need this custom metric because the model output (B, H, W, 5)
# and label (B, H, W) shapes are different.
class SparseMeanIoU(Metric):
    """IoU metric compatible with sparse labels and softmax outputs."""
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

# ----------------- DUMMY DATA LOADER (THE FIX) -----------------
def load_small_sample_data():
    """
    Creates small, DUMMY NumPy arrays for fast testing the MLOps pipeline.
    """
    print(f"Loading DUMMY data for {EPOCHS} epochs and batch size {BATCH_SIZE}")
    
    # X_train: (batch_size, height, width, channels) - The stacked images
    X_train = np.random.rand(
        BATCH_SIZE, IMG_SIZE, IMG_SIZE, NUM_CHANNELS
    ).astype(np.float32)

    # Y_train: (batch_size, height, width) - The integer mask labels
    # This shape (B, H, W) is correct for sparse_categorical_crossentropy
    Y_train = np.random.randint(
        0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE, IMG_SIZE) 
    ).astype(np.int32)
    
    return X_train, Y_train

# ----------------- TRAIN FUNCTION -----------------
def train_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    with mlflow.start_run(run_name=f"Emergency_Prototype_Run_LR{LEARNING_RATE}") as run:
        print("-" * 60)
        print("MLflow Run Started. Logging parameters...")
        mlflow.log_params(params)

        # Load DUMMY data
        X_train, Y_train = load_small_sample_data()

        # Build and compile model
        print("Building model...")
        model = get_resnet_unet_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=LOSS_FUNCTION,
            metrics=["accuracy", SparseMeanIoU(NUM_CLASSES)],
        )

        # Train on DUMMY data
        print(f"Fitting model on DUMMY data for {EPOCHS} epochs...")
        history = model.fit(
            X_train,
            Y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        # Log metrics
        final_loss = float(history.history["loss"][-1])
        final_iou = float(history.history["IoU"][-1]) 

        mlflow.log_metric("final_loss", final_loss)
        mlflow.log_metric("IoU_validation", final_iou) # Changed name to be safe

        # Save artifacts
        model.save(MODEL_OUTPUT_PATH)
        mlflow.log_artifact(MODEL_OUTPUT_PATH, "model_artifact")

        with open(METRICS_FILE, "w") as f:
            json_metrics = {"iou": final_iou, "loss": final_loss}
            json.dump(json_metrics, f)

        print("\n✅ --- Training Run Completed (Dummy Data) ---")
        print(f"Model saved to {MODEL_OUTPUT_PATH}")
        print(f"Metrics saved to {METRICS_FILE}")
        print("-" * 60)


if __name__ == "__main__":
    train_model()
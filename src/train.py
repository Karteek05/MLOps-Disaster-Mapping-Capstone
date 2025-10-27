import mlflow
import tensorflow as tf
import numpy as np
import os
import json
import yaml
import sys
from src.model import get_resnet_unet_model as get_unet_model # Import the model from your model.py
# --- Configuration & Paths ---
# DVC tracks these, so we define them here
MODELS_DIR = "models"
METRICS_FILE = "metrics.json"
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "unet_model.h5")

# --- Load Parameters from params.yaml ---
try:
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)['train']
except FileNotFoundError:
    print("FATAL: params.yaml not found. Please ensure it is in the root directory.")
    sys.exit(1)

# --- Data Placeholders (Mimicking your actual preprocessed data) ---
# These must match the shapes defined in src/model.py
IMG_SIZE = 1024
NUM_CHANNELS = 6
NUM_CLASSES = 5 

def load_small_sample_data():
    """
    Creates small, dummy NumPy arrays to test the pipeline quickly.
    In the final version, this will load data from the data/processed directory.
    """
    print(f"Loading data... (Using DUMMY data for {params['epochs']} epochs)")
    
    # X_train: (batch_size, height, width, channels)
    X_train = np.random.rand(
        params['batch_size'], IMG_SIZE, IMG_SIZE, NUM_CHANNELS
    ).astype(np.float32)

    # Y_train: (batch_size, height, width, 1) - Integer labels (0, 1, 2, 3, 4)
    Y_train = np.random.randint(
        0, NUM_CLASSES, (params['batch_size'], IMG_SIZE, IMG_SIZE, 1)
    ).astype(np.int32)
    
    return X_train, Y_train

# --- Main Training Function ---

def train_model():
    # Ensure the models directory exists for saving artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with mlflow.start_run(run_name=f"Baseline_Run_LR{params['learning_rate']}") as run:
        print("-" * 50)
        print("MLflow Run Started. Logging parameters...")
        
        # 1. LOGGING PARAMETERS
        mlflow.log_params(params)
        
        # 2. MODEL & DATA
        model = get_unet_model()
        X_train, Y_train = load_small_sample_data()
        
        # 3. COMPILE MODEL (This was the error point)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=params['loss_function'], # Reads 'sparse_categorical_crossentropy' from params.yaml
            metrics=['accuracy']
        )
        
        # 4. FIT MODEL
        print(f"Fitting model for {params['epochs']} epochs...")
        Y_train = np.squeeze(Y_train, axis=-1)
        # NOTE: Model fitting will be very fast due to dummy data
        model.fit(X_train, Y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        # 5. MOCK METRICS (Replace with real evaluation later)
        mock_iou = 0.65 
        mlflow.log_metric("IoU_Baseline", mock_iou) 
        mlflow.log_metric("final_loss", 0.15)
        
        # 6. SAVE ARTIFACTS (This saves the model and satisfies DVC)
        model.save(MODEL_OUTPUT_PATH)
        mlflow.log_artifact(MODEL_OUTPUT_PATH, "model_artifact")
        
        # 7. SAVE FINAL METRICS (DVC tracks this metrics.json file)
        with open(METRICS_FILE, "w") as f:
            json.dump({"iou": mock_iou}, f)
        
        print("\n--- Training Run Completed ---")
        print(f"Model saved to {MODEL_OUTPUT_PATH}")
        print(f"Metrics saved to {METRICS_FILE}")
        print("-" * 50)


if __name__ == '__main__':
    train_model()
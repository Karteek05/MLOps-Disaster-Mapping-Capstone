import os
import sys
import json
import yaml
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

# Add the project root to the Python path to allow 'src' imports
# This resolves the ModuleNotFoundError we fixed earlier
sys.path.append(os.getcwd())

from src.model import get_resnet_unet_model # Use the corrected model name

# --- Configuration & Paths ---
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

# --- Define Model Constants from params.yaml for clarity ---
IMG_SIZE = 1024
NUM_CHANNELS = 6
NUM_CLASSES = 5 

# --- Loss Function for Segmentation ---
# We use SparseCategoricalCrossentropy because your masks use integer labels (0, 1, 2, 3, 4)
LOSS_FUNCTION = params['loss_function']

# --- Helper Function for Data Loading ---
def load_small_sample_data():
    """
    Creates small, DUMMY NumPy arrays for fast testing the MLOps pipeline.
    In the final version, this function will be updated to load actual data 
    from the 'data/processed' directory.
    """
    print(f"Loading DUMMY data for {params['epochs']} epochs and batch size {params['batch_size']}")
    
    # X_train: (batch_size, height, width, channels) - The stacked images
    X_train = np.random.rand(
        params['batch_size'], IMG_SIZE, IMG_SIZE, NUM_CHANNELS
    ).astype(np.float32)

    # Y_train: (batch_size, height, width) - The integer mask labels (CRITICAL SHAPE FIX!)
    # We use (B, H, W) because the labels are integers, resolving the ValueError.
    Y_train = np.random.randint(
        0, NUM_CLASSES, (params['batch_size'], IMG_SIZE, IMG_SIZE) 
    ).astype(np.int32)
    
    return X_train, Y_train

# --- Main Training Function ---
def train_model():
    """Builds, compiles, trains, and logs the model artifacts."""
    
    # Ensure the models directory exists for saving artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Start MLflow run to log everything
    with mlflow.start_run(run_name=f"Baseline_Run_LR{params['learning_rate']}") as run:
        print("-" * 50)
        print("MLflow Run Started. Logging parameters...")
        
        # 1. LOGGING PARAMETERS
        mlflow.log_params(params)
        
        # 2. MODEL & DATA
        # Get model architecture (will automatically load weights using the logic in model.py)
        model = get_resnet_unet_model()
        X_train, Y_train = load_small_sample_data()
        
        # 3. COMPILE MODEL
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=LOSS_FUNCTION, 
            metrics=['accuracy'] # Accuracy is simple; IoU/Dice should be added later
        )
        
        # 4. FIT MODEL
        print(f"Fitting model for {params['epochs']} epochs...")
        # NOTE: Model fitting will be very fast due to DUMMY data
        history = model.fit(
            X_train, 
            Y_train, 
            epochs=params['epochs'], 
            batch_size=params['batch_size'], 
            verbose=0 # Suppress verbose output for clean DVC run
        )
        
        # 5. MOCK METRICS & LOGGING (Replace with real evaluation later)
        mock_iou = 0.70  # Assume a reasonable initial IoU for the demo
        
        mlflow.log_metric("IoU_Baseline", mock_iou) 
        mlflow.log_metric("final_loss", history.history['loss'][-1])

        # 6. SAVE ARTIFACTS (Satisfies DVC and MLflow tracking)
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
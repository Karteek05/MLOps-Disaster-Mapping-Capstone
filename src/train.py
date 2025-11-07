import os
import sys
import json
import yaml
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
import mlflow
import mlflow.tensorflow

# Add the project root to the Python path to allow 'src' imports
sys.path.append(os.getcwd())

from src.model import get_resnet_unet_model # Use the corrected model name

# --- Configuration & Paths ---
MODELS_DIR = "models"
METRICS_FILE = "metrics.json"
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "unet_model.h5")
DATA_DIR = "data/processed"

# --- Load Parameters from params.yaml ---
try:
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)['train']
except FileNotFoundError:
    print("FATAL: params.yaml not found. Please ensure it is in the root directory.")
    sys.exit(1)

# --- Define Model Constants ---
IMG_SIZE = 1024
NUM_CHANNELS = 6
NUM_CLASSES = 5 
BATCH_SIZE = params['batch_size']
EPOCHS = params['epochs']
LEARNING_RATE = params['learning_rate']
LOSS_FUNCTION = params['loss_function']

# --- Helper Function for tf.data Pipeline ---

def load_and_parse(image_path_tensor, mask_path_tensor):
    """
    Loads the .npy image stack and the .png mask, 
    and applies the final shape fix for the loss function.
    """
    
    # 1. Load the .npy file (the 6-channel image)
    # We must wrap np.load in tf.numpy_function to use it in the tf.data graph
    def _load_npy(path):
        return np.load(path.decode()).astype(np.float32)
    
    image = tf.numpy_function(_load_npy, [image_path_tensor], tf.float32)
    image.set_shape([IMG_SIZE, IMG_SIZE, NUM_CHANNELS]) # Must set shape after py_function

    # 2. Load the .png file (the 1-channel mask)
    mask = tf.io.read_file(mask_path_tensor)
    mask = tf.io.decode_png(mask, channels=1) # Shape (H, W, 1)
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    
    # 3. CRITICAL SHAPE FIX: Squeeze mask from (H, W, 1) to (H, W)
    # This is required for sparse_categorical_crossentropy loss function
    mask = tf.squeeze(mask) 
    mask = tf.cast(mask, tf.int32)
    
    return image, mask

# --- Main Training Function ---
def train_model():
    """Builds, compiles, trains, and logs the model artifacts."""
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with mlflow.start_run(run_name=f"Final_Run_LR{LEARNING_RATE}") as run:
        print("-" * 50)
        print("MLflow Run Started. Logging parameters...")
        
        # 1. LOGGING PARAMETERS
        mlflow.log_params(params)
        
        # 2. MODEL & DATA
        print("Loading REAL data paths...")
        image_paths = sorted(glob.glob(os.path.join(DATA_DIR, '*_stacked.npy')))
        mask_paths = sorted(glob.glob(os.path.join(DATA_DIR, '*_mask.png')))
        
        if not image_paths:
            print(f"FATAL: No .npy files found in {DATA_DIR}. Check 'dvc pull' or 'prepare' stage.")
            sys.exit(1)
            
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        
        # Create a train/validation split (80% train, 20% validation)
        DATASET_SIZE = len(image_paths)
        TRAIN_SIZE = int(DATASET_SIZE * 0.8)
        
        dataset = dataset.shuffle(DATASET_SIZE, seed=42) # Shuffle paths
        train_dataset = dataset.take(TRAIN_SIZE)
        val_dataset = dataset.skip(TRAIN_SIZE)
        
        # --- Build the tf.data pipeline ---
        AUTOTUNE = tf.data.AUTOTUNE
        
        train_dataset = train_dataset.map(load_and_parse, num_parallel_calls=AUTOTUNE)
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        
        val_dataset = val_dataset.map(load_and_parse, num_parallel_calls=AUTOTUNE)
        val_dataset = val_dataset.batch(BATCH_SIZE)
        val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
        
        print("Data pipeline built. Building model...")
        model = get_resnet_unet_model()
        
        # 3. COMPILE MODEL (With REAL metrics)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=LOSS_FUNCTION, 
            metrics=['accuracy', MeanIoU(num_classes=NUM_CLASSES, name="IoU")]
        )
        
        # 4. FIT MODEL
        print(f"Fitting model on REAL data for {EPOCHS} epochs...")
        history = model.fit(
            train_dataset,
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            validation_data=val_dataset,
            verbose=1 # Show progress in terminal
        )
        
        # 5. LOG REAL METRICS
        final_loss = history.history['loss'][-1]
        final_val_iou = history.history['val_IoU'][-1] # Keras names this 'val_IoU'
        
        mlflow.log_metric("final_loss", final_loss)
        mlflow.log_metric("IoU (validation)", final_val_iou)

        # 6. SAVE ARTIFACTS
        model.save(MODEL_OUTPUT_PATH)
        mlflow.log_artifact(MODEL_OUTPUT_PATH, "model_artifact")
        
        # 7. SAVE FINAL METRICS
        with open(METRICS_FILE, "w") as f:
            json.dump({"iou": final_val_iou, "loss": final_loss}, f)
        
        print("\n--- Training Run Completed ---")
        print(f"Model saved to {MODEL_OUTPUT_PATH}")
        print(f"Metrics saved to {METRICS_FILE}")
        print("-" * 50)

if __name__ == '__main__':
    train_model()
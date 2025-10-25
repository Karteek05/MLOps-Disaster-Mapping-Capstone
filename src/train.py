# src/train.py (Key actions to include)

import mlflow
import tensorflow as tf
import numpy as np
import os
# ... (rest of imports)

# 1. Start the MLflow UI in a new terminal: mlflow ui
# 2. Add the function to load and preprocess a SMALL SAMPLE of data 
#    from data/processed (e.g., 5 images) for a fast test run.
# 3. Inside the train function:

def train_model():
    # Load HPs from a file (best practice) or define them directly
    params = {
        'epochs': 5,  # Keep this low for the first test!
        'learning_rate': 0.001
    }
    
    with mlflow.start_run(run_name="Baseline_Test") as run:
        # --- LOGGING ---
        mlflow.log_params(params)
        
        # --- MODEL & DATA ---
        model = get_unet_model() # Your model from src/model.py
        X_train, Y_train = load_small_sample_data() # Load a few images
        
        model.compile(...) # Define loss/optimizer
        
        model.fit(X_train, Y_train, epochs=params['epochs'])
        
        # --- METRICS (Use IoU, but log a mock score for the first test) ---
        mlflow.log_metric("IoU_Baseline", 0.65) # Mock metric
        
        # --- SAVE ARTIFACTS ---
        MODEL_OUTPUT_PATH = "models/unet_model.h5"
        mlflow.log_artifact(MODEL_OUTPUT_PATH)
        
        # --- SAVE FINAL METRICS (for dvc.yaml to track) ---
        with open("metrics.json", "w") as f:
            json.dump({"iou": 0.65}, f)
        
        print("MLflow run complete. Metrics logged.")
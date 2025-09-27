# src/data_prep.py

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

# ... (Insert the create_mask_from_json and process_image_pair functions here)

def main():
    # This function will run when your DVC pipeline calls it
    RAW_DIR = 'data/raw'
    PROCESSED_DIR = 'data/processed'
    
    # 1. Unzip the downloaded file
    # 2. Loop through all disaster event folders
    # 3. Call process_image_pair() for each pre/post/json set
    
    print("Data processing pipeline script ready.")

if __name__ == '__main__':
    main()
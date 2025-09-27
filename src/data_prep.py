import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
import glob
import tarfile

# Define the mapping for damage grades to integer values (pixel intensity)
# 0 = No building/Background, 1 = No Damage, 2 = Minor, 3 = Major, 4 = Destroyed
DAMAGE_GRADES = {
    'no-damage': 1,
    'minor-damage': 2,
    'major-damage': 3,
    'destroyed': 4
}

def create_mask_from_json(image_path, json_path, output_mask_path):
    """
    Reads a JSON annotation and creates a pixel-level segmentation mask.
    """
    try:
        image = Image.open(image_path)
        # Create a new mask initialized to 0 (Background/No building)
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Iterate over all building polygons in the post-disaster image
        for feature in data.get('features', {}).get('xy', []):
            polygon_wkt = feature['wkt'].replace('POLYGON ((', '').replace('))', '').split(',')
            # Convert WKT coordinates to a list of tuples
            polygon_coords = [tuple(map(float, p.strip().split(' '))) for p in polygon_wkt]
            
            damage_grade = feature['properties'].get('damage_grade', 'no-damage')
            damage_value = DAMAGE_GRADES.get(damage_grade, 0)
            
            # Draw the polygon on the mask with the corresponding damage value
            draw.polygon(polygon_coords, fill=damage_value)
        
        mask.save(output_mask_path)
        return True
    except Exception as e:
        print(f"Error creating mask for {json_path}: {e}")
        return False


def process_event_files(event_id, raw_dir, processed_dir):
    """
    Processes a single pre/post image pair to create a stacked input and a mask output.
    """
    base_name = event_id.replace('_post_disaster', '')
    
    pre_image_path = os.path.join(raw_dir, 'images', f'{base_name}_pre_disaster.png')
    post_image_path = os.path.join(raw_dir, 'images', f'{base_name}_post_disaster.png')
    post_json_path = os.path.join(raw_dir, 'labels', f'{base_name}_post_disaster.json')
    
    # Define output paths
    mask_path = os.path.join(processed_dir, f'{base_name}_mask.png')
    stacked_img_path = os.path.join(processed_dir, f'{base_name}_stacked.npy')

    if not all(os.path.exists(p) for p in [pre_image_path, post_image_path, post_json_path]):
        print(f"Skipping {base_name}: Missing raw files.")
        return

    # 1. Create the damage mask
    if not create_mask_from_json(post_image_path, post_json_path, mask_path):
        return

    # 2. Stack the images
    try:
        pre_img = cv2.imread(pre_image_path)
        post_img = cv2.imread(post_image_path)
        
        # Stack the two 3-channel (RGB) images to create one 6-channel input
        # Note: We skip normalization here; it's often done later by the ML framework (TensorFlow/PyTorch)
        stacked_img = np.dstack((pre_img, post_img)) 
        
        # Save the stacked image to a numpy file 
        np.save(stacked_img_path, stacked_img)
        print(f"Successfully processed {base_name}. Shape: {stacked_img.shape}")
        
    except Exception as e:
        print(f"Error stacking images for {base_name}: {e}")


def extract_data_if_needed(raw_dir):
    """
    Checks for the .tgz archive and extracts it if the main images folder doesn't exist.
    """
    image_folder = os.path.join(raw_dir, 'images')
    if os.path.exists(image_folder):
        print("Data already extracted. Skipping archive step.")
        return

    archive_path = glob.glob(os.path.join(raw_dir, '*train*.tgz'))
    if not archive_path:
        print("Error: No *.tgz archive found in data/raw. Please ensure the file is there.")
        return

    print(f"Extracting data from {os.path.basename(archive_path[0])}...")
    try:
        with tarfile.open(archive_path[0], 'r:gz') as tar:
            tar.extractall(raw_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}. Check if the file is fully downloaded.")


def main():
    """
    Orchestrates the entire data preparation process.
    """
    RAW_DIR = 'data/raw'
    PROCESSED_DIR = 'data/processed'
    
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Check/Extract the raw data
    extract_data_if_needed(RAW_DIR)

    # The image and label files are now expected to be in data/raw/images and data/raw/labels
    labels_path = os.path.join(RAW_DIR, 'labels')
    
    if not os.path.exists(labels_path):
        print("Error: Labels folder not found after extraction. Cannot proceed.")
        return

    # 2. Find all post-disaster JSON files to process
    post_disaster_jsons = glob.glob(os.path.join(labels_path, '*post_disaster.json'))
    
    if not post_disaster_jsons:
        print("No JSON label files found in data/raw/labels. Check extraction path.")
        return

    print(f"Found {len(post_disaster_jsons)} files to process. Starting...")

    for json_file in post_disaster_jsons:
        event_id_with_ext = os.path.basename(json_file).replace('.json', '')
        process_event_files(event_id_with_ext, RAW_DIR, PROCESSED_DIR)

    print("--- Data Preprocessing Complete ---")
    print(f"Processed files saved to {PROCESSED_DIR}")


if __name__ == '__main__':
    main()
import os
import sys

import gradio as gr
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image

sys.path.append(os.getcwd())

from src.metrics import SparseMeanIoU

# --- CONFIGURATION ---
MODEL_PATH = "models/unet_model.h5"
NUM_CLASSES = 5

# Must match whatever resolution the loaded model was actually trained at
# (see params.yaml's train.img_size) - the model's input layer has a fixed
# shape, so this can't just default to the native 1024 image resolution.
with open("params.yaml", "r") as f:
    IMG_SIZE = yaml.safe_load(f)["train"].get("img_size", 1024)

COLOR_MAP = {
    0: [0, 0, 0],        # Background
    1: [0, 255, 0],      # No Damage
    2: [255, 255, 0],    # Minor Damage
    3: [255, 128, 0],    # Major Damage
    4: [255, 0, 0],      # Destroyed
}

_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading trained U-Net model...")
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"SparseMeanIoU": SparseMeanIoU},
            compile=False,  # skip the metric/loss config, we only need inference
        )
        print("Model loaded successfully.")
    return _model


def predict_damage(pre_disaster_image, post_disaster_image):
    """
    Takes a pre/post-disaster image pair, runs prediction, and returns a
    color-coded damage map blended over the post-disaster image.
    """
    if pre_disaster_image is None or post_disaster_image is None:
        raise gr.Error("Please upload both a pre-disaster and a post-disaster image.")

    model = get_model()

    pre_resized = tf.image.resize(pre_disaster_image, [IMG_SIZE, IMG_SIZE])
    post_resized = tf.image.resize(post_disaster_image, [IMG_SIZE, IMG_SIZE])

    # No normalization here - the model was trained on raw [0, 255] pixel
    # values (see src/data_loader.py, which never divides by 255), so
    # inference must match that or the model's activations are thrown off.
    pre_norm = tf.cast(pre_resized, tf.float32)
    post_norm = tf.cast(post_resized, tf.float32)

    stacked_image = np.dstack((pre_norm, post_norm))
    input_tensor = np.expand_dims(stacked_image, axis=0)  # add batch dimension

    print("Running model prediction...")
    prediction_mask = model.predict(input_tensor)
    prediction_labels = np.argmax(prediction_mask[0], axis=-1)

    class_names = {0: "background", 1: "no_damage", 2: "minor", 3: "major", 4: "destroyed"}
    counts = {class_names[c]: int((prediction_labels == c).sum()) for c in class_names}
    print(f"Predicted class pixel counts (of {prediction_labels.size} total): {counts}")

    output_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        output_image[prediction_labels == class_id] = color

    overlay = Image.fromarray(output_image).convert("RGBA")
    original_resized_pil = Image.fromarray(post_resized.numpy().astype(np.uint8)).convert("RGBA")
    blended_image = Image.blend(original_resized_pil, overlay, alpha=0.5)

    return blended_image


LEGEND = """
**Legend:** 🟩 No Damage &nbsp;&nbsp; 🟨 Minor Damage &nbsp;&nbsp; 🟧 Major Damage &nbsp;&nbsp; 🟥 Destroyed
"""

iface = gr.Interface(
    fn=predict_damage,
    inputs=[
        gr.Image(label="Pre-Disaster Image"),
        gr.Image(label="Post-Disaster Image"),
    ],
    outputs=gr.Image(label="Predicted Damage Assessment Map"),
    title="Disaster Damage Mapping",
    description=(
        "Upload a pre-disaster and post-disaster satellite image pair of the same area "
        "to get a building damage assessment map.\n\n" + LEGEND
    ),
    analytics_enabled=False,
)

if __name__ == "__main__":
    iface.launch(show_error=True)

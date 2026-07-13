import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import SparseMeanIoU  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models" / "unet_model.h5"))
PARAMS_PATH = os.environ.get("PARAMS_PATH", str(PROJECT_ROOT / "params.yaml"))

with open(PARAMS_PATH, "r") as f:
    IMG_SIZE = yaml.safe_load(f)["train"].get("img_size", 1024)

CLASS_NAMES = {0: "background", 1: "no_damage", 2: "minor_damage", 3: "major_damage", 4: "destroyed"}
COLOR_MAP = {0: (0, 0, 0), 1: (0, 255, 0), 2: (255, 255, 0), 3: (255, 128, 0), 4: (255, 0, 0)}

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = tf.keras.models.load_model(
                    MODEL_PATH,
                    custom_objects={"SparseMeanIoU": SparseMeanIoU},
                    compile=False,
                )
    return _model


def predict_damage(pre_bytes: bytes, post_bytes: bytes) -> tuple[Image.Image, dict[int, int], int]:
    model = get_model()

    pre = np.array(Image.open(io.BytesIO(pre_bytes)).convert("RGB"))
    post = np.array(Image.open(io.BytesIO(post_bytes)).convert("RGB"))

    pre_resized = tf.image.resize(pre, [IMG_SIZE, IMG_SIZE])
    post_resized = tf.image.resize(post, [IMG_SIZE, IMG_SIZE])

    # No normalization - the model was trained on raw [0, 255] pixel values
    # (see src/data_loader.py), so inference must match that.
    pre_norm = tf.cast(pre_resized, tf.float32)
    post_norm = tf.cast(post_resized, tf.float32)
    stacked = np.dstack((pre_norm, post_norm))

    with _model_lock:
        prediction_mask = model.predict(np.expand_dims(stacked, axis=0))

    labels = np.argmax(prediction_mask[0], axis=-1)
    counts = {c: int((labels == c).sum()) for c in CLASS_NAMES}

    output = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        output[labels == class_id] = color

    overlay = Image.fromarray(output).convert("RGBA")
    base = Image.fromarray(post_resized.numpy().astype(np.uint8)).convert("RGBA")
    blended = Image.blend(base, overlay, alpha=0.5)

    return blended, counts, IMG_SIZE

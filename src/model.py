import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, Concatenate, Activation,
    BatchNormalization, UpSampling2D
)
from tensorflow.keras.models import Model

# --- CONFIG ---
IMG_SIZE = 1024        # H=W
NUM_CHANNELS = 6       # 3 (pre) + 3 (post)
NUM_CLASSES = 5        # background + 4 damage levels

# ---- Blocks ----
def conv_block(tensor, filters, name):
    x = Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(tensor)
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = Activation("relu", name=f"{name}_relu1")(x)

    x = Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = BatchNormalization(name=f"{name}_bn2")(x)
    x = Activation("relu", name=f"{name}_relu2")(x)
    return x

def decoder_block(x_in, skip, filters, name):
    x = Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"{name}_upconv")(x_in)
    x = Concatenate(axis=-1, name=f"{name}_concat")([x, skip])
    x = conv_block(x, filters, name=f"{name}_conv")
    return x

# ---- Model ----
def get_resnet_unet_model():
    """
    UNet with ResNet50 encoder.
    Fixes:
      - Ensures final logits are 1024x1024 (no 2048 blow-up).
      - Adapts 6-channel input to 3-channel for pretrained ImageNet weights.
    """
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name="stacked_input")

    # Adapt 6 -> 3 channels so we can use weights="imagenet" stably.
    x_in = Conv2D(3, 1, padding="same", name="six_to_three")(inputs)

    # ResNet50 encoder (downsampling factor = 32)
    base = ResNet50(include_top=False, weights="imagenet", input_tensor=x_in)

    # Skip connections (spatial sizes for 1024x1024 input):
    # conv1_relu: 512x512, conv2_block3_out: 256x256, conv3_block4_out: 128x128,
    # conv4_block6_out: 64x64, bottleneck conv5_block3_out: 32x32
    skip1 = base.get_layer("conv4_block6_out").output   # 64x64
    skip2 = base.get_layer("conv3_block4_out").output   # 128x128
    skip3 = base.get_layer("conv2_block3_out").output   # 256x256
    skip4 = base.get_layer("conv1_relu").output         # 512x512
    bottleneck = base.get_layer("conv5_block3_out").output  # 32x32

    # Decoder: 32->64->128->256->512
    d1 = decoder_block(bottleneck, skip1, 512, name="dec1")   # 32->64
    d2 = decoder_block(d1,        skip2, 256, name="dec2")    # 64->128
    d3 = decoder_block(d2,        skip3, 128, name="dec3")    # 128->256
    d4 = decoder_block(d3,        skip4, 64,  name="dec4")    # 256->512

    # Final upsample: 512 -> 1024 (only ONCE)
    x = UpSampling2D(size=(2, 2), name="up_final")(d4)        # 512->1024
    x = conv_block(x, 64, name="final")

    # Output: 1024x1024xNUM_CLASSES (use softmax since your loss is sparse_categorical_crossentropy)
    outputs = Conv2D(NUM_CLASSES, 1, activation="softmax", name="output_segmentation")(x)

    return Model(inputs=inputs, outputs=outputs, name="ResNetUNet_xBD")

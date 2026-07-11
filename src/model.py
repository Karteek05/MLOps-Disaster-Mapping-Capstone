import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, Concatenate, Activation,
    BatchNormalization, UpSampling2D
)
from tensorflow.keras.models import Model

# --- Defaults for real training ---
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
def get_resnet_unet_model(
    img_size=IMG_SIZE,
    num_channels=NUM_CHANNELS,
    num_classes=NUM_CLASSES,
    pretrained=True,
    pretrained_trainable=False,
):
    """
    UNet with ResNet50 encoder.
    Ensures final output = img_size x img_size (no 2048 blow-up).
    Adds 6->3 conv for pretrained weights compatibility.

    pretrained: load ImageNet weights (skip_mismatch on the first conv).
                Set False for fast/offline smoke tests.
    pretrained_trainable: whether the ResNet50 encoder is fine-tuned.
                Frozen by default to keep memory/compute down on small GPUs.
    """

    inputs = Input(shape=(img_size, img_size, num_channels), name="stacked_input")

    # Adapt N -> 3 channels
    x_in = Conv2D(3, 1, padding="same", name="six_to_three")(inputs)

    # Backbone
    base = ResNet50(include_top=False, weights=None, input_tensor=x_in)

    if pretrained:
        base.load_weights(
            tf.keras.utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            ),
            by_name=True,
            skip_mismatch=True
        )

    base.trainable = pretrained_trainable

    # Skip connections
    skip1 = base.get_layer("conv4_block6_out").output
    skip2 = base.get_layer("conv3_block4_out").output
    skip3 = base.get_layer("conv2_block3_out").output
    skip4 = base.get_layer("conv1_relu").output
    bottleneck = base.get_layer("conv5_block3_out").output

    # Decoder
    d1 = decoder_block(bottleneck, skip1, 512, name="dec1")
    d2 = decoder_block(d1,        skip2, 256, name="dec2")
    d3 = decoder_block(d2,        skip3, 128, name="dec3")
    d4 = decoder_block(d3,        skip4, 64,  name="dec4")

    # Final upsample back to img_size
    x = UpSampling2D(size=(2, 2), name="up_final")(d4)
    x = conv_block(x, 64, name="final")

    # Output layer (float32 for numerical stability under mixed precision)
    outputs = Conv2D(
        num_classes, 1, activation="softmax", name="output_segmentation",
        dtype="float32",
    )(x)

    return Model(inputs=inputs, outputs=outputs, name="ResNetUNet_xBD")

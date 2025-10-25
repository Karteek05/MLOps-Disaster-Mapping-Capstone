import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Activation, BatchNormalization, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file 

# --- CONFIGURATION (Matches your stacked data) ---
IMG_SIZE = 1024       # Images are 1024x1024
NUM_CHANNELS = 6      # 3 (Pre) + 3 (Post) disaster images
NUM_CLASSES = 5       # 0=BKG, 1=No Dmg, 2=Minor, 3=Major, 4=Destroyed

# --- 1. Define the Core Blocks ---

def conv_block(tensor, filters, name):
    """Standard Convolution Block used in the Decoder path."""
    x = Conv2D(filters, 3, padding="same", name=name + "_conv1")(tensor)
    x = BatchNormalization(name=name + "_bn1")(x)
    x = Activation('relu', name=name + "_relu1")(x)
    
    x = Conv2D(filters, 3, padding="same", name=name + "_conv2")(x)
    x = BatchNormalization(name=name + "_bn2")(x)
    x = Activation('relu', name=name + "_relu2")(x)
    return x

def decoder_block(input_tensor, skip_connection, filters, name):
    """Upsamples input and merges with skip connection."""
    # Upsampling using Transposed Convolution
    x = Conv2DTranspose(filters, (2, 2), strides=2, padding='same', name=name + "_upconv")(input_tensor)
    
    # Concatenate with the skip connection from the Encoder
    x = Concatenate(axis=-1, name=name + "_concat")([x, skip_connection])
    
    # Run through convolution block
    x = conv_block(x, filters, name=name + "_convblock")
    return x

# --- 2. The Fixed Model Builder ---

def get_resnet_unet_model():
    """Builds the U-Net model using a pre-trained ResNet50 as the Encoder."""
    
    # Define Input (1024x1024x6 stacked array)
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name="stacked_input")
    
    # --- ENCODER (ResNet50 Backbone) ---
    
    # 1. Load ResNet50 Architecture with NO weights initially
    # We must set weights=None because our input is 6 channels, not 3.
    base_model = ResNet50(weights=None, include_top=False, input_tensor=inputs)
    
    # 2. Manually load the weights, skipping the incompatible input layer
    WEIGHTS_URL = 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    WEIGHTS_PATH = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_URL,
                            cache_subdir='models')

    # This line loads weights for all compatible layers (the entire network minus the first layer)
    # This solves the 'Shape mismatch' error and completes the transfer learning setup.
    base_model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True) 

    # --- Extract Skip Connections (Feature Maps) ---
    skip_connections = [
        base_model.get_layer('conv4_block6_out').output,  # Skip 1 (Deep)
        base_model.get_layer('conv3_block4_out').output,  # Skip 2
        base_model.get_layer('conv2_block3_out').output,  # Skip 3
        base_model.get_layer('conv1_relu').output        # Skip 4 (Shallow)
    ]
    
    bottleneck = base_model.get_layer('conv5_block3_out').output # 16x16x2048

    # --- DECODER (Expansive Path) ---
    
    d1 = decoder_block(bottleneck, skip_connections[0], 1024, name="decoder_1") 
    d2 = decoder_block(d1, skip_connections[1], 512, name="decoder_2")          
    d3 = decoder_block(d2, skip_connections[2], 256, name="decoder_3")          
    d4 = decoder_block(d3, skip_connections[3], 128, name="decoder_4")          
    
    # Final Up-sampling to match the 1024x1024 input size
    x = UpSampling2D(size=(2, 2))(d4)   # 256 -> 512
    x = UpSampling2D(size=(2, 2))(x)   # 512 -> 1024
    
    # Final Output Layer: 5 channels with softmax for multi-class segmentation
    outputs = Conv2D(NUM_CLASSES, 1, activation="softmax", name="output_segmentation")(x)

    model = Model(inputs=[inputs], outputs=[outputs], name="ResNet_UNet_DisasterMapper")
    return model

if __name__ == '__main__':
    model = get_resnet_unet_model()
    model.summary()
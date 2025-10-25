import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Activation, BatchNormalization, UpSampling2D # Added UpSampling2D for simplicity
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file # Needed for manual weight loading

# --- CONFIGURATION (Matches your stacked data) ---
IMG_SIZE = 1024       # Images are 1024x1024
NUM_CHANNELS = 6      # 3 (Pre) + 3 (Post) disaster images
NUM_CLASSES = 5       # 0=BKG, 1=No Dmg, 2=Minor, 3=Major, 4=Destroyed

# --- Core Blocks (Keeping these as you defined them) ---
def conv_block(tensor, filters, name):
    # ... (Your existing conv_block code)
    x = Conv2D(filters, 3, padding="same", name=name + "_conv1")(tensor)
    x = BatchNormalization(name=name + "_bn1")(x)
    x = Activation('relu', name=name + "_relu1")(x)
    
    x = Conv2D(filters, 3, padding="same", name=name + "_conv2")(x)
    x = BatchNormalization(name=name + "_bn2")(x)
    x = Activation('relu', name=name + "_relu2")(x)
    return x

def decoder_block(input_tensor, skip_connection, filters, name):
    # Upsampling using Transposed Convolution
    x = Conv2DTranspose(filters, (2, 2), strides=2, padding='same', name=name + "_upconv")(input_tensor)
    
    # Concatenate with the skip connection from the Encoder
    x = Concatenate(axis=-1, name=name + "_concat")([x, skip_connection])
    
    # Run through convolution block
    x = conv_block(x, filters, name=name + "_convblock")
    return x

# --- 2. The Fixed Model Builder ---

def get_resnet_unet_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name="stacked_input")
    
    # **FIX 1: Load ResNet50 Architecture with NO weights initially**
    # We use None to bypass the shape mismatch on the first layer during initialization
    base_model = ResNet50(weights=None, include_top=False, input_tensor=inputs)
    
    # **FIX 2: Manually load the weights, skipping the first layer**
    # Define the weights path (Keras cache)
    WEIGHTS_URL = 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    WEIGHTS_PATH = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_URL,
                            cache_subdir='models') # Keras already downloaded this file

    # Load weights, skipping the incompatible input layer (6 vs 3 channels)
    # This loads weights for the remaining 49 layers onto your model
    base_model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True) 

    # --- Encoder/Skip Connections (Rest of your logic) ---
    
    # Get outputs of specific layers for Skip Connections (The feature maps)
    skip_connections = [
        base_model.get_layer('conv4_block6_out').output,  # 32x32x1024
        base_model.get_layer('conv3_block4_out').output,  # 64x64x512
        base_model.get_layer('conv2_block3_out').output,  # 128x128x256
        base_model.get_layer('conv1_relu').output        # 256x256x64 
    ]
    
    bottleneck = base_model.get_layer('conv5_block3_out').output # 16x16x2048

    # --- Decoder ---
    d1 = decoder_block(bottleneck, skip_connections[0], 1024, name="decoder_1") # 32x32
    d2 = decoder_block(d1, skip_connections[1], 512, name="decoder_2")          # 64x64
    d3 = decoder_block(d2, skip_connections[2], 256, name="decoder_3")          # 128x128
    d4 = decoder_block(d3, skip_connections[3], 128, name="decoder_4")          # 256x256
    
    # Final Up-sample (You need to adjust this from Strides=4 to match UNet upsampling)
    # Simplest way to upsample 256x256 to 1024x1024 is using 2 steps of UpSampling2D
    x = UpSampling2D(size=(2, 2))(d4)   # 256 -> 512
    x = UpSampling2D(size=(2, 2))(x)   # 512 -> 1024
    
    # Final Output Layer
    outputs = Conv2D(NUM_CLASSES, 1, activation="softmax", name="output_segmentation")(x)

    model = Model(inputs=[inputs], outputs=[outputs], name="ResNet_UNet_DisasterMapper")
    return model

# --- Rest of the file remains the same ---
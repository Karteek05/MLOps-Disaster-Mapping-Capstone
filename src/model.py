import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Activation, BatchNormalization
from tensorflow.keras.models import Model

# --- CONFIGURATION (Matches your stacked data) ---
IMG_SIZE = 1024       # Images are 1024x1024
NUM_CHANNELS = 6      # 3 (Pre) + 3 (Post) disaster images
NUM_CLASSES = 5       # 0=BKG, 1=No Dmg, 2=Minor, 3=Major, 4=Destroyed

# --- 1. Define the Core Blocks ---

def conv_block(tensor, filters, name):
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

# --- 2. Build the Model ---

def get_resnet_unet_model():
    # Define Input (Matching the 1024x1024x6 stacked array)
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name="stacked_input")
    
    # A. The ENCODER (ResNet50 Backbone)
    # Note: Keras ResNet50 is hardcoded for 3 input channels. 
    # We must use a simple Conv layer to change 6 channels -> 3 channels for compatibility.
    # We will adjust the first convolution to handle 6 channels later if needed.
    
    # **Simplified Encoder Setup (The Core)**
    # Load ResNet50 without the top classification layer, using ImageNet weights for transfer learning
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # Get outputs of specific layers for Skip Connections
    # These layers correspond to the output of each downsampling stage in ResNet
    skip_connections = [
        base_model.get_layer('conv4_block6_out').output,  # 32x32x1024 (deepest layer before bottleneck)
        base_model.get_layer('conv3_block4_out').output,  # 64x64x512
        base_model.get_layer('conv2_block3_out').output,  # 128x128x256
        base_model.get_layer('conv1_relu').output        # 256x256x64 (early features)
    ]
    
    # B. The BOTTLENECK (The deepest point in the 'U')
    bottleneck = base_model.get_layer('conv5_block3_out').output # 16x16x2048

    # C. The DECODER (Expansive Path with Skip Connections)
    
    # 1. Start decoding from the bottleneck
    d1 = decoder_block(bottleneck, skip_connections[0], 1024, name="decoder_1") # Rebuild to 32x32
    d2 = decoder_block(d1, skip_connections[1], 512, name="decoder_2")          # Rebuild to 64x64
    d3 = decoder_block(d2, skip_connections[2], 256, name="decoder_3")          # Rebuild to 128x128
    d4 = decoder_block(d3, skip_connections[3], 128, name="decoder_4")          # Rebuild to 256x256
    
    # Final Up-sample to original size (1024x1024). We'll simplify this.
    x = Conv2DTranspose(64, (2, 2), strides=4, padding='same', name="final_upsample")(d4) # 256 -> 1024
    
    # D. Final Output Layer
    # Use 'softmax' for 5 multi-class prediction (damage types)
    outputs = Conv2D(NUM_CLASSES, 1, activation="softmax", name="output_segmentation")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ResNet_UNet_DisasterMapper")
    return model

if __name__ == '__main__':
    # Test the model structure and print the architecture summary
    model = get_resnet_unet_model()
    model.summary()
    
# --- END OF src/model.py ---
import tensorflow as tf
import numpy as np
import tensorflow as tf
import keras
import glob

from path_to_folders import images_path, masks_path
from skimage import measure
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon_perimeter

# Define variables

CLASSES = 2
COLOR_MAP = {
    0: 'magenta',
    1: 'yellow'
}

SAMPLE_SIZE = (384, 384)
OUTPUT_SIZE = (768, 768)

# Function imege load

def image_load(image, mask):
    # Load, decoder and resize image .jpg
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, OUTPUT_SIZE)

    # Convert image data type to float32 and normalizes the pixels to the range [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image/ 255.0

    # Load, decoder and resize mask .png
    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
    # Convert mask to grayscale for it contains one channel
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    masks = []
    
    # For each class (CLASSES) creates a mask, where 1.0 indicates the presence of the class, and 0.0 - its absence
    for i in range(CLASSES):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

    # Combine all class masks into one tensor.
    masks = tf.stack(masks, axis=2)
    # Change the form of the combined mask tensor to match the output format
    masks = tf.reshape(masks, OUTPUT_SIZE+(CLASSES,))

    return image, masks


# Augumintate function
def augmentate_images(image, masks):
    # Randomly select a crop size between 30% to 100% of the original size.
    random_crop = tf.random.uniform((), 0.3, 1)
    # Apply central crop to the image with the randomly selected size.
    image = tf.image.central_crop(image, random_crop)
    # Apply the same central crop to the masks to ensure consistency.
    masks = tf.image.central_crop(masks, random_crop)

    # Generate a random number between 0 and 1 to decide whether to flip the images.
    random_flip = tf.random.uniform((), 0, 1)
    # If the random number is greater than or equal to 0.5, flip the images and masks horizontally.
    if random_flip >= 0.5:
        image = tf.image.flip_left_right(image)
        masks = tf.image.flip_left_right(masks)

    # Resize the cropped and possibly flipped image back to the desired sample size.
    image = tf.image.resize(image, SAMPLE_SIZE)
    # Resize the masks in the same way.
    masks = tf.image.resize(masks, SAMPLE_SIZE)

    # Return the augmented image and masks.
    return image, masks


# Load data from drive
images = sorted(glob.glob(images_path+'*.jpg'))
masks = sorted(glob.glob(masks_path+'*.png'))

#Formation of data sets from images and masks
images_dataset = tf.data.Dataset.from_tensor_slices(images)
masks_dataset = tf.data.Dataset.from_tensor_slices(masks)

# Union datasets for paralel processing
dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

# Load images to memory
dataset = dataset.map(image_load, num_parallel_calls = tf.data.AUTOTUNE)

# Increasing the volume of data by copying
dataset = dataset.repeat(50)

# Use augumentate function for all images
dataset = dataset.map(augmentate_images, num_parallel_calls = tf.data.AUTOTUNE)


# Division of sets into training and test sets
train_dataset = dataset.take(2000).cache()
test_dataset = dataset.skip(2000).take(100).cache()

# Set size of packadge
train_dataset = train_dataset.batch(16)
test_dataset = test_dataset.batch(16)

# Define input neural network model
def input_layer():
    return keras.layers.Input(shape=SAMPLE_SIZE +(3,))


# Describe bloks enkoder format
def downsample_block(filters, size, batch_norm=True):
    # Initialize weights using the Glorot normal initializer, also known as Xavier normal initializer.
    # It is suitable for networks with symmetric activations like tanh or when using the LeakyReLU.
    initializer = keras.initializers.GlorotNormal()

    # Create a sequential model to encapsulate the operations in this block.
    result = keras.Sequential()
    
    # Add a Conv2D layer for downsampling.
    # - `filters` specifies the number of output filters in the convolution.
    # - `size` is the kernel size.
    # - Strides of 2 reduce the dimensions of the input feature map by half, achieving downsampling.
    # - Padding is set to 'same' to ensure that the output size is mathematically computed to match the input size as closely as possible when the stride is 2.
    # - The kernel initializer is specified to use the Glorot normal method.
    # - `use_bias` is set to False when it is expected to be followed by a BatchNormalization layer, which will add a bias.
    result.add(keras.layers.Conv2D(filters, size, strides=2, padding='same', 
                                   kernel_initializer=initializer, use_bias=False))
    
    # Optionally add a BatchNormalization layer if `batch_norm` is True.
    # This layer normalizes the activations from the previous layer, which can help in stabilizing the learning process and speed up convergence.
    if batch_norm:
        result.add(keras.layers.BatchNormalization())

    # Add a LeakyReLU activation function.
    # Unlike the standard ReLU, LeakyReLU allows a small, non-zero gradient when the unit is not active, which can help prevent dead neurons during training.
    result.add(keras.layers.LeakyReLU())
    
    # Return the sequential model comprising the downsample block.
    return result


# Format decoder block for neural network
def upsample_block(filters, size, dropout=False):
    # Initialize weights using the Glorot normal initializer, also known as Xavier normal initializer.
    # It is a good default when using ReLU activation functions.
    initializer = keras.initializers.GlorotNormal()

    # Create a sequential model to encapsulate the operations in this block.
    result = keras.Sequential()
    
    # Add a Conv2DTranspose layer, which performs the upsampling operation.
    # - `filters` determines the number of output filters in the convolution.
    # - `size` is the kernel size.
    # - Strides of 2 double the dimensions of the input feature map, achieving the upsampling.
    # - Padding is set to 'same' to ensure the output has the same dimensions when the stride is 1.
    # - The kernel initializer is specified to use the Glorot normal method.
    # - `use_bias` is set to False when followed by a BatchNormalization layer, which will add a bias.
    result.add(keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', 
                                            kernel_initializer=initializer, use_bias=False))
    
    # Add a BatchNormalization layer to stabilize learning and reduce the number of training epochs required.
    result.add(keras.layers.BatchNormalization())

    # Optionally add a Dropout layer to prevent overfitting by randomly setting input units to 0 during training at a rate of 0.25.
    # This is particularly useful in deep networks or when a limited amount of training data is available.
    if dropout:
        result.add(keras.layers.Dropout(0.25))
        
    # Add a ReLU activation function to introduce non-linearity, allowing the model to learn more complex patterns.
    result.add(keras.layers.ReLU())
    
    # Return the sequential model comprising the upsample block.
    return result

# Define output neural network model
def output_layer(size):
    initializer = keras.initializers.GlorotNormal()
    return keras.layers.Conv2DTranspose(CLASSES, size, strides=2, padding='same', kernel_initializer=initializer,
                                        activation= 'sigmoid')

# Create stack layers
inp_layer = input_layer()

# Define the architecture components as specified
downsample_stack = [
    downsample_block(64, 4, batch_norm = False),
    downsample_block(128, 4),
    downsample_block(256, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
]

upsample_stack = [
    upsample_block(512, 4, dropout = True),
    upsample_block(512, 4, dropout = True),
    upsample_block(512, 4, dropout = True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4),
]

out_layer = output_layer(4) # Adjusted to clarify it's a function returning a layer

# Integrate these components into the U-Net-like model with skip connections
# (Refer to the previously provided skip connection realization code)

# The actual implementation of downsample_block, upsample_block, and output_layer functions 
# need to be provided for a complete model. This outline shows the structural setup.


# Skip connection realization
x = inp_layer

# Initialize a list to store references to the outputs of the downsampling blocks,
# which will be used later for skip connections.
downsample_skips = []

# Iterate over each block in the downsampling part of the network.
for block in downsample_stack:
    # Apply the downsampling block to the current tensor 'x'
    x = block(x)
    # Store the output of the downsampling block for later use in skip connections
    downsample_skips.append(x)

# Reverse the list of downsampling outputs, excluding the final output,
# to align them for concatenation with the corresponding upsampling layers.
# The final output is excluded because it is typically not used in a skip connection
# as there's no corresponding upsampling layer.
downsample_skips = reversed(downsample_skips[:-1])

# Iterate over each block in the upsampling part of the network, along with the reversed
# downsampling outputs for skip connections.
for up_block, down_block in zip(upsample_stack, downsample_skips):
    # Apply the upsampling block to the current tensor 'x'
    x = up_block(x)
    # Concatenate the output of the upsampling block with the corresponding downsampling output
    # This is the skip connection, reintroducing lost spatial information to the network
    x = keras.layers.Concatenate()([x, down_block])

# Apply the final output layer to the tensor 'x'
out_layer = out_layer(x)

# Create the U-Net-like model with the specified input and output layers
unet_like = keras.Model(inputs=inp_layer, outputs=out_layer)

# Define a function to calculate the Dice coefficient for multi-class segmentation tasks.
def dice_mc_metric(a, b):
    # Unstack the tensors along the channel dimension to process each class separately.
    a = tf.unstack(a, axis=3)
    b = tf.unstack(b, axis=3)
    
    # Initialize a variable to accumulate the Dice coefficients for all classes.
    dice_summ = 0
    
    # Iterate over each class, calculating the Dice coefficient for each.
    for i, (aa, bb) in enumerate(zip(a, b)):
        # Calculate the numerator of the Dice coefficient: 2 * intersection + smooth factor.
        numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
        # Calculate the denominator of the Dice coefficient: sum of each element in both sets + smooth factor.
        denomerator = tf.math.reduce_sum(aa + bb) + 1
        # Accumulate the average Dice coefficient by adding the ratio of the numerator to the denominator.
        dice_summ += numenator / denomerator
        
    # Calculate the average Dice coefficient across all classes.
    avg_dice = dice_summ / CLASSES
    
    # Return the average Dice coefficient.
    return avg_dice

# Define a function to calculate the loss based on the Dice coefficient for multi-class segmentation.
def dice_mc_loss(a, b):
    # Calculate the Dice loss, which is 1 minus the Dice coefficient.
    return 1 - dice_mc_metric(a, b)

# Define a function that combines Dice loss with Binary Cross-Entropy (BCE) loss for multi-class segmentation.
def dice_bce_mc_loss(a, b):
    # Calculate a weighted sum of Dice loss and BCE loss.
    # This combines the benefits of Dice loss (sensitivity to the shape match) and BCE loss (pixel-wise error).
    return 0.3 * dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)


# Compiling the model
unet_like.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_metric])

# Train neural network and save result
history_dice = unet_like.fit(train_dataset, validation_data=test_dataset, epochs=25, initial_epoch=0)

unet_like.save_weights('D:\IT\Python\Ship_Detection\Airbus_Ship_Detection_Kaggle\Resource\model\\')

# Load sempl model
unet_like.load_weights('D:\IT\Python\Ship_Detection\Airbus_Ship_Detection_Kaggle\Resource\model\\')


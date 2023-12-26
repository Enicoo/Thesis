# Let's implement the salt and pepper noise algorithm according to the provided conditions.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def add_salt_and_pepper_noise(image_array, noise_ratio):
    """
    Add salt and pepper noise to an image array based on the provided conditions.
    noise_ratio: The probability for each pixel to be turned into salt or pepper noise.
    """
    # Create a copy of the image array to not modify the original one
    noisy_image = np.copy(image_array)

    # Flatten the image array to work with it as a 1D array
    flat_image = image_array.flatten()

    # Generate random values for each pixel
    random_values = np.random.rand(flat_image.shape[0])

    # Apply salt noise where the random value is less than the noise_ratio
    salt_indices = random_values < noise_ratio
    flat_image[salt_indices] = 255

    # Apply pepper noise where the random value is greater than 1 - noise_ratio
    pepper_indices = random_values > (1 - noise_ratio)
    flat_image[pepper_indices] = 0

    # Reshape the flat image back to the original image shape
    noisy_image = flat_image.reshape(image_array.shape)

    return noisy_image

# Since we don't have an actual image, let's create a grayscale dummy image of size 256x256
image_path = 'D:/Users/Pictures/thesis pictures/original5x5.png'  # Replace with the correct path to your image file
image = Image.open(image_path).convert('L')  # Convert to grayscale
image_array = np.array(image)  # Convert the PIL image to a NumPy array

# Define the noise level at 5%
noise_level = 0.15

# Apply the salt and pepper noise according to the conditions provided
sp_noisy_image = add_salt_and_pepper_noise(image_array, noise_level)

# Display the original and noisy images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original Image
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

# Image with Salt and Pepper Noise
axs[1].imshow(sp_noisy_image, cmap='gray')
axs[1].set_title('Image with Salt and Pepper Noise')
axs[1].axis('off')

plt.show()

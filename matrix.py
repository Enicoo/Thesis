import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'D:/Users/Pictures/thesis pictures/small.png'
image = Image.open(image_path)

# Convert the image to RGB if it's not already
if image.mode != 'RGB':
    image = image.convert('RGB')
    
# Convert the image to a NumPy array
rgb_image = np.array(image)

# Define the transformation matrix
transformation_matrix = np.array([
    [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    [1 / np.sqrt(6), 0, -1 / np.sqrt(6)],
    [1 / (3 * np.sqrt(2)), -2 / (3 * np.sqrt(2)), 1 / (3 * np.sqrt(2))]
])

# Reshape the RGB image to a column vector
rgb_vector = rgb_image.reshape(-1, 3)

# Apply the transformation
ycc_matrix = np.dot(transformation_matrix, rgb_vector.T).T

# Reshape the result back to the original image shape
ycc_image = ycc_matrix.reshape(rgb_image.shape)

# Extract the transformed Y, C1, and C2 components
Y_matrix = ycc_image[:, :, 0]
C1_matrix = ycc_image[:, :, 1]
C2_matrix = ycc_image[:, :, 2]

# Normalize the Y, C1, and C2 matrices for displaying as images
Y_normalized = ((Y_matrix - np.min(Y_matrix)) / (np.max(Y_matrix) - np.min(Y_matrix))) * 255
C1_normalized = ((C1_matrix - np.min(C1_matrix)) / (np.max(C1_matrix) - np.min(C1_matrix))) * 255
C2_normalized = ((C2_matrix - np.min(C2_matrix)) / (np.max(C2_matrix) - np.min(C2_matrix))) * 255

# Convert the normalized matrices to uint8 for image display
Y_image = Y_normalized.astype(np.uint8)
C1_image = C1_normalized.astype(np.uint8)
C2_image = C2_normalized.astype(np.uint8)

# Save the images
output_folder = 'D:/Users/Pictures/thesis pictures/'
Y_image_pil = Image.fromarray(Y_image)
C1_image_pil = Image.fromarray(C1_image)
C2_image_pil = Image.fromarray(C2_image)

Y_image_pil.save(output_folder + 'Y_channel.png')
C1_image_pil.save(output_folder + 'C1_channel.png')
C2_image_pil.save(output_folder + 'C2_channel.png')

# Display the images with labels using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(Y_image, cmap='gray')
axes[0].set_title('Y Channel')

axes[1].imshow(C1_image, cmap='gray')
axes[1].set_title('C1 Channel')

axes[2].imshow(C2_image, cmap='gray')
axes[2].set_title('C2 Channel')

for ax in axes:
    ax.axis('off')

plt.show()

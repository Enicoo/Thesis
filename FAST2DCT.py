import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dct as scipy_dct, idct as scipy_idct

def compute_dct_matrix(f):
    return scipy_dct(scipy_dct(f.T, norm='ortho').T, norm='ortho')

def compute_idct_matrix(F):
    return scipy_idct(scipy_idct(F.T, norm='ortho').T, norm='ortho')

def display_idct(dct_matrix):
    # Compute the inverse DCT
    reconstructed_image = compute_idct_matrix(dct_matrix)
    
    # Normalize pixel values to the range [0, 255] for visualization
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    
    # Display the reconstructed image using Matplotlib
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image from DCT Coefficients')
    plt.axis('off')
    plt.show()
    
    return reconstructed_image

def display_dct(image_matrix):
     # Load the grayscale image as a matrix
    image = Image.open(image_matrix).convert('L')  # Ensure image is loaded in grayscale
    image_matrix = np.array(image)
    # Compute the DCT for the entire image
    dct_matrix = compute_dct_matrix(image_matrix)
    
    # Display the DCT coefficients as an image
    dct_coefficients_visual = np.log(np.abs(dct_matrix) + 1)  # Log scale for visualization
    plt.imshow(dct_coefficients_visual, cmap='gray')
    plt.title('DCT Coefficients')
    plt.colorbar()
    plt.show()
    
    return dct_matrix

def display_dct2(image_matrix):
    dct_matrix = compute_dct_matrix(image_matrix)
    
    # Display the DCT coefficients as an image
    dct_coefficients_visual = np.log(np.abs(dct_matrix) + 1)  # Log scale for visualization
    plt.imshow(dct_coefficients_visual, cmap='gray')
    plt.title('DCT Coefficients')
    plt.colorbar()
    plt.show()
    
    return dct_matrix


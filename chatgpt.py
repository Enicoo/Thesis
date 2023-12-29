import numpy as np
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt

# Helper function to load an image
def load_image(image_path):
    from PIL import Image
    return Image.open(image_path)

# Helper function to convert image to YCbCr color space
def convert_to_YCbCr(image):
    # Convert the image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    rgb_image = np.array(image)
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
    
    
    return Y_image,C1_image,C2_image

# Helper function to convert YCbCr back to RGB
def convert_to_RGB(y, cb, cr):
    transformation_matrix = np.array([
    [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    [1 / np.sqrt(6), 0, -1 / np.sqrt(6)],
    [1 / (3 * np.sqrt(2)), -2 / (3 * np.sqrt(2)), 1 / (3 * np.sqrt(2))]
])
    # Inverse of the transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Assuming Y_image, C1_image, C2_image are the matrices we obtained earlier and have the same shape
    # Recombine the Y, C1, and C2 channels into a single matrix
    ycc_combined = np.stack((y,cb,cr), axis=-1)

    # Reshape the combined YC1C2 image to a column vector for matrix multiplication
    ycc_vector = ycc_combined.reshape(-1, 3)

    # Apply the inverse transformation
    rgb_matrix = np.dot(inverse_transformation_matrix, ycc_vector.T).T

    # Reshape the result back to the original image shape
    rgb_reconstructed = rgb_matrix.reshape(y.shape[0], y.shape[1], 3)

    # Normalize and convert the values to uint8 if necessary
    rgb_reconstructed_normalized = np.clip(rgb_reconstructed, 0, 255).astype(np.uint8)
    # Convert the NumPy array to an image
    reconstructed_image = Image.fromarray(rgb_reconstructed_normalized)

    # Display the reconstructed image
    plt.imshow(reconstructed_image)
    plt.title("Reconstructed RGB Image")
    plt.axis('off')
    plt.show()

# Placeholder for any filters you apply to the Y channel
def apply_filters(channel):
    # Apply any pre-processing filters here
    return channel

# Placeholder for any post-processing
def post_process(channel):
    # Apply any post-processing here
    return channel

def compute_2d_dct(image_matrix):
    print("Im here")
    return dct(dct(image_matrix.T, norm='ortho').T, norm='ortho')

def compute_2d_idct(dct_matrix):
    return idct(idct(dct_matrix.T, norm='ortho').T, norm='ortho')

def threshold_dct(dct_matrix, threshold):
    print("thresholding")
    return np.where(np.abs(dct_matrix) > threshold, dct_matrix, 0)

def compute_alpha(dct_matrix):
    print("alpha!")
    # Compute alpha based on the energy of the DCT coefficients
    # Placeholder for actual implementation
    N = len(dct_matrix)  # Assuming a square DCT coefficient matrix
    F_00 = dct_matrix[0, 0]  # DC component
    sum_of_absolute_values = np.sum(np.abs(dct_matrix))  # Sum of absolute values excluding DC
    E = (sum_of_absolute_values  - np.abs(F_00) ) / ((N * N) - 1)  # Compute E
    En = E * -0.0052 + 1
    alpha = (1 - 0.0052 * E) * En
    
    return alpha



def calculate_Eh_Ev(dct_matrix):
    print("ewan")
    rows, cols = dct_matrix.shape  # Adjusting for non-square matrices
    Eh = 0
    Ev = 0

    for k in range(rows):
        for l in range(cols):
            Wh = 1 if k > l else 0
            Wv = 1 if k < l else 0
            Eh += Wh * abs(dct_matrix[k, l])
            Ev += Wv * abs(dct_matrix[k, l])

    return Eh, Ev

def calculate_THV(dct_matrix,rows,cols):
    print("letsgoo")
    T = np.zeros((rows, cols))
    H = np.zeros((rows, cols))
    V = np.zeros((rows, cols))
    Eh, Ev = calculate_Eh_Ev(dct_matrix)
    
    for k in range(rows):
        for l in range(cols):
            k_val = k if k != 0 else 1
            l_val = l if l != 0 else 1
            H[k, l] = 1.25 - (np.arctan(l_val / k_val) * (180 / np.pi)) / 90
            V[k, l] = 0.25 + (np.arctan(k_val / l_val) * (180 / np.pi)) / 90
            T[k, l] = (Eh / (Eh + Ev)) * H[k, l] + (Ev / (Eh + Ev)) * V[k, l]
    
    return T



def adaptive_attenuator(dct_matrix, alpha):
    print("malapit na")
    rows, cols = dct_matrix.shape
    T = calculate_THV(dct_matrix,rows,cols)
    # Apply the adaptive attenuation on the DCT coefficients
    # Placeholder for actual implementation
    attenuated_dct_matrix = (1 - alpha) * T * dct_matrix
    return attenuated_dct_matrix

def calculate_delta(dct_matrix):
    print("closer!!")
    DC_component = dct_matrix[0, 0]
    delta_DC= np.zeros_like(dct_matrix)
    delta_AC = np.zeros_like(dct_matrix)
    N = dct_matrix.shape[0]
    
    for i in range(N):
        for j in range(N):
            if i == 0 and j == 0:
                continue  # Skip the DC component itself
            delta_DC[i, j] = DC_component - dct_matrix[i, j]
    for x in range(N):
        for y in range(N):
            ac_sum = 0
            for i in range(N):
                for j in range(N):
                    if i == 0 and j == 0:
                        continue  # Skip the DC component
                    ac_sum += dct_matrix[i, j] - dct_matrix[(x-i) % N, (y-j) % N]
    
    # Subtract ΔDC(x, y, i, j) from the accumulated differences
    delta_AC -= delta_DC
    
    return delta_AC
    
  

def similar_patch_blender(dct_matrix, delta, sigma1, sigma2):
    print("getting there")
    M = dct_matrix.shape[0]
    blended_dct_matrix = np.zeros_like(dct_matrix)

    for k in range(M):
        for l in range(M):
            numerator_sum = 0
            denominator_sum = 0

            for i in range(-M//2, M//2):
                for j in range(-M//2, M//2):
                    # Periodic boundary conditions
                    periodic_i = (i + k) % M
                    periodic_j = (j + l) % M
                    
                    weight_spatial = np.exp(-(i**2 + j**2) / (2 * sigma1**2))
                    weight_delta = np.exp(-delta[periodic_i, periodic_j]**2 / (2 * sigma2**2))
                    
                    numerator_sum += dct_matrix[periodic_i, periodic_j] * weight_spatial * weight_delta
                    denominator_sum += weight_spatial * weight_delta

            # Update the DCT coefficient at position (k, l)
            blended_dct_matrix[k, l] = numerator_sum / denominator_sum if denominator_sum != 0 else 0
    return blended_dct_matrix

def denoise_image(image_path, threshold, sigma1, sigma2):
    # Load image and convert to YCbCr or any other color space if needed
    image = load_image(image_path)
    y, cb, cr = convert_to_YCbCr(image)
    
    # # Apply filters to the Y channel if needed
    # y_filtered = apply_filters(y)
    
    # Perform 2D DCT
    dct_matrix = compute_2d_dct(y)
    
    # Thresholding
    dct_thresholded = threshold_dct(dct_matrix, threshold)
    
    # Compute alpha
    alpha = compute_alpha(dct_thresholded)
    
    # Apply adaptive attenuation
    dct_attenuated = adaptive_attenuator(dct_thresholded, alpha)
    
    # Calculate delta for blending
    delta = calculate_delta(dct_attenuated)
    
    # Blend similar patches
    dct_blended = similar_patch_blender(dct_attenuated, delta, sigma1, sigma2)
    
    # Perform 2D iDCT
    idct_matrix = compute_2d_idct(dct_blended)
    
    # # Post-process if needed and convert back to RGB
    # denoised_image = post_process(idct_matrix)
    rgb_image = convert_to_RGB(idct_matrix, cb, cr)
    
    return rgb_image



# Use the function
denoised_image = denoise_image('D:/Users/Pictures/thesis pictures/small.png', threshold=1, sigma1=4, sigma2=10)

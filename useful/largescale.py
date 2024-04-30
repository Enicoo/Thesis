import numpy as np
import skimage as sk
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# Helper function to load an image

def load_image(image_path):
    from PIL import Image
    return Image.open(image_path)

# Helper function to convert image to YCbCr color space
def convert_to_YCbCr(image):
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

    # Extract the transformed Y, Cb, and Cr components
    Y_matrix = ycc_image[:, :, 0]
    Cb_matrix = ycc_image[:, :, 1]
    Cr_matrix = ycc_image[:, :, 2]
    # Y_matrix= ((Y_matrix - np.min(Y_matrix)) / (np.max(Y_matrix) - np.min(Y_matrix))) * 255
    # Cb_matrix = ((Cb_matrix - np.min(Cb_matrix)) / (np.max(Cb_matrix) - np.min(Cb_matrix))) * 255
    # Cr_matrix = ((Cr_matrix - np.min(Cr_matrix)) / (np.max(Cr_matrix) - np.min(Cr_matrix))) * 255
    # Y_matrix = Y_matrix.astype(np.uint8)
    # Cb_matrix =  Cb_matrix.astype(np.uint8)
    # Cr_matrix = Cr_matrix.astype(np.uint8)
    # Y_matrix = Y_matrix.astype(np.uint8)
    # Cb_matrix = Cb_matrix.astype(np.uint8)
    # Cr_matrix = Cr_matrix.astype(np.uint8)
    # plt.imshow(Y_normalized)
    # plt.title("Y_matrix")
    # plt.axis('off')
    # plt.show()
    print("Y = ",  Y_matrix)
    print("Cb = ",  Cb_matrix)
    print("Cr = ",   Cr_matrix)
    # Return the Y, Cb, and Cr matrices
    return Y_matrix, Cb_matrix, Cr_matrix

# Helper function to convert YCbCr back to RGB
def convert_to_RGB(Y, Cb, Cr):
    Y = Y.astype(float)
    Cb = Cb.astype(float)
    Cr = Cr.astype(float)

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    rgb_reconstructed = np.stack((R, G, B), axis=-1)
    reconstructed_image = Image.fromarray(rgb_reconstructed)

    # Display the reconstructed image
    plt.imshow(reconstructed_image, cmap="gray")
    plt.title("Reconstructed RGB Image")
    plt.axis('off')
    plt.show()

def compute_2d_dct(image_matrix):
    return dct(dct(image_matrix.T, norm='ortho').T, norm='ortho')

def compute_2d_idct(dct_matrix):
    
    return idct(idct(dct_matrix.T, norm='ortho').T, norm='ortho')

def threshold_dct(dct_coefficients, threshold):
    thresholded_dct = np.copy(dct_coefficients)
    for k in range(dct_coefficients.shape[0]):
        for l in range(dct_coefficients.shape[1]):
            if k == 0 and l == 0:
                continue
            else:
                if dct_coefficients[k, l] > threshold:
                    thresholded_dct[k, l] = dct_coefficients[k,l] - threshold
                elif dct_coefficients[k, l] < -threshold:
                    thresholded_dct[k, l] = threshold + dct_coefficients[k,l]
                elif np.abs(dct_coefficients[k, l]) <= threshold:
                    thresholded_dct[k, l] = 0
    
    print("Thresholded DCT = ", np.array2string(thresholded_dct, separator=', '))
    return thresholded_dct

def compute_alpha(dct_coefficients):
    N = len(dct_coefficients)
    F_00 = dct_coefficients[0, 0]
    sum_of_absolute_values = np.sum(np.abs(dct_coefficients))
    E = (sum_of_absolute_values - np.abs(F_00)) / ((N * N) - 1)
    En = E * -0.05+ 1
    Alpha = (1 - 0.0052 * E) * En
    print(E)
    print(Alpha)
    return Alpha

def compute_alphaC(dct_coefficients):
    N = len(dct_coefficients)
    F_00 = dct_coefficients[0, 0]
    sum_of_absolute_values = np.sum(np.abs(dct_coefficients))
    E = (sum_of_absolute_values - np.abs(F_00)) / ((N * N) - 1)
    En = E * -0.25+ 1
    Alpha = (1 - 0.0082 * E) * En
    return Alpha

def calculate_Eh_Ev(dct_coefficients):
    rows, cols = dct_coefficients.shape
    Eh = 0
    Ev = 0
    for k in range(rows):
        for l in range(cols):
            Wh = 1 if k > l else 0
            Wv = 1 if k < l else 0
            Eh += Wh * abs(dct_coefficients[k, l])
            Ev += Wv * abs(dct_coefficients[k, l])
    print("Eh = ", Eh)
    print("Ev = ", Ev)
    return Eh, Ev

def calculate_THV(rows, cols, Eh, Ev):
    T = np.zeros((rows, cols))
    H = np.zeros((rows, cols))
    V = np.zeros((rows, cols))
    
    for k in range(rows):
        for l in range(cols):
            k_val = k if k != 0 else 1
            l_val = l if l != 0 else 1
            
            H[k, l] = 1.25 - (np.arctan(l_val / k_val) * 180 / np.pi) / 90
            V[k, l] = 0.25 + (np.arctan(k_val / l_val) * 180 / np.pi) / 90
            T[k, l] = (Eh / (Eh + Ev)) * H[k, l] + (Ev / (Eh + Ev)) * V[k, l]
    
    return T, H, V

def calculateTKL(dct_coefficients, threshold):
    rows, cols = dct_coefficients.shape
    Th = threshold_dct(dct_coefficients, threshold) 
    Eh, Ev = calculate_Eh_Ev(dct_coefficients)
    
    # Compute Th and T based on their respective functions
    T , H, V = calculate_THV(rows, cols, Eh, Ev)
    Th = threshold_dct(dct_coefficients, threshold)  # Threshold value set to 20, modify if needed
    alpha = compute_alpha(dct_coefficients)
    F_doubleprime = np.copy(dct_coefficients)
    
    # Preserve the DC coefficient (0, 0)
    F_doubleprime[0, 0] = dct_coefficients[0, 0]
    
    # Modify AC coefficients according to equation (11)
    for k in range(rows):
        for l in range(cols):
            if (k, l) != (0, 0):  # Skip the DC coefficient
                F_doubleprime[k, l] = (1 - alpha) * Th[k, l] * T[k, l]
    print("VKL = ", V)
    print("TKL = ", F_doubleprime)
    return F_doubleprime


def calculateTKLC(dct_coefficients, threshold):
    rows, cols = dct_coefficients.shape
    Th = threshold_dct(dct_coefficients, threshold) 
    Eh, Ev = calculate_Eh_Ev(dct_coefficients)
    
    # Compute Th and T based on their respective functions
    T , H, V = calculate_THV(rows, cols, Eh, Ev)
    Th = threshold_dct(dct_coefficients, threshold)  # Threshold value set to 20, modify if needed
    
    alpha = compute_alphaC(dct_coefficients)
    F_doubleprime = np.copy(dct_coefficients)
    
    # Preserve the DC coefficient (0, 0)
    F_doubleprime[0, 0] = dct_coefficients[0, 0]
    
    # Modify AC coefficients according to equation (11)
    for k in range(rows):
        for l in range(cols):
            if (k, l) != (0, 0):  # Skip the DC coefficient
                F_doubleprime[k, l] = (1 - alpha) * Th[k, l] * T[k, l]
    print("VKL = ", V)
    print("TKL = ", F_doubleprime)
    return F_doubleprime

def combine_channels(Y, Cb, Cr):

    ycc_matrix = np.stack((Y, Cb, Cr), axis=-1)

    # Define the transformation matrix
    transformation_matrix = np.array([
    [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    [1 / np.sqrt(6), 0, -1 / np.sqrt(6)],
    [1 / (3 * np.sqrt(2)), -2 / (3 * np.sqrt(2)), 1 / (3 * np.sqrt(2))]
    
    ])
    
    # Inverse of the transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Assuming Y_image, C1_image, C2_image are the matrices we obtained earlier and have the same shape
    # Recombine the Y, C1, and C2 channels into a single matrix
    ycc_combined = np.stack((Y, Cb, Cr), axis=-1)

    # Reshape the combined YC1C2 image to a column vector for matrix multiplication
    ycc_vector = ycc_combined.reshape(-1, 3)

    # Apply the inverse transformation
    rgb_matrix = np.dot(inverse_transformation_matrix, ycc_vector.T).T

    # Reshape the result back to the original image shape
    rgb_reconstructed = rgb_matrix.reshape(Y.shape[0], Y.shape[1], 3)

    # Normalize and convert the values to uint8 if necessary
    rgb_reconstructed_normalized = np.clip(rgb_reconstructed, 0, 255).astype(np.uint8)
    # Convert the NumPy array to an image
    reconstructed_image = Image.fromarray(rgb_reconstructed_normalized)
    # Display the RGB image using matplotlib
    plt.imshow(reconstructed_image)
    plt.axis('off')
    plt.show()
    return rgb_reconstructed_normalized

def save_channels_as_images(Y_image):
    # Convert NumPy arrays to PIL images
    output_folder = 'D:/Users/Pictures/thesis pictures/'
    Y_image_pil = Image.fromarray(Y_image)

    # Save each channel image
    Y_image_pil.save(output_folder + 'denoised_image.png')
    

def blend_dct_coefficients(patch, blend_factor):
    """
    Blend similar frequency components in the given patch using DCT.

    Args:
    - patch: Input patch (2D numpy array).
    - blend_factor: Blending factor for combining DCT coefficients (float).

    Returns:
    - blended_patch_dct: Blended DCT coefficients of the patch (2D numpy array).
    """
    # Apply the blending factor to the DCT coefficients
    blended_patch_dct = patch * blend_factor

    return blended_patch_dct

def denoise_luminance_patch_dct(dct_coefficients, patch_size, threshold, blend_factor):
    """
    Denoise the given image using patch-based luminance processing and DCT-based blending.

    Args:
    - dct_coefficients: DCT coefficients of the input image (2D numpy array).
    - patch_size: Size of the square patch centered around each pixel (integer).
    - threshold: Threshold for determining whether a pixel needs denoising (float).
    - blend_factor: Blending factor for combining DCT coefficients (float).

    Returns:
    - blended_dct: Denoised image DCT coefficients after blending (2D numpy array).
    """
    # Initialize the blended DCT coefficients
    blended_dct = np.zeros_like(dct_coefficients)

    # Iterate over the image pixels
    for i in range(patch_size, dct_coefficients.shape[0] - patch_size):
        for j in range(patch_size, dct_coefficients.shape[1] - patch_size):
            # Extract the patch around the current pixel
            patch = dct_coefficients[i - patch_size:i + patch_size + 1, j - patch_size:j + patch_size + 1]

            # Compute the mean intensity of the patch
            mean_intensity = np.mean(patch)

            # Check if the pixel needs denoising based on the threshold
            if np.abs(dct_coefficients[i, j] - mean_intensity) > threshold:
                # Blend similar frequency components using DCT
                blended_patch_dct = blend_dct_coefficients(patch, blend_factor)
                blended_dct[i - patch_size:i + patch_size + 1, j - patch_size:j + patch_size + 1] = blended_patch_dct
            else:
                # If pixel doesn't need denoising, keep the original DCT coefficients
                blended_dct[i - patch_size:i + patch_size + 1, j - patch_size:j + patch_size + 1] = patch

    return blended_dct


def denoise_image(image_path, threshold, sigma1, sigma2):
    # Load image and convert to YCbCr or any other color space if needed
    image = load_image(image_path)
    y, cb, cr = convert_to_YCbCr(image)
    rgb_image = combine_channels(y, cb, cr)
    grayscale_image = rgb2gray(rgb_image)
    print("Constructed image's matrix", rgb_image)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title("Reconstructed RGB Image" )
    plt.axis('off')
    plt.show()
    # Perform 2D DCT
    
    dct_matrix = compute_2d_dct(y)
    print("Y DCT = ", dct_matrix )
    # dct_coefficients1 = np.array([
    #     [396, 15, -6, 25, -78],
    #     [-21, 17, 32, -14, 43],
    #     [14, -10, 17, 14, -13],
    #     [27, 2, -45, 26, 16],
    #     [-63, 0, 30, 7, 14]
    # ])
    dct_attenuated = calculateTKL(dct_matrix, 25)
    # idct_matrix = compute_2d_idct(dct_attenuated)
    # dct_matrix = compute_2d_dct(idct_matrix)
    # dct_attenuated = calculateTKL(dct_matrix, 0)
    # idct_matrix = compute_2d_idct(dct_attenuated)
    # y = compute_2d_idct(dct_attenuated)
    # dct_matrix = compute_2d_dct(idct_matrix)
    # dct_attenuated = calculateTKL(dct_matrix, 2)
    # print("Sup")
    # # y = compute_2d_dct(idct_matrix )
    # # dct_attenuated = calculateTKL(dct_matrix, 0.5)
    # dct_blended = denoise_luminance_patch_dct(dct_attenuated, 20,10,10)
    # dct_blended = similar_patch_blender(dct_attenuated, 500)
    y = compute_2d_idct(dct_attenuated)

    
    print("Y = ", y)
    plt.imshow(y, cmap='gray')
    plt.title("Y")
    plt.axis('off')
    plt.show()
    
    
    dct_matrix = compute_2d_dct(cb)
    print("C1 DCT = ", dct_matrix)
    dct_attenuated = calculateTKLC(dct_matrix, 100)
    # idct_matrix = compute_2d_idct(dct_attenuated)   
    # dct_matrix = compute_2d_dct(idct_matrix )
    # dct_attenuated = calculateTKLC(dct_matrix, 12)
    # idct_matrix = compute_2d_idct(dct_attenuated)
    # dct_matrix = compute_2d_dct(idct_matrix)
    # dct_attenuated = calculateTKLC(dct_matrix, 2)
    # dct_blended = denoise_luminance_patch_improved(dct_attenuated, 80, 50000)
    cb = compute_2d_idct(dct_attenuated)

    # # print("Cb = ", cb)
    # # plt.imshow(cb)
    # # plt.title("Cb")
    # # plt.axis('off')
    # # plt.show()
    
    dct_matrix = compute_2d_dct(cr)
    print("C2 DCT = ", dct_matrix)
    dct_attenuated = calculateTKLC(dct_matrix, 100)
    # idct_matrix = compute_2d_idct(dct_attenuated)
    # dct_matrix = compute_2d_dct(idct_matrix )
    # dct_attenuated = calculateTKLC(dct_matrix, 5)
    # idct_matrix = compute_2d_idct(dct_attenuated)
    # dct_matrix = compute_2d_dct(idct_matrix)
    # dct_attenuated = calculateTKLC(dct_matrix, 2)
    # dct_blended = denoise_luminance_patch_improved(dct_attenuated, 2, 30000)
    cr = compute_2d_idct(dct_attenuated)

    # # print("Cr = ", cr)
    # # plt.imshow(cr)
    # # plt.title("Cr")
    # # plt.axis('off')
    # # plt.show()
    
    # # delta = calculate_Delta(dct_attenuated)
    # # dct_blended = similar_patch_blender(dct_attenuated, delta, sigma1, sigma2)
    # # idct_matrix = compute_2d_idct(dct_blended)
    # # plt.imshow(idct_matrix)
    # # plt.title("Reconstructed RGB Image")
    # # plt.axis('off')
    # # plt.show()
    
    
    
    # rgb_image = combine_channels(y, cb, cr)
    # grayscale_image = rgb2gray(rgb_image)
    # print("Constructed image's matrix", rgb_image)
    # plt.imshow(grayscale_image, cmap='gray')
    # plt.title("Reconstructed RGB Image" )
    # plt.axis('off')
    # plt.show()
    # # Calculate delta for blending
    # delta = calculate_delta(dct_attenuated)
    
    # # Blend similar patches
    # dct_blended = similar_patch_blender(dct_attenuated, delta, sigma1, sigma2)
    
    # # Perform 2D iDCT
    # idct_matrix = compute_2d_idct(dct_blended)
    
    # # Post-process if needed and convert back to RGB  
    # denoised_image = post_process(idct_matrix)
    rgb_image = combine_channels(y, cb, cr)       
    save_channels_as_images(rgb_image)
    return rgb_image



# Use the function
denoised_image = denoise_image('D:/Users/Pictures/thesis pictures/meowsy.png', threshold=500, sigma1=10, sigma2=56)

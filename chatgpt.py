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
    rgb_image = np.array(image)
    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    Y = np.clip(Y, 0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)

    return Y, Cb, Cr

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

def threshold_dct(dct_coefficients, threshold):
    thresholded_dct = np.copy(dct_coefficients)
    for k in range(dct_coefficients.shape[0]):
        for l in range(dct_coefficients.shape[1]):
            if k == 0 and l == 0:
                continue
            else:
                if dct_coefficients[k, l] > threshold:
                    thresholded_dct[k, l] -= threshold
                elif dct_coefficients[k, l] < -threshold:
                    thresholded_dct[k, l] += threshold
                elif np.abs(dct_coefficients[k, l]) <= threshold:
                    thresholded_dct[k, l] = 0
    return thresholded_dct

def compute_alpha(dct_coefficients):
    N = len(dct_coefficients)
    F_00 = dct_coefficients[0, 0]
    sum_of_absolute_values = np.sum(np.abs(dct_coefficients))
    E = (sum_of_absolute_values - np.abs(F_00)) / ((N * N) - 1)
    En = E * -0.0052 + 1
    Alpha = (1 - 0.0052 * E) * En
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
    
    return F_doubleprime

# Function to calculate ΔDC(x, y, i, j) for all i, j given a DCT coefficient matrix
def calculate_DeltaDC(F_xy, N):
    # Initialize matrix for ΔDC
    DeltaDC = np.zeros((N, N))

    # DC component is at the (0, 0) position, subtract it from all positions
    for i in range(N):
        for j in range(N):
            DeltaDC[i, j] = F_xy[0, 0] - F_xy[i, j]

    return DeltaDC

# Function to calculate ΔAC(x, y, i, j) for all i, j given a DCT coefficient matrix
def calculate_DeltaAC(F_xy, N):
    # Initialize matrix for ΔAC
    DeltaAC = np.zeros((N, N))

    # Accumulate differences from all AC coefficients
    for i in range(N):
        for j in range(N):
            if (i, j) != (0, 0):  # Skip the DC component
                DeltaAC[i, j] = np.sum(np.abs(F_xy - F_xy[i, j]))

    # Subtract ΔDC(x, y, i, j) from ΔAC(x, y, i, j) as per the equation
    DeltaDC = calculate_DeltaDC(F_xy, N)
    DeltaAC -= DeltaDC

    return DeltaAC

# Function to calculate Δ(x, y, i, j) for all i, j given a DCT coefficient matrix
def calculate_Delta(F_xy):
    N = 100
    # Calculate ΔDC and ΔAC
    DeltaDC = calculate_DeltaDC(F_xy, N)
    DeltaAC = calculate_DeltaAC(F_xy, N)

    # Calculate Δ using the equation
    Delta = (1 / (N**2)) * (DeltaDC + DeltaAC)

    return Delta


def similar_patch_blender(dct_matrix, delta, sigma1, sigma2):
    M, N = dct_matrix.shape  # Assuming a square DCT block for simplicity
    blended_dct_matrix = np.zeros_like(dct_matrix)

    # Adjust the range if M is even, otherwise adjust as needed for odd M
    i_range = range(-M//2 + 1, M//2) if M % 2 == 0 else range(-M//2, M//2 + 1)
    j_range = range(-N//2 + 1, N//2) if N % 2 == 0 else range(-N//2, N//2 + 1)

    # Iterating over each element in the DCT matrix
    for k in range(M):
        for l in range(N):
            numerator_sum = 0.0
            denominator_sum = 0.0
            
            # Iterating over the window centered around the current element
            for i in i_range:
                for j in j_range:
                    # Ensuring periodic boundary conditions
                    periodic_i = (i + k) % M
                    periodic_j = (j + l) % N
                    
                    # Calculating the spatial weight
                    weight_spatial = np.exp(-((i**2 + j**2) / (2 * sigma1**2)))
                    # Calculating the delta weight
                    weight_delta = np.exp(-((delta[periodic_i, periodic_j])**2 / (2 * sigma2**2)))
                    
                    # Adding to the numerator and denominator sums
                    numerator_sum += dct_matrix[periodic_i, periodic_j] * weight_spatial * weight_delta
                    denominator_sum += weight_spatial * weight_delta
            
            # Computing the blended DCT coefficient
            blended_dct_matrix[k, l] = numerator_sum / denominator_sum if denominator_sum != 0 else 0
    
    return blended_dct_matrix

def combine_channels(y_channel, cb_channel, cr_channel):
    # Apply any necessary post-processing or adjustments to the channels
    # For example, you might want to clip the channels to the valid range [0, 255]
    y_channel = np.clip(y_channel, 0, 255).astype(np.uint8)
    cb_channel = np.clip(cb_channel, 0, 255).astype(np.uint8)
    cr_channel = np.clip(cr_channel, 0, 255).astype(np.uint8)

    # Combine the channels (you can experiment with different blending techniques)
    combined_image = np.stack((y_channel, cb_channel, cr_channel), axis=-1)

    # Convert the combined image back to RGB
    rgb_combined = Image.fromarray(combined_image, mode='YCbCr').convert('RGB')

    return rgb_combined


def denoise_image(image_path, threshold, sigma1, sigma2):
    # Load image and convert to YCbCr or any other color space if needed
    image = load_image(image_path)
    y, cb, cr = convert_to_YCbCr(image)
    rgb_image = combine_channels(y, cb, cr)
    print(rgb_image)
    plt.imshow(rgb_image)
    plt.title("Reconstructed RGB Image")
    plt.axis('off')
    plt.show()
    # plt.imshow(y)
    # plt.title("Reconstructed RGB Image")
    # plt.axis('off')
    # plt.show()
    # plt.imshow(cb)
    # plt.title("Reconstructed RGB Image")
    # plt.axis('off')
    # plt.show()
    # plt.imshow(cr)
    # plt.title("Reconstructed RGB Image")
    # plt.axis('off')
    # plt.show()
    # rgb_image = convert_to_RGB(y, cb, cr)
    # # Apply filters to the Y channel if needed
    # y_filtered = apply_filters(y)
    
    # Perform 2D DCT
    dct_matrix = compute_2d_dct(y)
    dct_attenuated = calculateTKL(dct_matrix, 4)
    idct_matrix = compute_2d_idct(dct_attenuated)
    
    dct_matrix = compute_2d_dct(idct_matrix )
    dct_attenuated = calculateTKL(dct_matrix, 2)
    idct_matrix = compute_2d_idct(dct_attenuated)
    
    dct_matrix = compute_2d_dct(idct_matrix )
    dct_attenuated = calculateTKL(dct_matrix, 1)
    y= compute_2d_idct(dct_attenuated)
    plt.imshow(y)
    plt.title("Reconstructed RGB Image")
    plt.axis('off')
    plt.show()
    
    
    
    dct_matrix = compute_2d_dct(cb)
    dct_attenuated = calculateTKL(dct_matrix, 4)
    idct_matrix = compute_2d_idct(dct_attenuated)
    
    dct_matrix = compute_2d_dct(idct_matrix )
    dct_attenuated = calculateTKL(dct_matrix, 2)
    idct_matrix = compute_2d_idct(dct_attenuated)
    
    dct_matrix = compute_2d_dct(idct_matrix )
    dct_attenuated = calculateTKL(dct_matrix, 1)
    cb= compute_2d_idct(dct_attenuated)
    plt.imshow(cb)
    plt.title("Reconstructed RGB Image")
    plt.axis('off')
    plt.show()
    
    dct_matrix = compute_2d_dct(cr)
    dct_attenuated = calculateTKL(dct_matrix, 4)
    idct_matrix = compute_2d_idct(dct_attenuated)
    
    dct_matrix = compute_2d_dct(idct_matrix )
    dct_attenuated = calculateTKL(dct_matrix, 2)
    idct_matrix = compute_2d_idct(dct_attenuated)
    
    dct_matrix = compute_2d_dct(idct_matrix )
    dct_attenuated = calculateTKL(dct_matrix, 1)
    cr= compute_2d_idct(dct_attenuated)
    plt.imshow(cr)
    plt.title("Reconstructed RGB Image")
    plt.axis('off')
    plt.show()
    
    # delta = calculate_Delta(dct_attenuated)
    # dct_blended = similar_patch_blender(dct_attenuated, delta, sigma1, sigma2)
    # idct_matrix = compute_2d_idct(dct_blended)
    # plt.imshow(idct_matrix)
    # plt.title("Reconstructed RGB Image")
    # plt.axis('off')
    # plt.show()
    
    
    
    rgb_image = combine_channels(y, cb, cr)
    print(rgb_image)
    plt.imshow(rgb_image)
    plt.title("Reconstructed RGB Image")
    plt.axis('off')
    plt.show()
    # # Calculate delta for blending
    # delta = calculate_delta(dct_attenuated)
    
    # # Blend similar patches
    # dct_blended = similar_patch_blender(dct_attenuated, delta, sigma1, sigma2)
    
    # # Perform 2D iDCT
    # idct_matrix = compute_2d_idct(dct_blended)
    
    # # # Post-process if needed and convert back to RGB
    # # denoised_image = post_process(idct_matrix)
    # rgb_image = convert_to_RGB(idct_matrix, cb, cr)   
    
    return rgb_image



# Use the function
denoised_image = denoise_image('D:/Users/Pictures/thesis pictures/small.png', threshold=500, sigma1=20, sigma2=70)

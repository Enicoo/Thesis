import numpy as np
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
    print("PRINTING:", image_matrix.T)
    print("1D DCT:", dct(image_matrix.T, norm='ortho'))
    print("1D DCT_T:", dct(image_matrix.T, norm='ortho').T)
    
    return dct(dct(image_matrix.T, norm='ortho').T, norm='ortho')

def compute_2d_idct(dct_matrix):
    idct1 = idct(idct(dct_matrix.T, norm='ortho').T, norm='ortho')
    print("matrix", dct_matrix.T,)
    print("inverse2", idct(dct_matrix.T, norm='ortho'))
    return idct1

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
    
    # print("Thresholded DCT = ", np.array2string(thresholded_dct, separator=', '))
    return thresholded_dct

def compute_alpha(dct_coefficients):
    N = len(dct_coefficients)
    F_00 = dct_coefficients[0, 0]
    print("F_00 = ", F_00)
    sum_of_absolute_values = np.sum(np.abs(dct_coefficients))
    print("sun = ", sum_of_absolute_values)
    E = (sum_of_absolute_values - np.abs(F_00)) / ((N * N) - 1)
    En = E* -5+ 1
    Alpha = (1 - 0.0052 * E) * En
    print("En = ", En)
    print("E = ", E)
    print("Alpha = ", Alpha)
    return Alpha

def compute_alphaC(dct_coefficients):
    N = len(dct_coefficients)
    F_00 = dct_coefficients[0, 0]
    print("F_00 = ", F_00)
    sum_of_absolute_values = np.sum(np.abs(dct_coefficients))
    print("sun = ", sum_of_absolute_values)
    E = (sum_of_absolute_values - np.abs(F_00)) / ((N * N) - 1)
    En = E * -5 + 1
    Alpha = (1 - 0.0082 * E) * En
    print("SIZE:", N)
    # print("En = ", En)
    # print("E = ", E)
    print("En = ", En)
    print("E = ", E)
    print("Alpha = ", Alpha)
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
    # print("Eh = ", Eh)
    # print("Ev = ", Ev)
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
    # print("T = " , T)
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
    # print("VKL = ", V)
    # print("TKL = ", F_doubleprime)
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
    # print("VKL = ", V)
    # print("TKLc = ", F_doubleprime)
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
    print("RGB = " , rgb_reconstructed_normalized)
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
    Y_image_pil.save(output_folder + 'denoised_image_small.png')

def saveY(Y_image):
    # Convert NumPy arrays to PIL images
    output_folder = 'D:/Users/Pictures/thesis pictures/'
    Y_image_pil = Image.fromarray(Y_image)

    # Save each channel image
    Y_image_pil.save(output_folder + 'dy.png')
    
def saveC1(Y_image):
    # Convert NumPy arrays to PIL images
    output_folder = 'D:/Users/Pictures/thesis pictures/'
    Y_image_pil = Image.fromarray(Y_image)

    # Save each channel image
    Y_image_pil.save(output_folder + 'dc1.png')
    
def saveC2(Y_image):
    # Convert NumPy arrays to PIL images
    output_folder = 'D:/Users/Pictures/thesis pictures/'
    Y_image_pil = Image.fromarray(Y_image)

    # Save each channel image
    Y_image_pil.save(output_folder + 'dc2.png')

def normalized(channel):
    normalized = ((channel - np.min(channel)) / (np.max(channel) - np.min(channel))) * 255
    normal = normalized.astype(np.uint8)
    return normal
    
def denoise_luminance_patch_improved(image, patch_size, threshold):
    denoised_image = np.copy(image)

    for i in range(patch_size, image.shape[0] - patch_size):
        for j in range(patch_size, image.shape[1] - patch_size):
            patch = image[i - patch_size:i + patch_size + 1, j - patch_size:j + patch_size + 1]
            mean_intensity = np.mean(patch)

            condition = np.abs(image[i, j] - mean_intensity) > threshold
            denoised_image[i, j] = np.where(condition, mean_intensity, denoised_image[i, j])

    return denoised_image

def similar_patch_blender(dct_image, similarity_threshold):
    rows, cols = dct_image.shape
    if dct_image.dtype not in [np.float32, np.float64]:
        raise ValueError("dct_image should be of type float32 or float64")

    # Identify similar frequency components and group them
    similar_frequencies = []
    grouped_indices = []
    for i in range(rows):
        for j in range(cols):
            current_coeff = dct_image[i, j]
            if not any((i, j) in group for group in grouped_indices):
                similar_group = [(i, j)]
                for k in range(i, rows):
                    for l in range(cols):
                        if (k, l) not in similar_group:
                            compare_coeff = dct_image[k, l]
                            similarity = np.abs(current_coeff - compare_coeff)
                            if similarity < similarity_threshold:
                                similar_group.append((k, l))
                similar_frequencies.append(similar_group)
                grouped_indices.append(similar_group)
    
    # Blending similar frequency components
    for group in similar_frequencies:
        group_values = [dct_image[i, j] for i, j in group]
        avg_value = np.mean(group_values)
        for i, j in group:
            dct_image[i, j] = avg_value
    
    print("Blended = ",  dct_image)
    return dct_image

def denoise_image(image_path, threshold, sigma1, sigma2):
    # Load image and convert to YCbCr or any other color space if needed
    image = load_image(image_path)
    y, cb, cr = convert_to_YCbCr(image)
    rgb_image = combine_channels(y, cb, cr)
    grayscale_image = rgb2gray(rgb_image)
    # print("Constructed image's matrix", rgb_image)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title("Reconstructed RGB Image" )
    plt.axis('off')
    plt.show()
    # Perform 2D DCT
    
    # Perform denoising in Luminance Channel Y
    dct_matrix = compute_2d_dct(y)
    print("YPRIME = ", dct_matrix)
    dct_attenuated = calculateTKL(dct_matrix, 50)
    print("DCT ATT", dct_attenuated)
    y = compute_2d_idct(dct_attenuated)
    print("Ydct = ", y)
    plt.imshow(y, cmap='gray')
    plt.title("Y")
    plt.axis('off')
    plt.show()
    ynormal = normalized(y)
    saveY(ynormal)
    
   
    
    # # Perform denoising in Chrominance1 Channel c1
    dct_matrix = compute_2d_dct(cb)
    print("CBPRIME = ", dct_matrix)
    dct_attenuated1 = calculateTKLC(dct_matrix, 35)
    cb = compute_2d_idct(dct_attenuated1)
    cb2normal = normalized(cb)
    saveC2(cb2normal)
    print("Cbdct = ", cb)
    plt.imshow(cb)
    plt.title("Cb")
    plt.axis('off')
    plt.show()
    
    
    # Perform denoising in Chrominance2 Channel c2
    dct_matrix = compute_2d_dct(cr)
    print("CRPRIME = ", dct_matrix)
    dct_attenuated2 = calculateTKLC(dct_matrix, 25)
    cr = compute_2d_idct(dct_attenuated2)
    crnormal = normalized(cr)
    saveC2(crnormal)
    print("Crdct = ", cr)  
    plt.imshow(cr)
    plt.title("Cr")
    plt.axis('off')
    plt.show()
    
    
    # To see final output
    rgb_image = combine_channels(y, cb, cr)       
    save_channels_as_images(rgb_image)
    return rgb_image



# Use the function
denoised_image = denoise_image('D:/Users/Pictures/thesis pictures/65.png', threshold=500, sigma1=10, sigma2=56)

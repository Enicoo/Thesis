import numpy as np
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
from PIL import Image

def compute_dct_coefficient_fast(f, k, l, N, phi_cache):
    F_kl = 0
    for m in range(N):
        for n in range(N):
            F_kl += f[m, n] * phi_cache[k][m] * phi_cache[l][n]
    return F_kl

def compute_dct_matrix_fast(f):
    N = f.shape[0]
    F = np.zeros((N, N))
    phi_cache = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            if k == 0:
                phi_cache[k][n] = 1 / np.sqrt(2)
            else:
                phi_cache[k][n] = np.sqrt(2 / N) * np.cos((2 * n + 1) * k * np.pi / (2 * N))

    for k in range(N):
        for l in range(N):
            F[k, l] = compute_dct_coefficient_fast(f, k, l, N, phi_cache)
            if k == 0 and l == 0:
                F[k, l] *= 1 / np.sqrt(2)

    return F * (2 / N)

# Replace 'image_path' with the actual path to your image file when running this code locally.
image_path = 'D:/Users/Pictures/thesis pictures/Y_channel.png'  # Replace with the path to your image
image = Image.open(image_path).convert('L')  # Convert image to grayscale
image_matrix = np.array(image)

# Compute the DCT for the entire image
dct_matrix = compute_dct_matrix_fast(image_matrix)
integer_array = dct_matrix.astype(int)
print(integer_array)

# Perform the inverse DCT (iDCT)
inverse_dct_image = idct(idct(integer_array.T, norm='ortho').T, norm='ortho')
# Normalize pixel values to the range [0, 255] for visualization
reconstructed_image = np.clip(inverse_dct_image, 0, 255).astype(np.uint8)

# Display the original image
plt.imshow(image_matrix, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Display the reconstructed image using Matplotlib
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image from DCT Coefficients')
plt.axis('off')
plt.show()

# Display the DCT coefficients as an image
plt.imshow(dct_matrix, cmap='gray')
plt.title('DCT Coefficients')
plt.colorbar()
plt.show()

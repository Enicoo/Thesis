import numpy as np
import FAST2DCT as fast
from scipy.ndimage import gaussian_filter


def alpha(E): 
    En = 50
    a = 1 - (En * E)
    return a 

def threshold_dct(dct_coefficients, threshold):
    # dct_coefficients is a 2D numpy array of DCT coefficients
    # threshold is the value of Th

    # Create a copy to avoid modifying the original coefficients
    thresholded_dct = np.copy(dct_coefficients)

    # Apply the thresholding as per the given equations
    # DC component is not thresholded
    for k in range(dct_coefficients.shape[0]):
        for l in range(dct_coefficients.shape[1]):
            if k == 0 and l == 0:
                continue  # Skip the DC component
            else:
                if dct_coefficients[k, l] > threshold:
                    thresholded_dct[k, l] -= threshold
                elif dct_coefficients[k, l] < -threshold:
                    thresholded_dct[k, l] += threshold
                else:
                    thresholded_dct[k, l] = 0

    return thresholded_dct

def compute_E(dct_coefficients):
    N = len(dct_coefficients)  # Assuming a square DCT coefficient matrix
    F_00 = dct_coefficients[0, 0]  # DC component
    sum_of_absolute_values = np.abs(np.sum(dct_coefficients)) - np.abs(F_00)  # Sum of absolute values excluding DC
    E = sum_of_absolute_values / ((N * N) - 1)  # Compute E
    blurred_image = gaussian_filter(dct_coefficients, sigma=2)
    noise = dct_coefficients - blurred_image
    En = np.std(noise)
    print(En)
    
    return E

def calculate_Eh_Ev(dct_coefficients):
    rows, cols = dct_coefficients.shape  # Adjusting for non-square matrices
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


def calculateTKL(dct_coefficients):
    rows, cols = dct_coefficients.shape
    E = compute_E(dct_coefficients)
    Eh, Ev = calculate_Eh_Ev(dct_coefficients)
    N = 100  # Size of the DCT coefficients matrix (assuming square matrix)
    Th = 20 # Example threshold value
    th= threshold_dct(dct_coefficients, Th)
    T, H, V = calculate_THV(rows, cols, Eh, Ev)
    F_doubleprime = (1 - 0.5) * T * th
    
    return F_doubleprime

def compute_delta(F_prime, N):
    # Initialize delta matrices
    delta_DC = F_prime[0, 0] - F_prime
    delta_AC = np.zeros_like(F_prime)
    
    # Calculate delta_AC
    for y in range(N):
        for x in range(N):
            ac_sum = 0
            for k in range(N):
                for l in range(N):
                    if k == 0 and l == 0:
                        continue  # Skip the DC component
                    ac_sum += F_prime[k, l] - F_prime[(y-k) % N, (x-l) % N]
            delta_AC[y, x] = ac_sum
    delta = (delta_DC + delta_AC) / (N**2)
    
    return delta

def blend_frequency_components(F_prime, delta, N, sigma1, sigma2):
    M = F_prime.shape[0]  # Assuming F_prime is a square matrix
    F_double_prime = np.zeros_like(F_prime)

    # Adjust the range to ensure we stay within bounds of the array
    for i in range(-M//2, M//2):
        for j in range(-M//2, M//2):
            weighted_sum = 0
            weight_sum = 0
            for x in range(N):
                for y in range(N):
                    # Ensure the index calculations stay within the bounds of the array
                    i_index = (i + M//2) % M
                    j_index = (j + M//2) % M
                    weight = np.exp(-(i**2 + j**2) / (2 * sigma1**2)) * np.exp(-delta[y, x]**2 / (2 * sigma2**2))
                    weighted_sum += weight * F_prime[y, x]
                    weight_sum += weight
            # Use modulo to wrap around the index and stay within bounds
            F_double_prime[i_index, j_index] = weighted_sum / weight_sum if weight_sum != 0 else 0

    return F_double_prime

image_path = 'D:/Users/Pictures/thesis pictures/Y_channel.png'
dct_coefficients = fast.display_dct(image_path)
# Calculate E using the function\
E = compute_E(dct_coefficients)
print(E)    
print(dct_coefficients)
# N = 100
# sigma1 = 2
# sigma2 = 30 

# result1 = calculateTKL(dct_coefficients)
# new = fast.display_idct(result1)
# new1 = fast.display_dct2(new)
# result2 = calculateTKL(new1)
# new3 = fast.display_idct(result2)
# new4 = fast.display_dct2(new3)
# result3 = calculateTKL(new4)
# delta = compute_delta(result3, N)
# Final = blend_frequency_components(result3, delta, N, sigma1, sigma2)

# Last = fast.display_idct(Final)

# Calculate T, H, and V for a given N (size of DCT coefficients matrix)


# Given the 2D DCT coefficients matrix
# dct_coefficients = np.array([
#     [396, 15, -6, 25, -78],
#     [-21, 17, 32, -14, 43],
#     [14, -10, 17, 14, -13],
#     [27, 2, -45, 26, 16],
#     [-63, 0, 30, 7, 14]
# ])
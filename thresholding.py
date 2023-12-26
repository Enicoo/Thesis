import numpy as np

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

dct_matrix = np.array([
    [396, 15, -6, 25, -78],
    [-21, 17, 32, -14, 43],
    [14, -10, 17, 14, -13],
    [27, 2, -45, 26, 16],
    [-63, 0, 30, 7, 14]
])
# Example usage:
# dct_matrix is a 2D numpy array containing your DCT coefficients
# Th is the threshold value you want to apply
Th = 20 # Example threshold value
thresholded_dct_matrix = threshold_dct(dct_matrix, Th)

print(thresholded_dct_matrix)
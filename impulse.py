import numpy as np

# Generate a 5x5 matrix (you can replace this with your desired matrix)
original_matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

# Normalizing the matrix
original_matrix /= np.max(original_matrix)

# Blank matrix to store noisy image
x, y = original_matrix.shape
noisy_matrix = np.zeros((x, y), dtype=np.float32)

# Salt and pepper noise parameters
pepper = 0.1
salt = 0.95

# Add salt and pepper noise to the matrix
for i in range(x):
    for j in range(y):
        rdn = np.random.random()
        if rdn < pepper:
            noisy_matrix[i][j] = 0
        elif rdn > salt:
            noisy_matrix[i][j] = 1
        else:
            noisy_matrix[i][j] = original_matrix[i][j]

print("Original Matrix:")
print(original_matrix)
print("\nMatrix with Salt-and-Pepper Noise:")
print(noisy_matrix)

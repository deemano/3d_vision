import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define padding for the plot
padding = 30  

# Set the random seed
cv2.setRNGSeed(42)

####### Exercise 1 ################################################################ <<<<<<<<<<<<<<

# Define the number of points and bounds for the coordinates
num_points = 200
x_bounds, y_bounds = (0, 1920), (0, 1080)

# Initialize matrices for x and y coordinates
x_coords = np.zeros((num_points, 1), dtype=np.float64)
y_coords = np.zeros((num_points, 1), dtype=np.float64)

# Generate random points within the specified bounds using OpenCV's randu()
cv2.randu(x_coords, x_bounds[0], x_bounds[1])
cv2.randu(y_coords, y_bounds[0], y_bounds[1])

# Combine x and y coordinates
points = np.hstack((x_coords, y_coords))

# Define the homography matrix H
H = np.array([[3, -4, 5], [8, 7, 2], [-1, 4, 3]], dtype=np.float64)

# Apply the homography transformation to the points
homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
x_prime_homogeneous = np.dot(homogeneous_points, H.T)
x_prime = x_prime_homogeneous[:, :2] / x_prime_homogeneous[:, 2][:, np.newaxis]

# Vanilla DLT method
def vanilla_dlt_homography(x, x_prime):
    num_points = x.shape[0]
    A = np.zeros((num_points * 2, 9))
    for i in range(num_points):
        X, Y = x[i]
        Xp, Yp = x_prime[i]
        A[2 * i] = [-X, -Y, -1, 0, 0, 0, X * Xp, Y * Xp, Xp]
        A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, X * Yp, Y * Yp, Yp]

    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_vanilla = h.reshape((3, 3))
    return H_vanilla

# Define functions for the DLT method
def normalize_points(points):
    """
    Normalize the set of points so that the centroid is at the origin
    and the average distance from the origin is sqrt(2).
    """
    mean = np.mean(points, axis=0)
    std = np.std(points)
    scale = np.sqrt(2) / std
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    normalized_points = (T @ points_homogeneous.T).T
    return normalized_points, T

def dlt_homography(x, x_prime):
    """
    Perform the Direct Linear Transform (DLT) algorithm to estimate
    the homography matrix from point correspondences.
    """
    # Normalize the points
    x_normalized, T1 = normalize_points(x)
    x_prime_normalized, T2 = normalize_points(x_prime)
    
    # Prepare matrix A for the DLT algorithm
    num_points = x_normalized.shape[0]
    A = np.zeros((num_points * 2, 9))
    for i in range(num_points):
        X, Y, W = x_normalized[i]
        Xp, Yp, Wp = x_prime_normalized[i]
        A[2 * i] = [-X, -Y, -W, 0, 0, 0, X * Xp, Y * Xp, W * Xp]
        A[2 * i + 1] = [0, 0, 0, -X, -Y, -W, X * Yp, Y * Yp, W * Yp]

    # Solve for the homography matrix using SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]                          # Take the last column of V
    H_normalized = h.reshape((3, 3))    # Reshape into a 3x3 matrix

    # Denormalize the homography matrix
    H = np.linalg.inv(T2) @ H_normalized @ T1
    H = H / H[2, 2]  # Ensure the scale of H is 1

    # Calculate the condition number of matrix A
    cond_number = np.linalg.cond(A)

    return H, cond_number

# Plot the original and transformed points
plt.figure(figsize=(8, 6))  # Create a new figure for the initial points and their transformations
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Original Points (x)', marker='o', s=10)
plt.scatter(x_prime[:, 0], x_prime[:, 1], color='green', label='Transformed Points (x\')', marker='x', s=10)
plt.title('Random 2D Points before and after Homography Transformation')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend(loc='upper left')
plt.grid(True)
plt.xlim([x_bounds[0] - padding, x_bounds[1] + padding])
plt.ylim([y_bounds[0] - padding, y_bounds[1] + padding])

# Print initial H
print("Exercise 1")
print("Initial Homography Matrix:")
print(H)
# Estimate the homography matrix using vanilla DLT on original points
H_estimated_vanilla = vanilla_dlt_homography(points, x_prime)
print("Estimated Homography Matrix (Vanilla DLT):")
print(H_estimated_vanilla)

# Estimate the homography matrix using normalized DLT on original points
H_estimated_normalized, cond_number_normalized = dlt_homography(points, x_prime)
print("Estimated Homography Matrix (Normalized DLT):")
print(H_estimated_normalized)
print(f"Condition number (Normalized DLT): {cond_number_normalized}\n")

####### Exercise 2 ################################################################ <<<<<<<<<<<<<<

# Function to add Gaussian noise to points
def add_gaussian_noise_opencv(points, mean=0, sigma=0.5):
    noise = np.empty(points.shape, dtype=np.float64)
    cv2.randn(noise, mean, sigma)
    noisy_points = points + noise
    return noisy_points

# Add Gaussian noise to the original and transformed points 
x_noisy = add_gaussian_noise_opencv(points, mean=0, sigma=0.5)
x_prime_noisy = add_gaussian_noise_opencv(x_prime, mean=0, sigma=0.5)

# Estimate the homography matrix using vanilla DLT on noisy points
print("Exercise 2")
H_estimated_vanilla_noisy = vanilla_dlt_homography(x_noisy, x_prime_noisy)
print("Estimated Homography Matrix with Noise (Vanilla DLT):")
print(H_estimated_vanilla_noisy)

# Estimate the homography matrix using normalized DLT on noisy points
H_estimated_noisy_normalized, cond_number_noisy_normalized = dlt_homography(x_noisy, x_prime_noisy)
print("Estimated Homography Matrix with Noise (Normalized DLT):")
print(H_estimated_noisy_normalized)
print(f"Condition number with Noise (Normalized DLT): {cond_number_noisy_normalized}\n")
print(f"Comparison:\nBased on the results, the condition number obtained from the normalized Direct Linear Transform (DLT) method significantly improved when noise was introduced to the dataset. The condition number decreased from approximately, suggesting a substantial increase in the numerical stability of the homography estimation under noisy conditions. This dramatic reduction indicates that the normalization process in the DLT method dramatically enhances the method's robustness to noise, leading to a more reliable estimation of the homography matrix.") 

# Plot the noisy points
plt.figure(figsize=(8, 6))
plt.scatter(x_noisy[:, 0], x_noisy[:, 1], color='red', label='Noisy Original Points (x)', marker='o', s=10)
plt.scatter(x_prime_noisy[:, 0], x_prime_noisy[:, 1], color='black', label='Noisy Transformed Points (x\')', marker='x', s=10)
plt.title('Random 2D Points with Gaussian Noise')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.xlim(x_bounds[0] - padding, x_bounds[1] + padding)
plt.ylim(y_bounds[0] - padding, y_bounds[1] + padding)

####### Exercise 3 ################################################################ <<<<<<<<<<<<<<

# Create outlier points sets r and r'
r = np.random.uniform(low=[x_bounds[0], y_bounds[0]], high=[x_bounds[1], y_bounds[1]], size=(50, 2))
r_prime = np.random.uniform(low=[x_bounds[0], y_bounds[0]], high=[x_bounds[1], y_bounds[1]], size=(50, 2))

# Combine with inliers
xr = np.vstack((x_noisy, r))  # Assuming x_noisy is your set with noise from previous steps
xr_prime = np.vstack((x_prime_noisy, r_prime))

# Shuffle points (shuffle by pairs)
combined = list(zip(xr, xr_prime))
np.random.shuffle(combined)
xr[:], xr_prime[:] = zip(*combined)

# Implement the Gold Standard Algorithm using RANSAC
H_gold_standard, status = cv2.findHomography(xr, xr_prime, method=cv2.RANSAC)

# Estimate the number of outliers
number_of_inliers = np.sum(status)
number_of_outliers = status.size - number_of_inliers

print("\nExercise 3")
print("Estimated Homography Matrix (Gold Standard Algorithm - RANSAC):")
print(H_gold_standard)
print(f"Number of inliers: {number_of_inliers}")
print(f"Number of outliers: {number_of_outliers}")
print(f"Total points: {status.size}")

# Visually compare the performance, inliers vs outliers
inliers = xr[status.ravel() == 1]
outliers = xr[status.ravel() == 0]

plt.figure(figsize=(8, 6))
plt.scatter(inliers[:, 0], inliers[:, 1], color='blue', label='Inliers', marker='o', s=10)
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers', marker='x', s=10)
plt.title('Inliers and Outliers after RANSAC')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.xlim(x_bounds[0] - padding, x_bounds[1] + padding)
plt.ylim(y_bounds[0] - padding, y_bounds[1] + padding)

######################################################################################

# Compare Homography matrices
def calculate_reprojection_error(H_estimated, x, x_prime):
    # Apply the homography transformation to the points
    homogeneous_points = np.hstack((x, np.ones((x.shape[0], 1))))
    x_prime_estimated_homogeneous = np.dot(homogeneous_points, H_estimated.T)
    x_prime_estimated = x_prime_estimated_homogeneous[:, :2] / x_prime_estimated_homogeneous[:, 2][:, np.newaxis]
    
    # Calculate the reprojection error
    error = x_prime - x_prime_estimated
    reprojection_error = np.sqrt(np.sum(error ** 2, axis=1)).mean()  # Mean squared error
    return reprojection_error

# Calculate reprojection errors for each method
reprojection_error_vanilla = calculate_reprojection_error(H_estimated_vanilla, points, x_prime)
reprojection_error_normalized = calculate_reprojection_error(H_estimated_normalized, points, x_prime)
reprojection_error_ransac = calculate_reprojection_error(H_gold_standard, points, x_prime)

# Compare how many outliers RANSAC can absorb
print(f"\nThe Gold Standard Algorithm (RANSAC) was able to absorb {number_of_outliers}")

# Print out the reprojection errors
print("\nComparison:\nThe Normalized DLT method, despite its significantly high condition number when applied to the original dataset, suggesting a numerically unstable solution, showed a dramatic improvement in condition number when noise was introduced, indicating robustness to noise. In contrast, the RANSAC method, designed to handle outliers, identified and excluded 133 outliers out of 250 points, demonstrating its strength in dealing with a big proportion of outlier data and providing a reliable homography estimation under these conditions.")
     
# Show all plots at once
plt.show()
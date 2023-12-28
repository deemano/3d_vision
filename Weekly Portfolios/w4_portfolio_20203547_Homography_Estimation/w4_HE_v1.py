import numpy as np

def generate_2d_observations(num_observations=210, num_3D_points=24):
    # Known rotation matrix (Identity for simplicity)
    R_known = np.eye(3)
    # Known translation vector, will vary for each observation to simulate different poses
    t_known_base = np.array([1, 2, 3])
    # Camera intrinsic matrix
    K = np.array([[500, 0, 320],
                  [0, 500, 240],
                  [0, 0, 1]])

    # Generate 12 reference 3D points
    points_3D = np.random.rand(num_3D_points, 3)

    # Generate 210 observations of 2D points, each from a different pose
    points_2D_observations = np.zeros((num_observations, num_3D_points * 2))
    for i in range(num_observations):
        # Randomly perturb the translation vector to simulate a different pose
        t_known = t_known_base + np.random.rand(3) * 0.1  # Small random perturbation
        
        # Project 3D points to 2D for each pose
        points_2D_hom = np.array([K @ (R_known @ X + t_known) for X in points_3D])
        # Normalize the 2D points to convert from homogeneous to Cartesian coordinates
        points_2D = points_2D_hom[:, :2] / points_2D_hom[:, 2, np.newaxis]
        
        # Flatten the 2D points to match the exercise's format
        points_2D_observations[i, :] = points_2D.flatten()

    return points_2D_observations

# Generate the 210 rows x 12 columns 2D points
points_2D_observations_210x12 = generate_2d_observations()

# Function to print the 2D points with each set of 12 points on one line, separated by commas
def print_2d_points_comma_separated(points_2D, num_points_per_set=24):
    for row in points_2D:
        # Separate each set of 12 points with a comma
        formatted_row = ' '.join(
            f"{row[i]:.2f}" for i in range(num_points_per_set)
        ) + ',' + ' '.join(
            f"{row[i]:.2f}" for i in range(num_points_per_set, len(row))
        )
        print(formatted_row)

# Print the first 5 rows of the 2D points, formatted with commas after each set of 12
print("Formatted 2D Points Observations (first 5 rows):")
print_2d_points_comma_separated(points_2D_observations_210x12[:210])
#print( np.random.rand(num_3D_points, 3))

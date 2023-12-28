import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

# Load your data
points = np.loadtxt('./resources/pnp/points_3d_noDelimiters.txt')
camera_positions = np.loadtxt('./resources/pnp/translations.txt')
camera_orientations = np.loadtxt('./resources/pnp/rotations.txt')

# Prepare the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot function
def plot_camera_view(ax, points, camera_pos, camera_orient):
    ax.clear()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')  # 3D points
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='r', marker='^')  # Camera position
    # You can add more details like camera orientation, frustums, etc.

# Loop over frames
frames = []
for pos, orient in zip(camera_positions, camera_orientations):
    plot_camera_view(ax, points, pos, orient)
    plt.draw()
    plt.pause(0.01)  # Pause to update the plot

    # Save frame
    frame_filename = f'frame_{len(frames)}.png'
    plt.savefig(frame_filename)
    frames.append(frame_filename)

# Compile frames into a video
with imageio.get_writer('camera_motion.mp4', fps=20) as writer:
    for frame_file in frames:
        writer.append_data(imageio.imread(frame_file))
        os.remove(frame_file)  # Clean up frames

plt.close()

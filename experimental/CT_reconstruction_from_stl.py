import numpy as np
import open3d as o3d
from stl import mesh

import open3d as o3d

# Load the STL file
stl_mesh = o3d.io.read_triangle_mesh("phantom_data/GT_stenose_self_centerline_with_side_branch.stl")

# Convert the mesh to a point cloud
point_cloud = stl_mesh.sample_points_poisson_disk(number_of_points=10000)

# Downsample the point cloud
downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=0.8)

# Add 0.2mm of noise to the downsampled point cloud
downsampled_points = np.asarray(downsampled_point_cloud.points)
noise = np.random.normal(0, 0.2, downsampled_points.shape)
noisy_downsampled_point_cloud = o3d.geometry.PointCloud()
noisy_downsampled_point_cloud.points = o3d.utility.Vector3dVector(downsampled_points + noise)



# Set colors for the point clouds
green_color = [0, 1, 0]  # Green
blue_color = [0, 0, 1]   # Blue
red_color = [1, 0, 0] # Red

# Colorize the point clouds
downsampled_point_cloud.paint_uniform_color(red_color)
noisy_downsampled_point_cloud.paint_uniform_color(blue_color)
point_cloud.paint_uniform_color(green_color)

# Save the downsampled point cloud to a PLY file
o3d.io.write_point_cloud("phantom_data/noisy_downsampled_point_cloud_with_branch.ply", noisy_downsampled_point_cloud)

# Visualize the original and downsampled point clouds
o3d.visualization.draw_geometries([point_cloud], window_name="Original Point Cloud")
o3d.visualization.draw_geometries([downsampled_point_cloud, noisy_downsampled_point_cloud], window_name="Downsampled vs. Noisy Downsampled Point Clouds")

# Load the noisy point cloud
noisy_downsampled_point_cloud = o3d.io.read_point_cloud("phantom_data/noisy_downsampled_point_cloud_with_branch.ply")

# Define the output file path
output_file = "phantom_data/noisy_downsampled_point_cloud_with_branch.txt"

# Extract the points as a numpy array
points = noisy_downsampled_point_cloud.points
# Write the coordinates to the text file
with open(output_file, "w") as file:
    for point in points:
        x, y, z = point
        file.write(f"{x} {y} {z}\n")


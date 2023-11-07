import open3d as o3d
import numpy as np

# Load the red and blue point clouds
red_pcd = o3d.io.read_point_cloud("original_cropped.pcd")
blue_pcd = o3d.io.read_point_cloud("original_cropped2.pcd")

# Create KDTree for blue point cloud
blue_tree = o3d.geometry.KDTreeFlann(blue_pcd)

# Define a radius for removing blue points
radius = 4  # 0.4mm in meters

# Initialize an empty list to store indices of blue points to remove
indices_to_remove = []

# Iterate through each red point and find nearby blue points
for red_point in red_pcd.points:
    [k, idx, _] = blue_tree.search_radius_vector_3d(red_point, radius)
    indices_to_remove.extend(idx[1:])  # Skip the first index, which is the red point itself

# Create a new blue point cloud without the selected points
new_blue_points = np.delete(np.asarray(blue_pcd.points), indices_to_remove, axis=0)
new_blue_pcd = o3d.geometry.PointCloud()
new_blue_pcd.points = o3d.utility.Vector3dVector(new_blue_points)

# Update colors for visualization
red_pcd.paint_uniform_color([1, 0, 0])  # Red
new_blue_pcd.paint_uniform_color([0, 0, 1])  # Blue

# Visualize the results
o3d.visualization.draw_geometries([red_pcd, new_blue_pcd])

import open3d as o3d
import numpy as np

# Load the PLY file
ply_file = "phantom_data/GT_stenose_self_centerline_quality.ply"
point_cloud = o3d.io.read_point_cloud(ply_file)

# Extract the XYZ coordinates and quality values as NumPy arrays
points = np.asarray(point_cloud.points)

with open("phantom_data/GT_stenose_self_centerline_quality_noheader.txt", "r") as file:
    lines = file.readlines()

quality_values = []
for line in lines:
    parts = line.split()
    if len(parts) >= 4:
        quality = float(parts[3])
        if quality < 1:
            quality_values.append(quality)

quality = np.asarray(quality_values)  
# Find the highest and lowest quality values
highest_quality = np.max(quality)
lowest_quality = np.min(quality)

# Define a colormap from red to green
colormap = np.zeros((len(quality), 3))
colormap[:, 0] = ((quality - lowest_quality) / (highest_quality - lowest_quality))  # Red component
colormap[:, 1] = (1 - (quality - lowest_quality) / (highest_quality - lowest_quality))  # Green component

# Create a new PointCloud with colors based on quality
colorized_point_cloud = o3d.geometry.PointCloud()
colorized_point_cloud.points = o3d.utility.Vector3dVector(points)
colorized_point_cloud.colors = o3d.utility.Vector3dVector(colormap)

# Visualize the colorized point cloud as a heatmap
o3d.visualization.draw_geometries([colorized_point_cloud], window_name="Heatmap Point Cloud")

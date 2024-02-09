import open3d as o3d
import numpy as np

# Load your point cloud from a .xyz file
data = np.loadtxt('ArCoMo_Data/ArCoMo3/output/ArCoMo3_point_cloud_calc.xyz')

# Create a PointCloud object
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(data)

# Downsample the point cloud
voxel_size = 0.025
cloud_downsampled = cloud.voxel_down_sample(voxel_size)

# Upsample the point cloud using nearest neighbor interpolation
radius = 0.02
max_nn = 10
cloud_upsampled = cloud_downsampled.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

# Save your output point cloud to a .xyz file
np.savetxt('your_resampled_point_cloud.xyz', np.asarray(cloud_upsampled.points))

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def parse_lumen_point_cloud(file_path):
    # Initialize lists to store the parsed values
    data = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into three values
            parts = line.strip().split()

            # Ensure there are three values on each line
            if len(parts) == 3:
                # Parse the values as floats and append them to the respective lists
                px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                data.append((px, py, pz))
            else:
                print(f"Skipping invalid line: {line.strip()}")

    return np.array(data)

def parse_point_cloud_CT_lumen(file_path):
    # Initialize lists to store the parsed values
    data = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into three values
            parts = line.strip().split()

            # Ensure there are three values on each line
            if len(parts) == 3:
                # Parse the values as floats and append them to the respective lists
                px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                data.append((px, py, pz))
            else:
                print(f"Skipping invalid line: {line.strip()}")
    # Convert each inner list to strings with spaces between elements
    formatted_data = [' '.join(map(str, inner)) for inner in data]

    # Convert the formatted strings back to a list of lists
    result = [list(map(float, inner.split())) for inner in formatted_data]
    return np.array(result)



file_path_2 = "ct_point_cloud_filtered.txt"  # Replace with the path to your text file
file_path_3 = "saved_registered_splines.txt"
point_cloud1 = parse_point_cloud_CT_lumen(file_path_2)
point_cloud2 = parse_lumen_point_cloud(file_path_3)




# Create an empty point cloud to store the fused cloud
fused_cloud = np.concatenate((point_cloud1, point_cloud2), axis=0)

# Create an Open3D PointCloud object from the fused point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(fused_cloud)
# Estimate nromals 
pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
pcd.orient_normals_consistent_tangent_plane(1000)
#o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# Perform surface reconstruction using Poisson surface reconstruction
poisson_mesh, possion_density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, width=0, scale=1, linear_fit=False)
bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)
o3d.visualization.draw_geometries([p_mesh_crop])

densities = np.asarray(possion_density)
desnity_colors = plt.get_cmap("plasma")((densities - densities.min()) / (densities.max() - densities))
denisty_colors = desnity_colors[:,:3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = poisson_mesh.vertices
density_mesh.triangles = poisson_mesh.triangles
density_mesh.triangle_normals = poisson_mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(denisty_colors)
o3d.visualization.draw_geometries([density_mesh])


# Save the reconstructed surface to a mesh file (e.g., STL or PLY)
#o3d.io.write_triangle_mesh("reconstructed_surface.ply", mesh)
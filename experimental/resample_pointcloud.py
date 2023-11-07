import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud

def parse_lumen_point_cloud(file_path, file_path_2):
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
    
    data = np.array(data)

    data2 = []
    # Open the text file for reading
    with open(file_path_2, 'r') as file:
        for line in file:
            # Split the line into three values
            parts = line.strip().split()

            # Ensure there are three values on each line
            if len(parts) == 3:
                # Parse the values as floats and append them to the respective lists
                px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                data2.append((px, py, pz))
            else:
                print(f"Skipping invalid line: {line.strip()}")
    
    data2 = np.array(data2)

    if False:
        x_filtered = data[:, 0]
        y_filtered = data[:, 1]
        z_filtered = data[:, 2]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_filtered, y_filtered, z_filtered, c="blue", marker='o')
        ax.set_xlabel('Px')
        ax.set_ylabel('Py')
        ax.set_zlabel('Pz')
        plt.show()
    
    if True:
        x_filtered = data[:, 0]
        y_filtered = data[:, 1]
        z_filtered = data[:, 2]

        x_filtered2 = data2[:, 0]
        y_filtered2 = data2[:, 1]
        z_filtered2 = data2[:, 2]

        plt.plot(z_filtered, x_filtered, 'o')
        plt.plot(z_filtered2, x_filtered2, 'x')
        plt.show()
        plt.plot(z_filtered, y_filtered, 'o')
        plt.plot(z_filtered2, y_filtered2, 'x')
        plt.show()

    return data


def parse_point_cloud(file_path):
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
    
    data = np.array(data)
    return data


def poisson_disk_sampling_3d(points, radius, num_samples=1000):
    # Convert the input point cloud to a NumPy array
    points = np.array(points)

    # Initialize the list of sampled points
    sampled_points = [points[0]]

    # Initialize the grid
    cell_size = radius / np.sqrt(3)
    grid = [[[] for _ in range(int(np.ceil(np.max(points[:, 0]) / cell_size)))] for _ in range(int(np.ceil(np.max(points[:, 1]) / cell_size)))]

    # Initialize the active list with the first point
    active_list = [0]

    while active_list:
        index = np.random.choice(active_list)
        current_point = points[index]

        found = False
        for _ in range(num_samples):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            distance = np.random.uniform(radius, 2 * radius)
            x = current_point[0] + distance * np.sin(phi) * np.cos(theta)
            y = current_point[1] + distance * np.sin(phi) * np.sin(theta)
            z = current_point[2] + distance * np.cos(phi)
            new_point = np.array([x, y, z])
            grid_x, grid_y, grid_z = int(x / cell_size), int(y / cell_size), int(z / cell_size)

            if (
                grid_x >= 0 and grid_x < len(grid) and
                grid_y >= 0 and grid_y < len(grid[0]) and
                grid_z >= 0 and grid_z < len(grid[0][0]) and
                all(np.linalg.norm(new_point - p) >= radius for p in grid[grid_x][grid_y][grid_z])
            ):
                found = True
                sampled_points.append(new_point)
                active_list.append(len(sampled_points) - 1)
                grid[grid_x][grid_y][grid_z].append(new_point)

        if not found:
            active_list.remove(index)

    return np.array(sampled_points)


if __name__ == "__main__":
    if False:
        file_path = "G:/NEW/Anonymous Male_Centerline 2_copy_copy_002.txt"
        file_path_2 = "G:/ArCoMo6_lumen_pp.txt"

        #point_cloud = parse_lumen_point_cloud(file_path, file_path_2)

            # Load the .PLY file
        mesh = o3d.io.read_triangle_mesh("G:/NEW/Anonymous Male_Centerline 2_copy_copy_001.ply")
        orig_point_cloud = o3d.io.read_point_cloud("G:/NEW/Anonymous Male_Centerline 2_copy_copy_001.ply")

        # Decimate the mesh (optional)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)

        # Create a regular point cloud from the mesh with a smaller voxel size
        voxel_size = 0.02  # Adjust this value to control point density (smaller values = denser point cloud)
        pcd = mesh.sample_points_uniformly(number_of_points=int(mesh.get_surface_area() / voxel_size))

        o3d.io.write_point_cloud("regular_point_cloud.ply", pcd)
        point_cloud = o3d.io.read_point_cloud("regular_point_cloud.ply")

        # Extract the points as a numpy array
        points = point_cloud.points

        # Define the output file name
        output_file = "ct_point_cloud_dense.txt"

        # Write the points to the text file
        with open(output_file, "w") as file:
            for point in points:
                file.write(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}\n")

        # Create visualizations for the original mesh and the regular point cloud
        o3d.visualization.draw_geometries([orig_point_cloud], window_name="Original Point cloud")
        o3d.visualization.draw_geometries([point_cloud], window_name="Regual denser Point Cloud")

        

    #----------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------#
    if False:
        # Load the .PLY file
        mesh = o3d.io.read_triangle_mesh("workflow_processed_3d_models/mesh_low_density.ply")

        # Decimate the mesh (optional)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)

        # Create a regular point cloud from the mesh with a smaller voxel size
        voxel_size = 0.05  # Adjust this value to control point density (smaller values = denser point cloud)
        pcd = mesh.sample_points_uniformly(number_of_points=int(mesh.get_surface_area() / voxel_size))

        o3d.io.write_point_cloud("workflow_processed_3d_models/regular_point_cloud_low_density.ply", pcd)
        point_cloud = o3d.io.read_point_cloud("workflow_processed_3d_models/regular_point_cloud_low_density.ply")

        # Extract the points as a numpy array
        points = point_cloud.points

        # Define the output file name
        output_file = "regular_point_cloud_low_density.XYZ"

        # Write the points to the text file
        with open(output_file, "w") as file:
            for point in points:
                file.write(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}\n")

        # Create visualizations for the original mesh and the regular point cloud
        o3d.visualization.draw_geometries([pcd])

    if False:
        file_path = "workflow_processed_data_output/fused_point_cloud.xyz"
        # Load the .XYZ file
        # Load your XYZ point cloud
        points = np.loadtxt(file_path)

        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Save it as a PLY file
        o3d.io.write_point_cloud("workflow_processed_data_output/fused_point_cloud.ply", pcd)

        # Load the PLY file using PyntCloud
        cloud = PyntCloud.from_file("workflow_processed_data_output/fused_point_cloud.ply")

        # Define the voxel size (controls point density, smaller values = denser point cloud)
        voxel_size = 0.05

        # Resample the point cloud with a uniform distribution
        sampled_cloud = cloud.add_scalar_field("uniform_sampling", radius=voxel_size)

        # Get the sampled point cloud as a NumPy array
        sampled_points = np.array(sampled_cloud.points)

    if False: 
        file_path = "workflow_processed_data_output/fused_point_cloud.xyz"

        # Load your 3D point cloud (replace this with your data)
        points = parse_point_cloud(file_path) # Example random 3D points for demonstration

        # Define the desired minimum distance between points (radius)
        radius = 0.00005

        # Perform Poisson Disk Sampling
        sampled_points = poisson_disk_sampling_3d(points, radius)
        print(len(sampled_points))
    # Create a 3D scatter plot of the original and sampled points
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Original points (in blue)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', label='Original Points', s=10)

        # Sampled points (in red)
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c='red', label='Sampled Points', s=20)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Original vs. Sampled 3D Points')

        plt.show()

    if False:
        import open3d as o3d
        file_path = "workflow_processed_data_output/fused_point_cloud.xyz"

        # Load your point cloud with holes (replace with your data)
        pcd = o3d.io.read_point_cloud(file_path)

        # Use MovingLeastSquares for hole filling
        pcd = pcd.voxel_down_sample(voxel_size=0.05)  # Optional downsampling
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Use MLS for hole filling
        radius = 0.1
        mls = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd, radius=radius)
        mls.paint_uniform_color([0.7, 0.7, 0.7])

        pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
        pcd = pcd.orient_normals_consistent_tangent_plane(k_tangent=15, k_normal=25)
        pcd = pcd.select_by_index(pcd, np.isfinite(pcd.points).all(axis=1))

    if True:
        file_path = "workflow_processed_data_output/fused_point_cloud.xyz"

        # Load your point cloud (replace with your file path)
        pcd = o3d.io.read_point_cloud(file_path)
        # Define the sampling rate (every_k_points) for uniform downsampling
        every_k_points = 10  # Adjust this value as needed

        # Perform uniform down sampling
        downsampled_pcd = o3d.geometry.uniform_down_sample(pcd, every_k_points)
        # Visualize the downsampled point cloud
        o3d.visualization.draw_geometries([downsampled_pcd])


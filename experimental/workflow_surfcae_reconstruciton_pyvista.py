import numpy as np
import pyvista as pv
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


def pyvista_method_CT(points, alpha):
    # Create a PyVista point cloud object
    poly = pv.PolyData(points)

    # Perform Delaunay triangulation with the specified alpha value
    shell = poly.delaunay_3d(alpha=alpha, progress_bar=True)
    shell = shell.extract_geometry().triangulate()

    # Create a PyVista mesh from the 'shell'
    mesh = pv.PolyData(shell)

    return mesh

# Define the file paths for your point cloud data
file_path = "workflow_processed_data_output/ct_point_cloud_dense.txt"  # Replace with the path to your text file

# Parse the point clouds from the files
point_cloud = parse_lumen_point_cloud(file_path)

#fused_cloud = np.concatenate((point_cloud1, point_cloud2), axis=0)

# Define different alpha values for high-density and low-density regions
alpha_low_density = 0.4  # Adjust this value as needed
alpha_medium_density = 0.3   # Adjust this value as needed
alpha_high_density = 0.2   # Adjust this value as needed

# Call the pyvista_method_CT function twice with different alpha values
mesh_high_density = pyvista_method_CT(point_cloud, alpha_high_density)
mesh_medium_density = pyvista_method_CT(point_cloud, alpha_medium_density)
mesh_low_density = pyvista_method_CT(point_cloud, alpha_low_density)


# Define the file path for saving the fused mesh in STL format
output_mesh_file_high = "workflow_processed_3d_models/mesh_high_density.ply"
output_mesh_file_medium = "workflow_processed_3d_models/mesh_medium_density.ply"
output_mesh_file_low = "workflow_processed_3d_models/mesh_low_density.ply"

# Save the fused mesh to the specified file
pv.save_meshio(output_mesh_file_high, mesh_high_density)
pv.save_meshio(output_mesh_file_medium, mesh_medium_density)
pv.save_meshio(output_mesh_file_low, mesh_low_density)

# Create a plotter object for visualization
plotter = pv.Plotter()

# Add the fused mesh to the plotter
plotter.add_mesh(mesh_high_density, color="green", opacity=0.5, show_edges=True)

# Show the plot
plotter.show()

# Create a plotter object for visualization
plotter = pv.Plotter()

# Add the fused mesh to the plotter
plotter.add_mesh(mesh_low_density, color="green", opacity=0.5, show_edges=True)

# Show the plot
plotter.show()
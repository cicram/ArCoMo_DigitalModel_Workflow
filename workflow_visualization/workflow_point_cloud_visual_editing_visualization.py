import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, LassoSelector
from matplotlib.path import Path
import numpy as np
from scipy.spatial import KDTree


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


# Define a function to remove overlapping points from point_cloud1
def remove_overlap(point_cloud1, point_cloud2, overlap_threshold=0.001):
    # Build KD-tree for faster nearest neighbor search
    kdtree = KDTree(point_cloud2)

    # Find indices of points in point_cloud1 that don't overlap with point_cloud2
    non_overlap_indices = []
    for point in point_cloud1:
        _, idx = kdtree.query(point)
        distance = np.linalg.norm(point - point_cloud2[idx])
        if distance > overlap_threshold:
            non_overlap_indices.append(point)

    # Create a new point cloud with non-overlapping points
    non_overlap_point_cloud1 = np.array(non_overlap_indices)

    return non_overlap_point_cloud1


# Sample point clouds (replace with your own data)
file_path_2 = "phantom_data/noisy_downsampled_point_cloud.txt"  # Replace with the path to your text file
file_path_3 = "workflow_processed_data_output/saved_registered_splines.txt"
point_cloud1 = parse_point_cloud_CT_lumen(file_path_2)
point_cloud2 = parse_lumen_point_cloud(file_path_3)

# Set an overlap threshold (adjust as needed)
overlap_threshold = 2

# Remove overlapping points from point_cloud1
non_overlap_point_cloud1 = remove_overlap(point_cloud1, point_cloud2, overlap_threshold)

# Get the min and max z values from point_cloud2
min_z = np.min(point_cloud2[:, 2])
max_z = np.max(point_cloud2[:, 2])

# Filter the removed points from point_cloud1 based on z values
filtered_points_from_point_cloud1 = [point for point in point_cloud1 if point[2] < min_z or point[2] > max_z]

# Create a list to store removed points
removed_points = []

# Create figure and subplots for 3D view and 2D views
fig = plt.figure(figsize=(12, 8))

# Create subplots for 3D view and 2D views
ax3d = fig.add_subplot(141, projection='3d')
ax_x = fig.add_subplot(142)
ax_y = fig.add_subplot(143)
ax_xy = fig.add_subplot(144)

# Plot 3D and 2D point clouds
scatter1 = ax3d.scatter(point_cloud1[:, 0], point_cloud1[:, 1], point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
scatter2 = ax3d.scatter(point_cloud2[:, 0], point_cloud2[:, 1], point_cloud2[:, 2], c='r', marker='x', label='Cloud 2')

ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')


scatter_x = ax_x.scatter(point_cloud1[:, 0], point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
scatter_x_2 = ax_x.scatter(point_cloud2[:, 0], point_cloud2[:, 2], c='r', marker='x', label='Cloud 2', alpha=0.5)

scatter_y = ax_y.scatter(point_cloud1[:, 1], point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
scatter_y_2 = ax_y.scatter(point_cloud2[:, 1], point_cloud2[:, 2], c='r', marker='x', label='Cloud 2', alpha=0.5)

scatter_xy = ax_xy.scatter(point_cloud1[:, 0], point_cloud1[:, 1], c='b', marker='o', label='Cloud 1')
scatter_xy_2 = ax_xy.scatter(point_cloud2[:, 0], point_cloud2[:, 1], c='r', marker='x', label='Cloud 2', alpha=0.5)

# Initialize selected indices
selected_indices = set()

# Function to handle point selection in 3D view
def onpick(event):
    if event.artist == scatter1:
        ind = event.ind
        selected_indices.update(ind)
        update_selection()

# Function to update the selection appearance in all subplots
def update_selection_lasso():
    mask = np.ones_like(point_cloud1[:, 0], dtype=bool)
    mask[list(selected_indices)] = False
    new_point_cloud1 = point_cloud1[mask]

    mask = np.zeros_like(point_cloud1[:, 0], dtype=bool)
    mask[list(selected_indices)] = True
    green_point_cloud1 = point_cloud1[mask]

    # Store the current axis limits
    xlim_3d, ylim_3d = ax3d.get_xlim(), ax3d.get_ylim()
    xlim_x, ylim_x = ax_x.get_xlim(), ax_x.get_ylim()
    xlim_y, ylim_y = ax_y.get_xlim(), ax_y.get_ylim()
    xlim_xy, ylim_xy = ax_xy.get_xlim(), ax_xy.get_ylim()

    # Update 3D plot
    ax3d.clear()
    scatter1 = ax3d.scatter(new_point_cloud1[:, 0], new_point_cloud1[:, 1], new_point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
    scatter1 = ax3d.scatter(green_point_cloud1[:, 0], green_point_cloud1[:, 1], green_point_cloud1[:, 2], c='green', marker='o', label='Cloud 1')
    scatter2 = ax3d.scatter(point_cloud2[:, 0], point_cloud2[:, 1], point_cloud2[:, 2], c='r', marker='x', label='Cloud 2', alpha=0.5)
    ax_x.set_xlim(xlim_3d)
    ax_x.set_ylim(ylim_3d)

    # Update XZ-plane plot
    ax_x.clear()
    #ax_x.view_init(elev=current_elevation, azim=90)
    scatter_x = ax_x.scatter(new_point_cloud1[:, 0], new_point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
    scatter_x = ax_x.scatter(green_point_cloud1[:, 0], green_point_cloud1[:, 2], c='green', marker='o', label='Cloud 1')
    scatter_x_2 = ax_x.scatter(point_cloud2[:, 0], point_cloud2[:, 2], c='r', marker='x', label='Cloud 2', alpha=0.5)
    ax_x.set_xlabel('X')
    ax_x.set_ylabel('Z')
    # Restore the axis limits
    ax_x.set_xlim(xlim_x)
    ax_x.set_ylim(ylim_x)

    # Update YZ-plane plot
    ax_y.clear()
    #ax_y.view_init(elev=current_elevation, azim=0)
    scatter_y = ax_y.scatter(new_point_cloud1[:, 1], new_point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
    scatter_y = ax_y.scatter(green_point_cloud1[:, 1], green_point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
    scatter_y_2 = ax_y.scatter(point_cloud2[:, 1], point_cloud2[:, 2], c='r', marker='x', label='Cloud 2', alpha=0.5)
    ax_y.set_xlabel('Y')
    ax_y.set_ylabel('Z')
    # Restore the axis limits
    ax_y.set_xlim(xlim_y)
    ax_y.set_ylim(ylim_y)

    # Update XY-plane plot
    ax_xy.clear()
    #ax_xy.view_init(elev=0, azim=current_azimuth)
    scatter_xy = ax_xy.scatter(new_point_cloud1[:, 0], new_point_cloud1[:, 1], c='b', marker='o', label='Cloud 1')
    scatter_xy = ax_xy.scatter(green_point_cloud1[:, 0], green_point_cloud1[:, 1], c='b', marker='o', label='Cloud 1')
    scatter_xy_2 = ax_xy.scatter(point_cloud2[:, 0], point_cloud2[:, 1], c='r', marker='x', label='Cloud 2', alpha=0.5)
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    # Restore the axis limits
    ax_xy.set_xlim(xlim_xy)
    ax_xy.set_ylim(ylim_xy)

    fig.canvas.draw()


# Function to update the selection appearance in all subplots
def update_selection():
    mask = np.ones_like(point_cloud1[:, 0], dtype=bool)
    mask[list(selected_indices)] = False
    new_point_cloud1 = point_cloud1[mask]

    # Store the current axis limits
    xlim_3d, ylim_3d = ax3d.get_xlim(), ax3d.get_ylim()
    xlim_x, ylim_x = ax_x.get_xlim(), ax_x.get_ylim()
    xlim_y, ylim_y = ax_y.get_xlim(), ax_y.get_ylim()
    xlim_xy, ylim_xy = ax_xy.get_xlim(), ax_xy.get_ylim()

    # Update 3D plot
    ax3d.clear()
    scatter1 = ax3d.scatter(new_point_cloud1[:, 0], new_point_cloud1[:, 1], new_point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
    scatter2 = ax3d.scatter(point_cloud2[:, 0], point_cloud2[:, 1], point_cloud2[:, 2], c='r', marker='x', label='Cloud 2', alpha=0.5)
    ax_x.set_xlim(xlim_3d)
    ax_x.set_ylim(ylim_3d)
    
    # Update XZ-plane plot
    ax_x.clear()
    #ax_x.view_init(elev=current_elevation, azim=90)
    scatter_x = ax_x.scatter(new_point_cloud1[:, 0], new_point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
    scatter_x_2 = ax_x.scatter(point_cloud2[:, 0], point_cloud2[:, 2], c='r', marker='x', label='Cloud 2', alpha=0.5)
    ax_x.set_xlabel('X')
    ax_x.set_ylabel('Z')
    # Restore the axis limits
    ax_x.set_xlim(xlim_x)
    ax_x.set_ylim(ylim_x)

    # Update YZ-plane plot
    ax_y.clear()
    #ax_y.view_init(elev=current_elevation, azim=0)
    scatter_y = ax_y.scatter(new_point_cloud1[:, 1], new_point_cloud1[:, 2], c='b', marker='o', label='Cloud 1')
    scatter_y_2 = ax_y.scatter(point_cloud2[:, 1], point_cloud2[:, 2], c='r', marker='x', label='Cloud 2', alpha=0.5)
    ax_y.set_xlabel('Y')
    ax_y.set_ylabel('Z')
    # Restore the axis limits
    ax_y.set_xlim(xlim_y)
    ax_y.set_ylim(ylim_y)

    # Update XY-plane plot
    ax_xy.clear()
    #ax_xy.view_init(elev=0, azim=current_azimuth)
    scatter_xy = ax_xy.scatter(new_point_cloud1[:, 0], new_point_cloud1[:, 1], c='b', marker='o', label='Cloud 1')
    scatter_xy_2 = ax_xy.scatter(point_cloud2[:, 0], point_cloud2[:, 1], c='r', marker='x', label='Cloud 2', alpha=0.5)
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    # Restore the axis limits
    ax_xy.set_xlim(xlim_xy)
    ax_xy.set_ylim(ylim_xy)

    fig.canvas.draw()

# Connect the pick event to the point selection function
fig.canvas.mpl_connect('pick_event', onpick)

# Create a button to remove selected points
ax_remove_button = plt.axes([0.8, 0.01, 0.1, 0.04])
remove_button = Button(ax_remove_button, 'Remove Selected')

def remove_selection(event):
    global point_cloud1, removed_points
    for i in sorted(selected_indices, reverse=True):
        removed_points.append(point_cloud1[i])
        point_cloud1 = np.delete(point_cloud1, i, axis=0)
    selected_indices.clear()
    update_selection()

remove_button.on_clicked(remove_selection)

# Enable LassoSelector for 2D views and update selected points
def onlasso(verts, ax):
    path = Path(verts)
    selected_indices.clear()
    
    # Determine which axis we are working with
    if ax == ax_x:
        for i, point in enumerate(point_cloud1):
            if path.contains_point((point[0], point[2])):
                selected_indices.add(i)
    elif ax == ax_y:
        for i, point in enumerate(point_cloud1):
            if path.contains_point((point[1], point[2])):
                selected_indices.add(i)
    elif ax == ax_xy:
        for i, point in enumerate(point_cloud1):
            if path.contains_point((point[0], point[1])):
                selected_indices.add(i)
    
    update_selection_lasso()

lasso_x = LassoSelector(ax_x, lambda verts: onlasso(verts, ax_x))
lasso_y = LassoSelector(ax_y, lambda verts: onlasso(verts, ax_y))
lasso_xy = LassoSelector(ax_xy, lambda verts: onlasso(verts, ax_xy))

# Create a button to save the point cloud
ax_save_button = plt.axes([0.65, 0.01, 0.1, 0.04])
save_button = Button(ax_save_button, 'Save Point Cloud')

def save_point_cloud(event):
    global point_cloud1
    if len(point_cloud1) > 0:
        # Save point_cloud_CT_filtered to a text file
        with open("ct_point_cloud_filtered.txt", "w") as file:
            for point in point_cloud1:
                file.write(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}\n")
    plt.close()

save_button.on_clicked(save_point_cloud)

plt.show()

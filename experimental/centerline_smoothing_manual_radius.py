# Function to handle key presses
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D


def on_key(event, ax, pc_line, colors):
    global current_point_index
    if event.key == 'right':
        if current_point_index < len(pc_line) - 1:
            current_point_index += 1
            colors[current_point_index] = 'blue'
            colors[current_point_index - 1] = 'red'
            update_plot(ax, pc_line)
    elif event.key == 'left':
        if current_point_index > 0:
            current_point_index -= 1
            colors[current_point_index] = 'blue'
            colors[current_point_index + 1] = 'red'
            update_plot(ax, pc_line)
    elif event.key == 'up':
        if current_point_index > 0:
            current_point_index -= 10
            colors[current_point_index] = 'blue'
            colors[current_point_index + 10] = 'red'
            update_plot(ax, pc_line)
    elif event.key == 'down':
        if current_point_index < len(pc_line) - 10:
            current_point_index += 10
            colors[current_point_index] = 'blue'
            colors[current_point_index - 10] = 'red'
            update_plot(ax, pc_line)
    elif event.key == ' ':
        save_point_cloud_to_file("location_center.txt", pc_line[current_point_index, :])


def save_point_cloud_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(f"{data[0]} {data[1]} {data[2]}\n")


# Create a function to update the plot
def update_plot(ax, pc_line):
    ax.clear()
    px = pc_line[:, 0]
    py = pc_line[:, 1]
    pz = pc_line[:, 2]

    ax.scatter(px, py, pz, c=colors, marker='o')

    ax.set_xlabel('Px')
    ax.set_ylabel('Py')
    ax.set_zlabel('Pz')


# Updated visualize_point_cloud_lumen function
def visualize_point_cloud_artery(pc_line, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    update_plot(ax, pc_line)

    ax.mouse_init(rotate_btn=3)

    # Connect the key press event to the key handler function
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, ax, pc_line, colors))

    plt.show()


def parse_point_cloud_centerline(file_path, display_results):
    flag_header = False
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 21:  # Ensure the line has at least 3 values
                if not flag_header:
                    flag_header = True
                else:
                    px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                    data.append((px, py, pz))
            else:
                flag_header = False

    data = np.array(data)
    return data

def parse_marked_point(file_path):
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

    data = np.array(data)
    return data

def calculate_radius(point_start, point_end):
    # Calculate the radius as half the distance between point_start and point_end
    return np.linalg.norm(point_end - point_start) / 2

def calculate_normal_vector(point_start, point_end, point_center):
    # Calculate the normal vector of the plane defined by point_start, point_end, and point_center
    v1 = point_end - point_start
    v2 = point_center - point_start
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def calculate_circle_center(point_start, point_end, radius):
    # Calculate the center of the circle
    return point_start + (point_end - point_start) / 2


def create_3d_curve(point_start, point_end, point_center, num_points=100):
    radius = calculate_radius(point_start, point_end)
    normal = calculate_normal_vector(point_start, point_end, point_center)
    center = calculate_circle_center(point_start, point_end, radius)

    # Create points on the circle using a for loop
    theta = np.linspace(0, np.pi, num_points)

    # Construct a rotation matrix to align the circle with the plane
    z_axis = normal
    x_axis = np.cross(np.array([1, 0, 0]), z_axis)  # Use [1, 0, 0] as a reference for the x-axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.array([x_axis, y_axis, z_axis])
    rotation_matrix = rotation_matrix[:, 0]

    # Apply the rotation to the points on the circle
    points_on_circle = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(num_points)))
    rotated_points = np.dot(points_on_circle, rotation_matrix) + center

    return rotated_points


if __name__ == "__main__":
    current_point_index = 0
    file_path_1 = "centerline.txt" 
    display_results = False
    pc_centerline = parse_point_cloud_centerline(file_path_1, display_results)
    if False:
        colors = ['blue'] + ['red'] * (len(pc_centerline) - 1)
        visualize_point_cloud_artery(pc_centerline, colors)
    point_start = parse_marked_point("location_1.txt")
    point_end = parse_marked_point("location_2.txt")
    point_center = parse_marked_point("location_center.txt")

    # Create the 3D curve
    curve_points = create_3d_curve(point_start, point_end, point_center)

    # Extract x, y, and z coordinates
    x_filtered = curve_points[:, 0]
    y_filtered = curve_points[:, 1]
    z_filtered = curve_points[:, 2]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_filtered, y_filtered, z_filtered, c="blue", marker='o')

    # Show the plot
    x_filtered = pc_centerline[:, 0]
    y_filtered = pc_centerline[:, 1]
    z_filtered = pc_centerline[:, 2]
    ax.scatter(x_filtered, y_filtered, z_filtered, c="red", marker='o')

    ax.set_xlabel('Px')
    ax.set_ylabel('Py')
    ax.set_zlabel('Pz')
    plt.show()

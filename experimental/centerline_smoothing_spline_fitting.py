# Function to handle key presses
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
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

if __name__ == "__main__":
    current_point_index = 0
    file_path_1 = "workflow_data/centerline.txt" 
    display_results = False
    pc_centerline = parse_point_cloud_centerline(file_path_1, display_results)
    if True:
        colors = ['blue'] + ['red'] * (len(pc_centerline) - 1)
        visualize_point_cloud_artery(pc_centerline, colors)

    point_start = parse_marked_point("location_2.txt")
    point_end = parse_marked_point("location_1.txt")
    point_center = parse_marked_point("location_center.txt")

    indices_start = np.where(np.all(pc_centerline == point_start, axis=1))[0][0]
    indices_end = np.where(np.all(pc_centerline == point_end, axis=1))[0][0]

    fitting_points = []
    number_points = 10

    for i in range(number_points):
        fitting_points.append(pc_centerline[indices_start-number_points-1+i])
  
    for i in range(number_points):
        fitting_points.append(pc_centerline[indices_end+i])

    indices_start_crop =  indices_start - number_points - 1
    indices_stop_crop = indices_end + number_points

    points = fitting_points
    x, y, z = zip(*points)
    # Create a cubic spline using splprep
    tck, u = splprep([x, y, z], s=0, per=0)

    # Evaluate the spline at a higher number of points for smoother appearance
    u_new = np.linspace(0, 1, 50)
    spline_points = splev(u_new, tck)

    pc_centerline = np.delete(pc_centerline, slice(indices_start_crop, indices_stop_crop), axis=0)

    for i, p in enumerate(spline_points[0]):
        point = [spline_points[0][i], spline_points[1][i], spline_points[2][i]]
        pc_centerline = np.insert(pc_centerline, indices_start_crop, point, axis=0)
        indices_start += 1   

    if False: 
        # Create a 3D scatter plot for the original points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', label='Original Points', marker='o')

        # Create a 3D line plot for the cubic spline
        ax.plot(spline_points[0], spline_points[1], spline_points[2], "x", c='r', label='Cubic Spline')


        # Set labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()

        # Show the plot
        x_filtered = pc_centerline[:, 0]
        y_filtered = pc_centerline[:, 1]
        z_filtered = pc_centerline[:, 2]
        ax.scatter(x_filtered, y_filtered, z_filtered, c="blue", marker='o')

        ax.set_xlabel('Px')
        ax.set_ylabel('Py')
        ax.set_zlabel('Pz')
        plt.show()

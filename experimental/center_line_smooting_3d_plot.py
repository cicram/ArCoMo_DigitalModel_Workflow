import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d import Axes3D

current_green_index = 0  # Initialize the current green point index
current_blue_index = 1  # Initialize the current blue point index
view_limits = None  # Initialize view limits as None
move_blue_point = True  # Flag to determine whether to move the blue or green point

def on_key(event, ax, pc_line, colors):
    global current_green_index, current_blue_index, view_limits, move_blue_point
    if event.key == 'right':
        if current_blue_index < len(pc_line) - 1:
            if move_blue_point:
                current_blue_index += 1
                colors[current_blue_index] = 'red'
                colors[current_blue_index - 1] = 'black'
            else:
                current_green_index += 1
                colors[current_green_index] = 'yellow'
                colors[current_green_index - 1] = 'black'

            if current_blue_index <= current_green_index:
                colors[current_blue_index] = 'black'
                current_blue_index = current_green_index + 1
                colors[current_blue_index] = 'red'
            elif current_blue_index == current_green_index:
                colors[current_blue_index] = 'yellow'
                current_blue_index = current_green_index + 1
                colors[current_blue_index] = 'red'

            view_limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())  # Save view limits
            update_plot(ax, pc_line, current_blue_index, current_green_index, colors)
    elif event.key == 'left':
        if current_blue_index > 0:
            if move_blue_point:
                current_blue_index -= 1
                colors[current_blue_index] = 'red'
                colors[current_blue_index + 1] = 'black'
            else:
                current_green_index -= 1
                colors[current_green_index] = 'yellow'
                colors[current_green_index + 1] = 'black'

            if current_blue_index <= current_green_index:
                colors[current_blue_index] = 'black'
                current_blue_index = current_green_index + 1
                colors[current_blue_index] = 'red'
            elif current_blue_index == current_green_index:
                colors[current_blue_index] = 'yellow'
                current_blue_index = current_green_index + 1
                colors[current_blue_index] = 'red'

            view_limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())  # Save view limits
            update_plot(ax, pc_line, current_blue_index, current_green_index, colors)
    elif event.key == 'up':
        if current_blue_index > 0:
            if move_blue_point:
                current_blue_index -= 10
                colors[current_blue_index] = 'red'
                colors[current_blue_index + 10] = 'black'
            else:
                current_green_index -= 10
                colors[current_green_index] = 'yellow'
                colors[current_green_index + 10] = 'black'
            
            if current_blue_index < current_green_index:
                colors[current_blue_index] = 'black'
                current_blue_index = current_green_index + 1
                colors[current_blue_index] = 'red'
            elif current_blue_index == current_green_index:
                colors[current_blue_index] = 'yellow'
                current_blue_index = current_green_index + 1
                colors[current_blue_index] = 'red'

            view_limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())  # Save view limits
            update_plot(ax, pc_line, current_blue_index, current_green_index, colors)
    elif event.key == 'down':
        if current_blue_index < len(pc_line) - 10:
            if move_blue_point:
                current_blue_index += 10
                colors[current_blue_index] = 'red'
                colors[current_blue_index - 10] = 'black'
            else:
                current_green_index += 10
                colors[current_green_index] = 'yellow'
                colors[current_green_index - 10] = 'black'

            if current_blue_index <= current_green_index:
                colors[current_blue_index] = 'black'
                current_blue_index = current_green_index + 1
                colors[current_blue_index] = 'red'
            elif current_blue_index == current_green_index:
                colors[current_blue_index] = 'yellow'
                current_blue_index = current_green_index + 1
                colors[current_blue_index] = 'red'
                
            view_limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())  # Save view limits
            update_plot(ax, pc_line, current_blue_index, current_green_index, colors)

    elif event.key == 't':
        move_blue_point = not move_blue_point  # Toggle between moving blue and green points
    elif event.key == ' ':
        save_point_cloud_to_file("location_center.txt", pc_line[current_blue_index, :])

def save_point_cloud_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(f"{data[0]} {data[1]} {data[2]}\n")

# Create a function to update the plot
def update_plot(ax, pc_line, blue_index, green_index, colors):
    ax.clear()
    px = pc_line[:, 0]
    py = pc_line[:, 1]
    pz = pc_line[:, 2]

    ax.scatter(px, py, pz, c=colors, marker='o')

    ax.set_xlabel('Px')
    ax.set_ylabel('Py')
    ax.set_zlabel('Pz')

    if view_limits:
        ax.set_xlim(view_limits[0])
        ax.set_ylim(view_limits[1])
        ax.set_zlim(view_limits[2])

# Updated visualize_point_cloud_lumen function
def visualize_point_cloud_artery(pc_line, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    update_plot(ax, pc_line, current_blue_index, current_green_index, colors)  # Initialize the plot

    ax.mouse_init(rotate_btn=3)

    # Connect the key press event to the key handler function
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, ax, pc_line, colors))

    plt.show()

def parse_point_cloud_centerline(file_path):
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

if __name__ == "__main__":
    file_path_1 = "workflow_data/centerline.txt"

    pc_centerline = parse_point_cloud_centerline(file_path_1)
    colors = ['black'] * len(pc_centerline)
    colors[current_blue_index] = 'red'
    colors[current_green_index] = 'yellow'
    visualize_point_cloud_artery(pc_centerline, colors)

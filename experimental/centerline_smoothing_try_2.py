import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Cursor
import pandas as pd
import matplotlib.patches as patches

current_green_index = 0  # Initialize the current green point index
current_blue_index = 1  # Initialize the current blue point index
move_blue_point = True  # Flag to determine whether to move the blue or green point
view_limits = None  # Initialize view limits as None

def on_blue_button_clicked(event):
    global current_blue_index
    if current_blue_index < len(pc_line) - 1:
        current_blue_index += 1
        update_plot()

def on_green_button_clicked(event):
    global current_green_index
    if current_green_index < len(pc_line) - 1:
        current_green_index += 1
        update_plot()

def on_toggle_button_clicked(event):
    global move_blue_point
    move_blue_point = not move_blue_point

def on_pick(event):
    ind = event.ind[0]
    if move_blue_point:
        global current_blue_index
        current_blue_index = ind
    else:
        global current_green_index
        current_green_index = ind
    update_plot()

def create_buttons(ax):
    blue_button = Button(ax, 'Move Blue')
    blue_button.on_clicked(on_blue_button_clicked)

    green_button = Button(ax, 'Move Green')
    green_button.on_clicked(on_green_button_clicked)

    toggle_button = Button(ax, 'Toggle')
    toggle_button.on_clicked(on_toggle_button_clicked)

def create_checkboxes(ax):
    move_blue_rect = patches.Rectangle((0.05, 0.85), 0.03, 0.03, fill=move_blue_point, color="blue")
    move_green_rect = patches.Rectangle((0.05, 0.8), 0.03, 0.03, fill=not move_blue_point, color="green")
    
    ax.add_patch(move_blue_rect)
    ax.add_patch(move_green_rect)
    
    ax.text(0.1, 0.865, 'Move Blue', transform=ax.transAxes)
    ax.text(0.1, 0.815, 'Move Green', transform=ax.transAxes)

    move_blue_rect.set_picker(True)
    move_green_rect.set_picker(True)

    return move_blue_rect, move_green_rect

def on_checkbox_clicked(event):
    global move_blue_point
    if event.artist.get_color() == "blue":
        move_blue_point = True
    else:
        move_blue_point = False
    update_plot()

def update_plot():
    ax.clear()
    ax.set_xlabel('Px')
    ax.set_ylabel('Py')
    ax.set_zlabel('Pz')
    ax.set_xlim(view_limits[0])
    ax.set_ylim(view_limits[1])
    ax.set_zlim(view_limits[2])

    px = pc_line[:, 0]
    py = pc_line[:, 1]
    pz = pc_line[:, 2]

    colors = ['black'] * len(pc_line)
    colors[current_blue_index] = 'red'
    colors[current_green_index] = 'yellow'

    ax.scatter(px, py, pz, c=colors, marker='o', picker=True)

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

    pc_line = parse_point_cloud_centerline(file_path_1)
    view_limits = (pc_line[:, 0].min(), pc_line[:, 0].max()), (pc_line[:, 1].min(), pc_line[:, 1].max()), (pc_line[:, 2].min(), pc_line[:, 2].max())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    create_buttons(plt.axes([0.7, 0.1, 0.2, 0.05]))  # Create buttons in the figure
    move_blue_rect, move_green_rect = create_checkboxes(plt.axes([0.7, 0.2, 0.2, 0.1]))
    update_plot()

    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('pick_event', on_checkbox_clicked)

    plt.show()

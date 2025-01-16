import pydicom
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import csv
from collections import defaultdict
import numpy as np

#######################################################
# Overlap splines
#######################################################

spline_points_4bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_spline_points_4_bar.csv'
spline_points_12bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_12bar_segmentation/output_spline_points_12_bar.csv'
spline_points_18bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_18bar_segmentation/output_spline_points_18_bar.csv'
centerline_csv = 'C:/Users/JL/Code/uCT/centerline.csv'

# Define a defaultdict to group data by the index (last value in each row)
grouped_points_4bar = defaultdict(list)
grouped_points_12bar = defaultdict(list)
grouped_points_18bar = defaultdict(list)
centerline = defaultdict(list)

# Read the CSV file and group the data
with open(spline_points_4bar_csv, 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        # Convert the elements from strings to floats or integers as needed
        x, y, idx = float(row[0]), float(row[1]), float(row[2])
        
        # Group by the index
        grouped_points_4bar[idx].append([x, y])

with open(spline_points_12bar_csv, 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        # Convert the elements from strings to floats or integers as needed
        x, y, idx = float(row[0]), float(row[1]), float(row[2])
        
        # Group by the index
        grouped_points_12bar[idx].append([x, y])

with open(spline_points_18bar_csv, 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        # Convert the elements from strings to floats or integers as needed
        x, y, idx = float(row[0]), float(row[1]), float(row[2])
        
        # Group by the index
        grouped_points_18bar[idx].append([x, y])

if False:
    with open(centerline_csv, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert the elements from strings to floats or integers as needed
            x, y, idx = float(row[0]), float(row[1]), float(row[2])
            
            # Group by the index
            centerline[idx].append([x, y])

# Load the DICOM file
dicom_path = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/C0003921_BIO18_LAD_balloon_4bar.dcm'
dicom_data = pydicom.dcmread(dicom_path)
pixel_array = dicom_data.pixel_array

# Get the center of the DICOM image
dicom_height, dicom_width = pixel_array.shape[1], pixel_array.shape[2]  # Assuming 3D volume [frames, height, width]
dicom_center_x = dicom_width / 2
dicom_center_y = dicom_height / 2

total_frames = pixel_array.shape[0]
length_per_frame = total_frames / 63  # Assume 63 slices
z_spacing = 0.0172
scaling = 1 / z_spacing  # Scaling factor

x_values_4bar_shifted = None
y_values_4bar_shifted = None
x_values_12bar_shifted = None
y_values_12bar_shifted = None
x_values_18bar_shifted = None
y_values_18bar_shifted = None
frame_index = None

# Assuming grouped_points_4bar, grouped_points_12bar, grouped_points_18bar, scaling, and z_spacing are defined

# Lists to store the results for plotting
z_heights = []
lengths_4bar = []
directions_4bar = []
lengths_12bar = []
directions_12bar = []
lengths_18bar = []
directions_18bar = []



# Function to compute vector lengths and directions
def compute_vectors(x_values, y_values):
    if len(x_values) > 1:  # Ensure there's enough data for SVD
        points = np.vstack((x_values, y_values)).T
        _, S, Vt = np.linalg.svd(points)
        lengths = S
        directions = [np.degrees(np.arctan2(v[1], v[0])) for v in Vt[:2]]
        if directions[1] < 0:
            directions[1] = 180 + directions[1]
        return lengths, directions
    return [0, 0], [0, 0]  # Return zeros if not enough points

# Iterate through z heights
for z_height, points_4bar in grouped_points_4bar.items():
    # Get corresponding points from the 18bar and 12bar data, if they exist
    points_18bar = grouped_points_18bar.get(z_height, [])
    points_12bar = grouped_points_12bar.get(z_height, [])

    # Unzip the list of points into x and y coordinates
    x_values_4bar, y_values_4bar = zip(*points_4bar) if points_4bar else ([], [])
    x_values_12bar, y_values_12bar = zip(*points_12bar) if points_12bar else ([], [])
    x_values_18bar, y_values_18bar = zip(*points_18bar) if points_18bar else ([], [])

    # Convert x and y values to NumPy arrays for scaling
    x_values_4bar_shifted = np.array(x_values_4bar) * scaling
    y_values_4bar_shifted = np.array(y_values_4bar) * scaling
    x_values_12bar_shifted = np.array(x_values_12bar) * scaling
    y_values_12bar_shifted = np.array(y_values_12bar) * scaling
    x_values_18bar_shifted = np.array(x_values_18bar) * scaling
    y_values_18bar_shifted = np.array(y_values_18bar) * scaling

    frame_index = int(z_height / z_spacing)

    if True:
        # Create a new figure for each idx
        fig, ax1 = plt.subplots(figsize=(12, 10))

        # Show the DICOM image for the current frame
        ax1.imshow(pixel_array[frame_index], cmap='gray')

        # Plot the shifted points from 4 bar, 12 bar, and 18 bar data
        #ax1.scatter(x_values_4bar_shifted[0::15], y_values_4bar_shifted[0::15], label='4 bar', color='b')
        #ax1.scatter(x_values_12bar_shifted[0::15], y_values_12bar_shifted[0::15], label='12 bar', color='g')
        #ax1.scatter(x_values_18bar_shifted[0::15], y_values_18bar_shifted[0::15], label='18 bar', color='r')

        # Add title, legend, and turn off axis ticks
        ax1.set_title(f'Z-height: {z_height}')
        ax1.legend()
        ax1.axis('off')  # Hide axis ticks for better visualization
        fig.savefig('C:/Users/JL/Code/uCT/4bar.png')   # save the figure to file
        # Show the plot
        plt.show()


    # Store the z-height
    z_heights.append(z_height)

    # Compute and store vectors for 4bar
    lengths, directions = compute_vectors(x_values_4bar_shifted, y_values_4bar_shifted)
    lengths_4bar.append(lengths)
    directions_4bar.append(directions)

    # Compute and store vectors for 12bar
    lengths, directions = compute_vectors(x_values_12bar_shifted, y_values_12bar_shifted)
    lengths_12bar.append(lengths)
    directions_12bar.append(directions)

    # Compute and store vectors for 18bar
    lengths, directions = compute_vectors(x_values_18bar_shifted, y_values_18bar_shifted)
    lengths_18bar.append(lengths)
    directions_18bar.append(directions)

# Convert lists to NumPy arrays for easier plotting
z_heights = np.array(z_heights)
lengths_4bar = np.array(lengths_4bar)
directions_4bar = np.array(directions_4bar)
lengths_12bar = np.array(lengths_12bar)
directions_12bar = np.array(directions_12bar)
lengths_18bar = np.array(lengths_18bar)
directions_18bar = np.array(directions_18bar)

# Create four subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot first vector lengths (SVD) for 4bar, 12bar, 18bar in one plot
axs[0, 0].plot(z_heights, lengths_4bar[:, 0], label='4bar - Vector 1', color='blue')
axs[0, 0].plot(z_heights, lengths_12bar[:, 0], label='12bar - Vector 1', color='green')
axs[0, 0].plot(z_heights, lengths_18bar[:, 0], label='18bar - Vector 1', color='red')
axs[0, 0].set_title('First Vector Lengths')
axs[0, 0].set_xlabel('Z Height')
axs[0, 0].set_ylabel('Length')
axs[0, 0].legend()

# Plot second vector lengths (SVD) for 4bar, 12bar, 18bar in another plot
axs[0, 1].plot(z_heights, lengths_4bar[:, 1], label='4bar - Vector 2', color='blue')
axs[0, 1].plot(z_heights, lengths_12bar[:, 1], label='12bar - Vector 2', color='green')
axs[0, 1].plot(z_heights, lengths_18bar[:, 1], label='18bar - Vector 2', color='red')
axs[0, 1].set_title('Second Vector Lengths')
axs[0, 1].set_xlabel('Z Height')
axs[0, 1].set_ylabel('Length')
axs[0, 1].legend()

# Plot first vector directions for 4bar, 12bar, 18bar in another plot
axs[1, 0].plot(z_heights, directions_4bar[:, 0], label='4bar - Vector 1', color='blue')
axs[1, 0].plot(z_heights, directions_12bar[:, 0], label='12bar - Vector 1', color='green')
axs[1, 0].plot(z_heights, directions_18bar[:, 0], label='18bar - Vector 1', color='red')
axs[1, 0].set_title('First Vector Directions')
axs[1, 0].set_xlabel('Z Height')
axs[1, 0].set_ylabel('Direction (degrees)')
axs[1, 0].legend()

# Plot second vector directions for 4bar, 12bar, 18bar in the last plot
axs[1, 1].plot(z_heights, directions_4bar[:, 1], label='4bar - Vector 2', color='blue')
axs[1, 1].plot(z_heights, directions_12bar[:, 1], label='12bar - Vector 2', color='green')
axs[1, 1].plot(z_heights, directions_18bar[:, 1], label='18bar - Vector 2', color='red')
axs[1, 1].set_title('Second Vector Directions')
axs[1, 1].set_xlabel('Z Height')
axs[1, 1].set_ylabel('Direction (degrees)')
axs[1, 1].legend()

# Adjust the layout
plt.tight_layout()
plt.show()

# Function to compute and print vector lengths and directions
def compute_and_print_vectors(x_values, y_values, label):
    points = np.vstack((x_values, y_values)).T
    U, S, Vt = np.linalg.svd(points)
    
    # Print the lengths (singular values) and directions (angles in degrees)
    print(f"\n{label}:")
    for i in range(2):  # Two principal directions in 2D
        vector = Vt[i]  # Principal vector
        length = S[i]  # Magnitude (singular value)
        # Compute the angle of the vector in degrees
        angle = np.degrees(np.arctan2(vector[1], vector[0]))
        print(f"  Principal Vector {i+1}: Length = {length:.4f}, Direction = {angle:.2f}°")


# Function to plot points and their principal vectors using SVD
def plot_points_and_vectors(x_values, y_values, label, color):
    points = np.vstack((x_values, y_values)).T
    U, S, Vt = np.linalg.svd(points)
    
    # Mean center the points (optional, depends on your application)
    points_mean = points.mean(axis=0)

    # Plot the points
    plt.scatter(points[:, 0], points[:, 1], label=label, color=color)

    # Plot the principal vectors (SVD directions)
    for i in range(2):  # There are two principal directions in 2D
        vector = Vt[i]  # The i-th principal component
        # Scale the vector by the singular values to reflect its significance
        scaling = 100*(1-i) + 50
        plt.quiver(points_mean[0], points_mean[1], vector[0] * S[i]/scaling, vector[1] * S[i]/scaling, 
                   angles='xy', scale_units='xy', scale=1, color=color, linewidth=2)

# Compute and print the vectors for 4bar, 12bar, and 18bar
compute_and_print_vectors(x_values_4bar_shifted, y_values_4bar_shifted, label='4bar')
compute_and_print_vectors(x_values_12bar_shifted, y_values_12bar_shifted, label='12bar')
compute_and_print_vectors(x_values_18bar_shifted, y_values_18bar_shifted, label='18bar')

# Create the plot
plt.figure(figsize=(8, 8))

# Plot 4bar points and vectors
plot_points_and_vectors(x_values_4bar_shifted, y_values_4bar_shifted, label='4bar Points', color='blue')

# Plot 12bar points and vectors
plot_points_and_vectors(x_values_12bar_shifted, y_values_12bar_shifted, label='12bar Points', color='green')

# Plot 18bar points and vectors
plot_points_and_vectors(x_values_18bar_shifted, y_values_18bar_shifted, label='18bar Points', color='red')

plt.gca().invert_yaxis()

# Set equal scaling for the axes and add labels
plt.gca().set_aspect('equal', adjustable='box')
plt.axvline(0, color='black',linewidth=0.5)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Points and Principal Vectors for 4bar, 12bar, and 18bar')
plt.legend()
plt.grid(True)
plt.show()

####################
# GIF

if False:
    # Create a new figure for each idx
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # Show the DICOM image for the current frame
    ax1.imshow(pixel_array[frame_index], cmap='gray')

    # Plot the shifted points from 4 bar, 12 bar, and 18 bar data
    ax1.scatter(x_values_4bar_shifted[0::10], y_values_4bar_shifted[0::10], label=f'4 bar z-height {z_height}', color='b')

    # Add title, legend, and turn off axis ticks
    ax1.set_title(f'Z-height: {z_height}')
    ax1.legend()
    ax1.axis('off')  # Hide axis ticks for better visualization
    fig.savefig('C:/Users/JL/Code/uCT/4bar.png')   # save the figure to file
    # Show the plot
    plt.show()


    # Create a new figure for each idx
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # Show the DICOM image for the current frame
    ax1.imshow(pixel_array[frame_index], cmap='gray')

    # Plot the shifted points from 4 bar, 12 bar, and 18 bar data
    ax1.scatter(x_values_12bar_shifted[0::10], y_values_12bar_shifted[0::10], label=f'12 bar z-height {z_height}', color='b')

    # Add title, legend, and turn off axis ticks
    ax1.set_title(f'Z-height: {z_height}')
    ax1.legend()
    ax1.axis('off')  # Hide axis ticks for better visualization
    fig.savefig('C:/Users/JL/Code/uCT/12bar.png')   # save the figure to file
    # Show the plot
    plt.show()

    # Create a new figure for each idx
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # Show the DICOM image for the current frame
    ax1.imshow(pixel_array[frame_index], cmap='gray')

    # Plot the shifted points from 4 bar, 12 bar, and 18 bar data
    ax1.scatter(x_values_18bar_shifted[0::10], y_values_18bar_shifted[0::10], label=f'18 bar z-height {z_height}', color='b')

    # Add title, legend, and turn off axis ticks
    ax1.set_title(f'Z-height: {z_height}')
    ax1.legend()
    ax1.axis('off')  # Hide axis ticks for better visualization
    fig.savefig('C:/Users/JL/Code/uCT/18bar.png')   # save the figure to file
    # Show the plot
    plt.show()


    import glob
    import contextlib
    from PIL import Image

    # filepaths
    fp_in = "C:/Users/JL/Code/uCT/*.png"
    fp_out = "C:/Users/JL/Code/uCT/image.gif"

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))

        # extract  first image from iterator
        img = next(imgs)

        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=600, loop=100)

##############################################################
if False:
    # Load the DICOM file
    dicom_path = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/C0003921_BIO18_LAD_balloon_4bar.dcm'
    dicom_data = pydicom.dcmread(dicom_path)

    bar_18_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_18bar_segmentation/output_areas_18_bar.csv'
    bar_12_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_12bar_segmentation/output_areas_12_bar.csv'
    bar_4_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_areas_4_bar.csv'

    df_18bar = pd.read_csv(bar_18_csv)
    df_12bar = pd.read_csv(bar_12_csv)
    df_4bar = pd.read_csv(bar_4_csv)

    colors = ['black', 'blue', 'green']

    # Extract pixel data
    pixel_array = dicom_data.pixel_array

    total_frames = pixel_array.shape[0]
    length_per_frame = total_frames/63 
    frame_index_1 = int(df_4bar['Centerline IDX'][1]/z_spacing)
    frame_index_2 = int(df_4bar['Centerline IDX'][5]/z_spacing)
    frame_index_3 = int(df_4bar['Centerline IDX'][10]/z_spacing)
    frame_index_4 = int(df_4bar['Centerline IDX'][15]/z_spacing)
    frame_index_5 = int(df_4bar['Centerline IDX'][19]/z_spacing)

    # Create a figure with gridspec for custom layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 5, height_ratios=[1, 2])  # 2 rows, 3 columns, larger height for the DICOM images

    # Plot 1: Line plot spanning the top row
    ax0 = fig.add_subplot(gs[0, :])
    colors = ['blue', 'red', 'green']
    ax0.plot(df_4bar['Centerline IDX'], df_4bar['Area'], marker='o', linestyle='-', color=colors[0], label='4 bar')
    ax0.plot(df_12bar['Centerline IDX'], df_12bar['Area'], marker='o', linestyle='-', color=colors[1], label='12 bar')
    ax0.plot(df_18bar['Centerline IDX'], df_18bar['Area'], marker='o', linestyle='-', color=colors[2], label='18 bar')
    ax0.set_xlabel('z-height')
    ax0.set_ylabel('Area')
    ax0.set_title('Z-height vs Area')
    ax0.legend()
    ax0.grid(True)

    # Plot 2: DICOM plane at index 15 (left)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(pixel_array[frame_index_1], cmap='gray')
    ax1.set_title(f"Centerline idx {df_4bar['Centerline IDX'][1]}")
    ax1.axis('off')

    # Plot 3: DICOM plane at index 20 (middle)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(pixel_array[frame_index_2], cmap='gray')
    ax2.set_title(f"Centerline idx {df_4bar['Centerline IDX'][5]}")
    ax2.axis('off')

    # Plot 4: DICOM plane at index 29 (right)
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(pixel_array[frame_index_3], cmap='gray')
    ax3.set_title(f"Centerline idx {df_4bar['Centerline IDX'][10]}")
    ax3.axis('off')

    # Plot 4: DICOM plane at index 29 (right)
    ax3 = fig.add_subplot(gs[1, 3])
    ax3.imshow(pixel_array[frame_index_4], cmap='gray')
    ax3.set_title(f"Centerline idx {df_4bar['Centerline IDX'][15]}")
    ax3.axis('off')

    # Plot 4: DICOM plane at index 29 (right)
    ax3 = fig.add_subplot(gs[1, 4])
    ax3.imshow(pixel_array[frame_index_5], cmap='gray')
    ax3.set_title(f"Centerline idx {df_4bar['Centerline IDX'][19]}")
    ax3.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Show the combined figure
    plt.show()







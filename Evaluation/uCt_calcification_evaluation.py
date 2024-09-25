import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter
from scipy.ndimage import center_of_mass
import csv
from collections import defaultdict
from math import atan2, degrees
from matplotlib.lines import Line2D
import pandas as pd

# Paths to CSV files
spline_points_4bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_spline_points_4_bar.csv'
spline_points_8bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_8bar_segmentation/output_spline_points_8_bar.csv'
spline_points_12bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_12bar_segmentation/output_spline_points_12_bar.csv'
spline_points_16bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_16bar_segmentation/output_spline_points_16_bar.csv'
spline_points_18bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_18bar_segmentation/output_spline_points_18_bar.csv'
spline_points_20bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_20bar_segmentation/output_spline_points_20_bar.csv'
calc_spline_points_4bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_spline_points_4_bar_calc.csv'

# Reading the CSV files into grouped data
grouped_points_4bar = defaultdict(list)
grouped_points_8bar = defaultdict(list)
grouped_points_12bar = defaultdict(list)
grouped_points_16bar = defaultdict(list)
grouped_points_18bar = defaultdict(list)
grouped_points_20bar = defaultdict(list)
grouped_points_4bar_calc = defaultdict(list)

areas_4bar_balloon_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_areas_4_bar.csv'  
df_area_4_bar_balloon = pd.read_csv(areas_4bar_balloon_csv)
areas_4bar_calc_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_areas_4_bar_calc.csv'  
df_area_4_bar_calc = pd.read_csv(areas_4bar_calc_csv)
areas_4bar_tissue_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_areas_4_bar_tissue.csv'  
df_area_4_bar_tissue = pd.read_csv(areas_4bar_tissue_csv)

area_ratio_calc_lumen = df_area_4_bar_balloon["Area"]/df_area_4_bar_calc["Area"]
area_ratio_calc_tissue = df_area_4_bar_tissue["Area"]/df_area_4_bar_calc["Area"]

# Example z_values and z_spacing for 3D stack
z_spacing = 0.0172

def read_spline_points_integer(csv_file, grouped_points):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            x, y, idx = float(row[0]), float(row[1]), float(row[2])
            grouped_points[idx].append([int(x/z_spacing), int(y/z_spacing)])

def read_spline_points(csv_file, grouped_points):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            x, y, idx = float(row[0]), float(row[1]), float(row[2])
            grouped_points[idx].append([x, y])

read_spline_points_integer(spline_points_4bar_csv, grouped_points_4bar)
read_spline_points(spline_points_8bar_csv, grouped_points_8bar)
read_spline_points(spline_points_12bar_csv, grouped_points_12bar)
read_spline_points(spline_points_16bar_csv, grouped_points_16bar)
read_spline_points(spline_points_18bar_csv, grouped_points_18bar)
read_spline_points(spline_points_20bar_csv, grouped_points_20bar)
read_spline_points_integer(calc_spline_points_4bar_csv, grouped_points_4bar_calc)

# Load the DICOM image
dicom_path = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/C0003921_BIO18_LAD_balloon_4bar.dcm'
dicom_data = pydicom.dcmread(dicom_path)
pixel_array = dicom_data.pixel_array

# Define cropping region
x_start, x_end = 150, 600
y_start, y_end = 150, 600

z_values = sorted(grouped_points_4bar.keys())

def compute_spline_centroid(spline_points):
    x_coords = [point[0] for point in spline_points]
    y_coords = [point[1] for point in spline_points]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return centroid_x, centroid_y

# Function to draw lines and handle user input
def draw_lines_in_image(cropped_image, spline_center):
    # Copy the original image to draw on
    image_copy = cropped_image.copy()
    
    # List to store the points where the user clicks
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
            # Store the clicked point
            points.append((x, y))
            # Draw a circle at the clicked point
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)  # Green circle

            # Draw a line from spline center to the clicked point
            cv2.line(image_copy, (int(spline_center[0]), int(spline_center[1])), (x, y), (255, 0, 0), 2)  # Blue line
            
            # Show the updated image
            cv2.imshow("Image", image_copy)

    # Show the image and set the mouse callback
    cv2.imshow("Image", image_copy)
    cv2.setMouseCallback("Image", click_event)
    no_flag = False

    while True:
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF
        
        # If the 'a' key is pressed, accept the lines and close the image
        if key == ord('a') and len(points) == 2:
            break
        # If 'q' key is pressed, exit without saving
        elif key == ord('q'):
            no_flag = True
            break

    # Close the window
    cv2.destroyAllWindows()
        # Compute the angle between the two points
    if no_flag:
        no_flag = False
        angle = 0
        points = None
    else:
        angle = compute_angle_between_lines(spline_center, points[0], points[1])
    
    return points, angle  # Return the points and the computed angl

def compute_angle_between_lines(center, point1, point2):
    # Calculate direction vectors
    vector1 = np.array(point1) - np.array(center)
    vector2 = np.array(point2) - np.array(center)

    # Normalize the vectors
    norm_vector1 = vector1 / np.linalg.norm(vector1)
    norm_vector2 = vector2 / np.linalg.norm(vector2)

    # Calculate the angle in radians
    angle_rad = np.arccos(np.clip(np.dot(norm_vector1, norm_vector2), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

fig2, axs = plt.subplots(3, 7, figsize=(15, 10))
axs = axs.ravel()  # Flatten the axis grid

for idx, z_value in enumerate(z_values):
    frame_index = int(z_value / z_spacing)
    cropped_image = pixel_array[frame_index]
  
    # Find the spline center
    spline_center_points = compute_spline_centroid(grouped_points_4bar.get(z_value, []))
    spline_center = (int(spline_center_points[0]), int(spline_center_points[1]))

    # Draw lines in image and get angles
    drawn_points, angle = draw_lines_in_image(cropped_image, spline_center)

    # Plot the image with spline center marked
    axs[idx].imshow(cropped_image, cmap='gray')
    axs[idx].plot(spline_center_points[0], spline_center_points[1], 'ro')  # Mark center
    # Draw lines for subplot
    if drawn_points is not None:
        axs[idx].plot([spline_center[0], drawn_points[0][0]], [spline_center[1], drawn_points[0][1]], 'b-', linewidth=2)  # Line to first click
        axs[idx].plot([spline_center[0], drawn_points[1][0]], [spline_center[1], drawn_points[1][1]], 'b-', linewidth=2)  # Line to second click
    axs[idx].set_title(f"Angle: {angle:.2f}°")  # Add the angle to the subplot title
    axs[idx].axis('off')  # Hide axes for clarity

plt.tight_layout()

fig3, ax = plt.subplots(figsize=(15, 10))

# Use the ax object to plot
#ax.plot(df_area_4_bar_balloon["z-height"], df_area_4_bar_balloon["Area"], label='Area Balloon', color="green")
#ax.plot(df_area_4_bar_balloon["z-height"], df_area_4_bar_calc["Area"], label='Area Calc', color="yellow")
#ax.plot(df_area_4_bar_balloon["z-height"], df_area_4_bar_tissue["Area"], label='Area Tissue', color="black")

ax.plot(df_area_4_bar_balloon["z-height"], area_ratio_calc_lumen, label='Area Ratio Balloon/Calc', color="blue")
ax.plot(df_area_4_bar_balloon["z-height"], area_ratio_calc_tissue, label='Area Ratio Tissue/Calc', color="red")

# Add labels, legend, and show the plot
ax.set_xlabel('Z-height')
ax.set_ylabel('Area Ratio')
ax.legend()

plt.show()



if False:
    # Calculate calcification angle
    #spline_calc = grouped_points_4bar_calc.get(z_value, [])

    #calc_angle_spread, calc_angles = compute_calcification_angle(cropped_image, spline_center)

    #calc_angle_spread, calc_angles = compute_calcification_angle_spline(cropped_image, spline_calc, spline_center)

    #x_coords = [point[0] for point in spline_calc]
    #y_coords = [point[1] for point in spline_calc]

    # Function to compute angle of calcification
    def compute_calcification_angle_spline(cropped_image, spline_calc, spline_center):
        height, width = cropped_image.shape
        angles_with_calc = []
        x_plot = []
        y_plot = []
        # Scan 360° around the spline center
        for angle in np.linspace(0, 2*np.pi, 360):
            for radius in range(1, max(height, width)):
                # Compute x, y coordinates in the direction of the current angle
                x = int(spline_center[0] + radius * np.cos(angle))
                y = int(spline_center[1] + radius * np.sin(angle))
                x_plot.append(x)
                y_plot.append(y)

                # Ensure the coordinates are within bounds
                if x < 0 or x >= height or y < 0 or y >= width:
                    break
                
                # Check if the pixel value exceeds the threshold
                if [x, y] in spline_calc:
                    angles_with_calc.append(np.degrees(angle))  # Store the angle in degrees
                    print("threshold reached, breaked")
                    break  # Exit the loop when calcification is found
    
        # Compute angular spread of calcification
        if angles_with_calc:
            min_angle, max_angle = min(angles_with_calc), max(angles_with_calc)
            calc_angle_spread = max_angle - min_angle
        else:
            calc_angle_spread = 0

        return calc_angle_spread, angles_with_calc

    # Function to compute angle of calcification
    def compute_calcification_angle(cropped_image, spline_center, threshold=15000):
        height, width = cropped_image.shape
        angles_with_calc = []

        # Scan 360° around the spline center
        for angle in np.linspace(0, 2*np.pi, 360):
            for radius in range(1, max(height, width)):
                # Compute x, y coordinates in the direction of the current angle
                x = int(spline_center[0] + radius * np.cos(angle))
                y = int(spline_center[1] + radius * np.sin(angle))
                
                # Ensure the coordinates are within bounds
                if x < 0 or x >= height or y < 0 or y >= width:
                    break
                
                # Check if the pixel value exceeds the threshold
                if cropped_image[x, y] > threshold:
                    angles_with_calc.append(np.degrees(angle))  # Store the angle in degrees
                    break  # Exit the loop when calcification is found

        # Compute angular spread of calcification
        if angles_with_calc:
            min_angle, max_angle = min(angles_with_calc), max(angles_with_calc)
            calc_angle_spread = max_angle - min_angle
        else:
            calc_angle_spread = 0

        return calc_angle_spread, angles_with_calc
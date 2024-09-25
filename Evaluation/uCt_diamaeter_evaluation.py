import pydicom
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import csv
from collections import defaultdict
import numpy as np

#######################################################
# File paths
#######################################################
spline_points_4bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_spline_points_4_bar.csv'
spline_points_8bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_8bar_segmentation/output_spline_points_8_bar.csv'
spline_points_12bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_12bar_segmentation/output_spline_points_12_bar.csv'
spline_points_16bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_16bar_segmentation/output_spline_points_16_bar.csv'
spline_points_18bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_18bar_segmentation/output_spline_points_18_bar.csv'
spline_points_20bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_20bar_segmentation/output_spline_points_20_bar.csv'
areas_4bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_areas_4_bar.csv'  
areas_8bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_8bar_segmentation/output_areas_8_bar.csv'  
areas_12bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_12bar_segmentation/output_areas_12_bar.csv'  
areas_16bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_16bar_segmentation/output_areas_16_bar.csv'  
areas_18bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_18bar_segmentation/output_areas_18_bar.csv'
areas_20bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_20bar_segmentation/output_areas_20_bar.csv'  

#######################################################
# Reading the CSV files into grouped data
#######################################################
df_area_4_bar = pd.read_csv(areas_4bar_csv)
df_area_8_bar = pd.read_csv(areas_8bar_csv)
df_area_12_bar = pd.read_csv(areas_12bar_csv)
df_area_16_bar = pd.read_csv(areas_16bar_csv)
df_area_18_bar = pd.read_csv(areas_18bar_csv)
df_area_20_bar = pd.read_csv(areas_20bar_csv)

grouped_points_4bar = defaultdict(list)
grouped_points_8bar = defaultdict(list)
grouped_points_12bar = defaultdict(list)
grouped_points_16bar = defaultdict(list)
grouped_points_18bar = defaultdict(list)
grouped_points_20bar = defaultdict(list)

def read_spline_points(csv_file, grouped_points):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            x, y, idx = float(row[0]), float(row[1]), float(row[2])
            grouped_points[idx].append([x, y])

read_spline_points(spline_points_4bar_csv, grouped_points_4bar)
read_spline_points(spline_points_8bar_csv, grouped_points_8bar)
read_spline_points(spline_points_12bar_csv, grouped_points_12bar)
read_spline_points(spline_points_16bar_csv, grouped_points_16bar)
read_spline_points(spline_points_18bar_csv, grouped_points_18bar)
read_spline_points(spline_points_20bar_csv, grouped_points_20bar)

#######################################################
# Hoop stress calculation function
#######################################################
def hoop_stress(r_i, P):
    r_o = r_i + 1 / 1000  # Outer radius = +1 mm
    return (P * r_i**2) / (r_o**2 - r_i**2) + (r_i**2 * r_o**2 * P) / (r_i**2 * (r_o**2 - r_i**2))

# Compute average radius from the points
def compute_average_radius(x_values, y_values):
    centroid_x = np.mean(x_values)
    centroid_y = np.mean(y_values)
    x_shifted = x_values - centroid_x
    y_shifted = y_values - centroid_y
    r_i = np.sqrt(x_shifted**2 + y_shifted**2)
    return np.mean(r_i)

#######################################################
# Data processing and calculations
#######################################################
# Internal pressures in Pascals
P_4bar = 4 * 1e5
P_8bar = 8 * 1e5
P_12bar = 12 * 1e5
P_16bar = 16 * 1e5
P_18bar = 18 * 1e5
P_20bar = 20 * 1e5
scaling = 1  # Scaling factor for point coordinates
z_spacing = 0.0172

# Initialize storage for radii and hoop stress at each pressure
radii = {4: [], 8: [], 12: [], 18: []}
sigmas = {4: [], 8: [], 12: [],  18: []}
z_values = sorted(grouped_points_4bar.keys())  # Assuming same z-values for all pressures

# Iterate through z-heights for each pressure
for z_height in z_values:
    for pressure, grouped_points in zip([4, 8, 12, 18],
                                        [grouped_points_4bar, grouped_points_8bar, grouped_points_12bar, 
                                          grouped_points_18bar]):
        points = grouped_points.get(z_height, [])
        if points:
            x_values, y_values = zip(*points)
            x_values, y_values = np.array(x_values) * scaling, np.array(y_values) * scaling
            r_i = compute_average_radius(x_values, y_values)
            sigma = hoop_stress(r_i / 1000, pressure * 1e5)  # Convert pressure to Pascals

            radii[pressure].append(r_i)
            sigmas[pressure].append(sigma)

#######################################################
# Plotting the results
#######################################################
fig = plt.figure(figsize=(10, 12))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
colors = ['black', 'blue', 'green', 'red', 'yellow', 'magenta']

# Subplot 1: Hoop stresses
ax1 = fig.add_subplot(gs[0, 0])
idx = 0
for pressure in [4, 8, 12, 18]:
    ax1.plot(z_values, sigmas[pressure], label=f'Sigma {pressure} bar', color=colors[idx])
    idx = idx+1
ax1.set_xlabel('Z Heights')
ax1.set_ylabel('Hoop Stress (Pa)')
ax1.legend()
ax1.set_title('Hoop Stress vs Z Heights')

# Subplot 2: Radii comparisons
ax2 = fig.add_subplot(gs[1, 0])
idx = 0
for pressure in [4, 8, 12, 18]:
    ax2.plot(z_values, radii[pressure], label=f'Radii {pressure} bar', color=colors[idx])
    idx = idx+1
ax2.set_xlabel('Z Heights')
ax2.set_ylabel('Radius (mm)')
ax2.legend()
ax2.set_title('Radius vs Z Heights')

# Subplot 3: Area comparisons
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(z_values, df_area_4_bar["Area"], label='Area 4bar', color=colors[0])
ax3.plot(z_values, df_area_8_bar["Area"], label='Area 8bar', color=colors[1])
ax3.plot(z_values, df_area_12_bar["Area"], label='Area 12bar', color=colors[2])
ax3.plot(z_values, df_area_18_bar["Area"], label='Area 18bar', color=colors[3])

ax3.set_xlabel('Z Heights')
ax3.set_ylabel('Area (mm^2)')
ax3.legend()
ax3.set_title('Area vs Z Heights')

### Load dicom data
dicom_path = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/C0003921_BIO18_LAD_balloon_4bar.dcm'
dicom_data = pydicom.dcmread(dicom_path)
pixel_array = dicom_data.pixel_array

x_start, x_end = 150, 600
y_start, y_end = 150, 600

# Create a figure
fig2 = plt.figure(figsize=(15, 10))

# Set up a 2x10 grid for 20 images
gs = GridSpec(3, 7, hspace=0.3, wspace=0.1)

# Loop over z_values to plot 20 images
for idx, z_value in enumerate(z_values):
    # Calculate the frame index based on z_value and z_spacing
    frame_index = int(z_value / z_spacing)
    
    # Create a subplot for each image
    ax = fig2.add_subplot(gs[idx // 7, idx % 7])
    
    cropped_image = pixel_array[frame_index, y_start:y_end, x_start:x_end]
    ax.imshow(cropped_image, cmap='gray')

    ax.axis('off')  # Hide axes for clarity
    ax.set_title(f'z = {z_value:.2f}', fontsize=8)

# Adjust the layout and display the plot
plt.tight_layout()

##################################################################
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression
import numpy as np

# Initialize dictionaries to store deltas
delta_radii = {}
delta_sigmas = {}
slopes = []  # Store slopes for each z_height

# Loop through each z_height
for z_height in z_values:
    # Collect the radii and sigmas for each pressure at the current z_height
    radii_at_z = {}
    sigmas_at_z = {}
    
    for pressure, grouped_points in zip([4, 8, 12, 18],
                                        [grouped_points_4bar, grouped_points_8bar, grouped_points_12bar, 
                                         grouped_points_18bar]):
        points = grouped_points.get(z_height, [])
        if points:
            x_values, y_values = zip(*points)
            x_values, y_values = np.array(x_values) * scaling, np.array(y_values) * scaling
            r_i = compute_average_radius(x_values, y_values)
            sigma = hoop_stress(r_i / 1000, pressure * 1e5)  # Convert pressure to Pascals
            
            radii_at_z[pressure] = r_i
            sigmas_at_z[pressure] = sigma
    
    # Ensure we have radius at 4 bar for comparison
    if 4 in radii_at_z:
        radius_4bar = radii_at_z[4]  # Radius at 4 bar

        # Now compute only the positive deltas (left pressure > right pressure)
        pressures = sorted(radii_at_z.keys(), reverse=True)  # Sort pressures in descending order
        for (p1, p2) in combinations(pressures, 2):  # Only keep combinations where p1 > p2
            if p2 == 4:    
                # Compute deltas for radius and sigma
                delta_radius = radii_at_z[p1] - radii_at_z[p2]
                delta_radius_percentage = (delta_radius / radius_4bar) * 100  # Percentage change
                delta_sigma = sigmas_at_z[p1] - sigmas_at_z[p2]
                
                # Store deltas in the dictionaries
                delta_radii[(p1, p2, z_height)] = delta_radius_percentage
                delta_sigmas[(p1, p2, z_height)] = delta_sigma

# Create subplots for each z_height (7 rows x 3 columns)
fig, axes = plt.subplots(3, 7, figsize=(15, 20))
axes = axes.flatten()  # Flatten for easy iteration

# Fit slopes array for final plot
z_heights_fitted = []

# Loop through each z_height and create a 2D plot for delta_radii vs delta_sigmas
for i, z_height in enumerate(z_values):
    ax = axes[i]
    
    # Prepare data for plotting
    delta_r_vals = []
    delta_s_vals = []
    
    for (p1, p2, z_h) in delta_radii.keys():
        if z_h == z_height:
            delta_r_vals.append(delta_radii[(p1, p2, z_height)])
            delta_s_vals.append(delta_sigmas[(p1, p2, z_height)])
    
    # Plot delta_radii (percentage) vs delta_sigmas for the current z_height
    ax.scatter(delta_r_vals, delta_s_vals, c='b', marker='o')
    
    # Fit a linear line
    if len(delta_r_vals) > 1:  # Avoid fitting if there's not enough data
        X = np.array(delta_r_vals).reshape(-1, 1)  # Reshape for sklearn
        y = np.array(delta_s_vals)
        
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        
        # Store slope for final slope plot
        slopes.append(slope)
        z_heights_fitted.append(z_height)
        
        # Plot the linear fit line
        line = model.predict(X)
        ax.plot(delta_r_vals, line, color='r', label=f"Slope: {slope:.3f}")
        ax.legend()
    
    # Set titles and labels for each subplot
    ax.set_title(f'Z Height = {z_height}')
    ax.set_xlabel('Delta Radius (%)')
    ax.set_ylabel('Delta Sigma')

# Adjust the layout to prevent overlap and add space between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Create a separate plot for slopes vs z_height
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(z_heights_fitted, slopes, marker='o', color='g')
ax2.set_title('Slope of Linear Fit vs Z Height')
ax2.set_xlabel('Z Height')
ax2.set_ylabel('Slope')
ax2.grid(True)

# Show both plots
plt.show()


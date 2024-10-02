import nibabel as nib
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull

# Load the NIfTI file
nii_file = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_0bar_segmentation/Segmentation_1.nii'
img = nib.load(nii_file)
data = img.get_fdata()  # Get voxel data (3D array)

radius = 1
data_model = []
data_model.append(["z-height", "Area calc", "Area", "Calcification Angles", "Angle range"])
points_3d = []

# Specify the Z-slice range you're interested in
start_cut = 100
end_cut = 500
z_spacing = 0.0172
cutting_points = np.linspace(z_spacing * start_cut, z_spacing * end_cut, 20)

# Function to compute the centroid of spline points
def compute_spline_centroid(points):
    if len(points) == 0:
        return [0, 0]
    return np.mean(points, axis=0)

# Function to find the largest gap without calcification
def compute_largest_no_calc_angle_span(angles_with_calc):
    if len(angles_with_calc) < 2:
        # If there are less than 2 angles with calcification, then the whole circle is without calc
        return 360.0, (0, 360)
    
    # Sort angles to find consecutive gaps
    sorted_angles = sorted(angles_with_calc)
    
    # Add the wrap-around case: gap from the last angle back to 0
    sorted_angles.append(sorted_angles[0] + 360)
    
    max_gap = 0
    max_span = (0, 0)
    
    # Iterate through sorted angles and find the largest gap
    for i in range(1, len(sorted_angles)):
        current_gap = sorted_angles[i] - sorted_angles[i - 1]
        if current_gap > max_gap:
            max_gap = current_gap
            max_span = (sorted_angles[i - 1], sorted_angles[i])
            if int(max_gap) == 1:
                max_gap = 0
    # Return the gap size and the angular span
    return max_gap, max_span



def compute_calcification_angle_and_tissue_area_inside_calc(slice_data, spline_center, calc_mask, tissue_mask):
    height, width = slice_data.shape
    angles_with_calc = []
    tissue_pixel_counts = []
    # Scan 360° around the spline center
    for angle in np.linspace(0, 2 * np.pi, 360):
        tissue_found = False
        tissue_start = None

        for radius in np.linspace(1, max(height, width), int(max(height, width)/3)):
            # Compute x, y coordinates in the direction of the current angle
            y = int(spline_center[0] + radius * np.cos(angle))
            x = int(spline_center[1] + radius * np.sin(angle))
            
            # Ensure the coordinates are within bounds
            if x < 0 or x >= height or y < 0 or y >= width:
                break
            
            if [x, y] in tissue_mask and not tissue_found:
                tissue_found = True
                tissue_start = (x, y)  # Record the first tissue pixel before calc

            # Check if the current pixel hits a calcified pixel in the mask
            if [x, y] in calc_mask:
                angles_with_calc.append(np.degrees(angle))  # Store the angle in degrees

                # Check/Compute numbers of pixels that lie between lumen and calc in the angle that is stored
                if tissue_found:
                    lumen_to_calc_distance = int(np.linalg.norm([x - tissue_start[0], y - tissue_start[1]]))
                    tissue_pixel_counts.append(lumen_to_calc_distance)
                else:
                    tissue_pixel_counts.append(0)

                break  # Exit the loop when calcification is found

    # Compute angular spread of calcification
    no_calc_span, angle_range = compute_largest_no_calc_angle_span(angles_with_calc)

    return no_calc_span, angle_range, angles_with_calc, tissue_pixel_counts




# Function to compute the calcification angle
def compute_calcification_angle(cropped_image, spline_center, calc_mask):
    height, width = cropped_image.shape
    angles_with_calc = []

    # Scan 360° around the spline center
    for angle in np.linspace(0, 2 * np.pi, 360):
        for radius in np.linspace(1, max(height, width), int(max(height, width)/3)):
            # Compute x, y coordinates in the direction of the current angle
            y = int(spline_center[0] + radius * np.cos(angle))
            x = int(spline_center[1] + radius * np.sin(angle))
            
            # Ensure the coordinates are within bounds
            if x < 0 or x >= height or y < 0 or y >= width:
                break
            
            # Check if the current pixel hits a calcified pixel in the mask
            if [x, y] in calc_mask:
                angles_with_calc.append(np.degrees(angle))  # Store the angle in degrees
                break  # Exit the loop when calcification is found

    # Compute angular spread of calcification
    no_calc_span, angle_range = compute_largest_no_calc_angle_span(angles_with_calc)

    return no_calc_span, angle_range, angles_with_calc

###############################################################
## ONLY FOR 0 BAR !!!!!
###############################################################

# Iterate through each Z-slice
for idx, val in enumerate(cutting_points):
    z_slice = int(val / z_spacing) - 15 # minus 15 to compensate for recording shift
    slice_data = data[:, :, z_slice]  # Extract the slice
    segment_label_calc = 1  # Label for calcified region ("calc"=1, "balloon"=3, etc.)
    segment_label_lumen = 2 
    segment_label_tissue = 4 
    
    ###################################################################
    #### AREAS
    ###################################################################
    # Assuming voxel size is isotropic (all sides are the same)
    voxel_size = img.header.get_zooms()[0] * img.header.get_zooms()[1]  # Voxel area (e.g., in mm²)

    # Calculate the area of calcification
    segmented_voxels_calc = np.sum(slice_data == segment_label_calc)  # Count voxels for the given segment label
    area_calc = segmented_voxels_calc * voxel_size  # Total area

    segmented_voxels_lumen_sum = np.sum(slice_data == segment_label_lumen)  # Count voxels for the given segment label
    area_lumen = segmented_voxels_lumen_sum * voxel_size  # Total area

    segmented_voxels_tissue_sum = np.sum(slice_data == segment_label_tissue)  # Count voxels for the given segment label
    area_tissue = segmented_voxels_tissue_sum * voxel_size  # Total area

    ################################################
    #### LUMEN SPLINE
    ################################################
    segmented_voxels_lumen = np.argwhere(slice_data == segment_label_lumen)  # Get voxel coordinates for the lumen

    # Convert segmented voxel coordinates to (x, y) format
    segmented_voxels_lumen_x_y = [(x, y) for x, y in segmented_voxels_lumen[:, :2]]

    # Ensure the points are in the correct format for spline fitting
    points = np.array(segmented_voxels_lumen_x_y)

    # Compute the convex hull of the points to get the boundary
    hull = ConvexHull(points)
    boundary_points = points[hull.vertices]

    # Fit a closed spline to the boundary points (s=0 for no smoothing, per=True for closed curve)
    tck, u = splprep(boundary_points.T, s=0, per=True)

    # Evaluate the spline at many points to form a smooth polygon
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_spline, y_spline = splev(u_new, tck, der=0)
    for x, y in zip(x_spline*z_spacing, y_spline*z_spacing):
        points_3d.append([x, y, cutting_points[idx]])


    ################################################
    #### CALC ANGLES
    ################################################
    spline_center_points = compute_spline_centroid(segmented_voxels_lumen_x_y)

    spline_center = (int(spline_center_points[1]), int(spline_center_points[0]))

    # Create a mask of calcified pixels for this slice
    calc_mask = np.argwhere(slice_data == segment_label_calc).tolist()
    tissue_mask = np.argwhere(slice_data == segment_label_tissue).tolist()
    lumen_mask = np.argwhere(slice_data == segment_label_lumen).tolist()

    # Calculate calcification angle
    no_calc_angle_spread, real_angle_range, angles_with_calc, tissue_pixels = compute_calcification_angle_and_tissue_area_inside_calc(slice_data, spline_center, calc_mask, tissue_mask)

    calc_angle = int(360 - no_calc_angle_spread)

    ################################################
    #### STORE DATA
    ################################################    
    # Store data: [z-height, area, filter radius, calcification angle]
    data_model.append([cutting_points[idx], area_calc, area_lumen, angles_with_calc, calc_angle])


    ################################################
    #### DISPLAYING
    ################################################
    # Transpose the slice data to switch x and y axes
    transposed_slice_data = np.transpose(slice_data)
    # Plot the slice and the angles
    plt.figure(figsize=(8, 8))
    plt.imshow(transposed_slice_data, cmap='gray', origin='lower')
    plt.scatter(*zip(*calc_mask), color='red', s=1, label='Calcification Mask')
    plt.scatter(*zip(*tissue_mask), color='blue', s=1, label='Tissue Mask')
    plt.scatter(x_spline, y_spline, color='yellow')
    plt.scatter(spline_center[1], spline_center[0], color='blue', s=50, label='Spline Center')

    # Draw lines for non-calcified angles
    for angle in angles_with_calc:
        end_x = int(spline_center[0] + 100 * np.cos(np.radians(angle)))  # Extend line outward
        end_y = int(spline_center[1] + 100 * np.sin(np.radians(angle)))
        plt.plot([spline_center[1], end_y], [spline_center[0], end_x], color='green', alpha=0.3)

    plt.title(f"Z-slice: {z_slice}, Calcification Angle: {calc_angle:.2f}°")
    plt.legend()
    plt.show()

################################################
#### DATA SAVING
################################################
# Save the data model to a CSV file
if False:
    output_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_0bar_segmentation/output_areas_0_bar_calc.csv'
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_model)

spline_points_0bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_0bar_segmentation/output_spline_points_0_bar.csv'
if False:
    with open(spline_points_0bar_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(points_3d)
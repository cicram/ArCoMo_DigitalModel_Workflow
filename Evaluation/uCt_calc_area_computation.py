import nibabel as nib
import numpy as np
import csv
# Load the NIfTI file
nii_file = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/Segmentation.nii'
img = nib.load(nii_file)
data = img.get_fdata()  # Get voxel data (3D array)

radius = 1
data_model = []
data_model.append(["z-height", "Area", "Filter radius"])

# Specify the Z-slice you're interested in
start_cut = 100
end_cut = 500
z_spacing = 0.0172
cutting_points = np.linspace(z_spacing*start_cut, z_spacing*end_cut, 20)

for idx, val in enumerate(cutting_points):
    z_slice = int(val/z_spacing)
    slice_data = data[:, :, z_slice]  # Extract the slice
    segment_label = 1  # Label for the segment ("calc"=1, "balloon"=3, etc.)

    # Assuming voxel size is isotropic (all sides are the same)
    voxel_size = img.header.get_zooms()[0] * img.header.get_zooms()[1]  # Voxel area (e.g., in mm²)

    # Calculate the area
    segmented_voxels = np.sum(slice_data == segment_label)  # Count voxels for the given segment label
    area = segmented_voxels * voxel_size  # Total area
    data_model.append([cutting_points[idx], area, radius])

ouput_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_areas_4_bar_calc.csv'

with open(ouput_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data_model)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from plyfile import PlyData, PlyElement
from matplotlib.cm import ScalarMappable
from scipy.stats import mannwhitneyu

# List of .ply file paths

ArCoMo_number = "300"
ArCoMo_number_gt = "3"

ply_file = 'Evaluation/MARC_PAPER/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_Colored_Qaulity_Overlap_Correction.ply'

names_for_plots = ["No correction", "Image correlation", "ICP", "Overlap", "Pure CT"]
all_quality_filtered = []  # List to store filtered quality data from all files


plydata = PlyData.read(ply_file)

# Extract the vertex data
vertex = plydata['vertex']
x = vertex['x']
y = vertex['y']
z = vertex['z']
quality = vertex['quality']
quality_abs = abs(quality)

# Normalize the quality values to range [0, 1]
quality_normalized = (quality_abs - np.min(quality_abs)) / (np.max(quality_abs) - np.min(quality_abs))

# Shift the scatterplot to origin (0,0,0)
x_shifted = x - np.min(x)
y_shifted = y - np.min(y)
z_shifted = z - np.min(z)

# Histogram
fig = plt.figure()
plt.hist(quality_abs[quality_abs >=0.03], bins=100, color='blue', edgecolor='black')
plt.title("Histrogram of vertex distances")
plt.xlabel('Vertex distances [mm]')
plt.ylabel('Number of vertex points')

# Create a colormap ranging from green to red (reversed RdYlGn)
cmap = cm.get_cmap('RdYlGn_r')

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x_shifted, y_shifted, z_shifted, c=cmap(quality_normalized))

# Add a colorbar
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(quality_abs), vmax=np.max(quality_abs)))
sm.set_array([])
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
fig.colorbar(sm, cax=cbar_ax, label='Absolute vertex distance difference [mm]')

# Set the labels for x, y, and z axes
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')

# Set the title for the plot
# ax.set_title("Vertex distances")

# Filter out quality values below 0.05
abs_quality = abs(quality)
quality_filtered = abs_quality[abs_quality >= 0.03]#[(quality >= 0.05) | (quality <= -0.05)]

plt.savefig('Evaluation\ArCoMo1400_superimposed')

plt.show()  # Show all histogram and scatter plots


# Calculate statistical measures
min_val = np.min(quality_filtered)
max_val = np.max(quality_filtered)
median_val = np.median(quality_filtered)
mean_val = np.mean(quality_filtered)
std_dev = np.std(quality_filtered)

# Description text
desc = f"Min: {min_val:.2f} mm\nMax: {max_val:.2f} mm\nMedian: {median_val:.2f} mm\nMean: {mean_val:.2f} mm\nStd Dev: {std_dev:.2f} mm"

# Prepare data to be written to CSV
data = [
    ["Measure", "Value"],
    ["Min", f"{min_val:.2f}"],
    ["Max", f"{max_val:.2f}"],
    ["Median", f"{median_val:.2f}"],
    ["Mean", f"{mean_val:.2f}"],
    ["Std Dev", f"{std_dev:.2f}"]
]

import csv
# Save the description to a text file
folder_path = "Evaluation/MARC_PAPER/Statistical_output/ArCoMo"  + ArCoMo_number + "/ArCoMo" + ArCoMo_number + "_"
file_path = f"{folder_path}statistical_results_mesh_gemoetry.csv"
with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)
file_path = f"{folder_path}statistical_results_mesh_gemoetry.txt"
with open(file_path, "w") as file:
    file.write(desc)


#############################################################################################################
# AREA EVALUATION #

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

z_score_flag = 0
oct_section = 1

# Load the CSV file
df_gt = pd.read_csv('Evaluation/MARC_PAPER/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number_gt + '_inner_shell.csv')
df_measured = pd.read_csv('Evaluation/MARC_PAPER/ArCoMo' + ArCoMo_number +'/ArCoMo' + ArCoMo_number + '_Colored_Qaulity_Overlap_Correction.csv')


colors = ['green', 'red', 'purple', 'orange']

if oct_section:
    if ArCoMo_number == "300":
        start_idx = 600
        end_idx = 850
    else:
        start_idx = 200
        end_idx = 350
else:
    start_idx = 1
    end_idx = -1

# Plot 'Centerline IDX' vs 'Area'
fig = plt.figure()
plt.plot(df_gt['Centerline IDX'][start_idx:end_idx], df_gt['Area'][start_idx:end_idx], marker='o', linestyle='-', color='black', label='Original Model')
plt.plot(df_measured['Centerline IDX'][start_idx:end_idx], df_measured['Area'][start_idx:end_idx], marker='o', linestyle='-', color=colors[0], label='Derivative Model')

plt.legend()
diff = abs(df_gt['Area']- df_measured['Area'])

#plt.plot(df_2['Centerline IDX'], diff, marker='o', linestyle='-', color='g')

import numpy as np
#print(f"Mean{np.mean(diff[500:800])}, SD: {np.std(diff[500:800])}")
plt.xlabel('Centerline Index [-]')
plt.ylabel(r'Area [mm^2]')
# plt.title('Centerline IDX vs Area')
plt.grid(True)

plt.savefig('Evaluation\ArCoMo1400_area_diff')
plt.show()

# Assuming the columns for volume measurements are named 'Volume'
if oct_section:
    gt_volume = df_gt['Area'][start_idx:end_idx]
    area1 = df_measured['Area'][start_idx:end_idx]       
else:
    gt_volume = df_gt['Area']
    area1 = df_measured['Area']

if z_score_flag:
    # Z-score normalization
    gt_volume = (gt_volume - gt_volume.mean()) / gt_volume.std()
    area1 = (area1 - area1.mean()) / area1.std()

analysis_data = []

abs_errors = np.abs(area1 - gt_volume)
mean_abs_error = np.mean(abs_errors)
median_abs_error = np.median(abs_errors)
max_abs_error = np.max(abs_errors)
mean_squared_error = np.mean((area1 - gt_volume) ** 2)
std_abs_error = np.std(abs_errors)


fig = plt.figure()
plt.hist(abs_errors, bins=40, color='blue', edgecolor='black')
# Add lines for mean, median, and std
plt.axvline(mean_abs_error, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_abs_error:.2f}')
plt.axvline(median_abs_error, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_abs_error:.2f}')
plt.axvline(mean_abs_error + std_abs_error, color='orange', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_abs_error:.2f}')
plt.axvline(mean_abs_error - std_abs_error, color='orange', linestyle='dashed', linewidth=1)

plt.title("Histrogram of Are difference")
plt.xlabel('Area diff [mm]')
plt.legend()
plt.ylabel('Number of areas')
plt.show() 

analysis_data.append(["Overlap", max_abs_error, mean_abs_error, median_abs_error, mean_squared_error, std_abs_error])

# Create a DataFrame for the statistical analysis data
analysis_df = pd.DataFrame(analysis_data, columns=['Method', 'Max Absolute Error', 'Mean Absolute Error','Median Absolute Error', 'Mean Squared Error', 'Std Absolute Error'])

# Save the DataFrame to an Excel file
file_path = f"{folder_path}statistical_results_area_difference_comparison.xlsx"

analysis_df.to_excel(file_path, index=False)


################################################################################
# LINEAR REGRESSION 
################################################################################

# Fit lines between ground truth volumes and volumes of other methods
slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = linregress(gt_volume, area1)


# Calculate R-squared
r_squared_1 = r_value_1 ** 2

# Calculate Pearson correlation coefficient
from scipy.stats import pearsonr
pearson_corr, _ = pearsonr(gt_volume, area1)

# Create a DataFrame for linear regression values
data = {
    'Slope': [slope_1],
    'Intercept': [intercept_1],
    'R-value': [r_value_1],
    'P-value': [p_value_1],
    'Standard Error': [std_err_1],
    'R-value-squared': [r_squared_1],
    'Pearson Correlation': [pearson_corr]
}


df_linear_regression = pd.DataFrame(data, index=['Overlap'])

# Save linear regression values to an Excel file
file_path = f"{folder_path}statistical_results_area_linear_regression.xlsx"

df_linear_regression.to_excel(file_path)


# Plotting
plt.figure(figsize=(10, 8))

# Define colors for each pair of scatter plot and fitted line
colors = ['green', 'red', 'purple', 'orange']

# Plotting data points and fitted lines with matching colors
plt.plot(gt_volume, area1, 'o', color=colors[0], label='Image correlation')
plt.plot(gt_volume, slope_1 * gt_volume + intercept_1, '--', color=colors[0], label=f'Fitted Line (slope={slope_1:.2f}, intercept={intercept_1:.2f})')

# Plotting the 45-degree line for reference
min_val = min(min(gt_volume), min(area1))
max_val = max(max(gt_volume), max(area1))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='45° Reference line')

plt.xlabel('Ground Truth Area')
plt.ylabel('Measured Area')
plt.title('Measured Areas with Fitted Lines to Ground Truth')
plt.legend()
plt.show()
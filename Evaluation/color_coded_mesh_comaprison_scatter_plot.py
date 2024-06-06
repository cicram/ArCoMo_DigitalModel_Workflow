import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from plyfile import PlyData, PlyElement
from matplotlib.cm import ScalarMappable
from scipy.stats import mannwhitneyu

# List of .ply file paths

ArCoMo_number = "300"

ply_files = [
    'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_Colored_Qaulity_No_Correction.ply',
    'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_Colored_Qaulity_Image_Correction.ply',
    'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_Colored_Qaulity_ICP_Correction.ply',
    'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_Colored_Qaulity_Overlap_Correction.ply',
    'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_Colored_Qaulity_PureCT.ply'
]

names_for_plots = ["No correction", "Image correlation", "ICP", "Overlap", "Pure CT"]
all_quality_filtered = []  # List to store filtered quality data from all files

for i, ply_file in enumerate(ply_files):
    # Load the .ply file
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
    plt.hist(quality_normalized[quality_normalized >=0.05], bins=100, color='blue', edgecolor='black')
    plt.title(f'Histrogram of vertex distances differences: ' + names_for_plots[i])
    plt.xlabel('Vertex Distance Difference [mm]')
    plt.ylabel('Number of Vertex points')

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
    ax.set_title(f'Vertex distances differences: ' + names_for_plots[i])

    # Filter out quality values below 0.05
    abs_quality = abs(quality)
    quality_filtered = abs_quality[abs_quality >= 0.05]#[(quality >= 0.05) | (quality <= -0.05)]
    all_quality_filtered.append(quality_filtered)  # Add filtered quality data to the list

plt.show()  # Show all histogram and scatter plots

# Create a combined boxplot for all .ply files
plt.figure(figsize=(12, 6))
bp = plt.boxplot(all_quality_filtered, vert=True, patch_artist=True, positions=np.arange(len(all_quality_filtered)) * 2 + 1)
plt.title('Boxplot of Vertex Distance Differences')
plt.ylabel('Vertex Distance Difference [mm]')
plt.xticks(np.arange(len(all_quality_filtered)) * 2 + 1, names_for_plots)

# Dynamically adjust and add statistical measures descriptions for each boxplot
for i, quality_filtered in enumerate(all_quality_filtered):
    # Calculate statistical measures
    min_val = np.min(quality_filtered)
    max_val = np.max(quality_filtered)
    median_val = np.median(quality_filtered)
    mean_val = np.mean(quality_filtered)
    std_dev = np.std(quality_filtered)

    # Description text
    desc = f"Min: {min_val:.2f} mm\nMax: {max_val:.2f} mm\nMedian: {median_val:.2f} mm\nMean: {mean_val:.2f} mm\nStd Dev: {std_dev:.2f} mm"

    # Calculate x position for the description text of each boxplot
    x_position = np.arange(len(all_quality_filtered)) * 2 + 1
    plt.text(x_position[i] + 0.5, plt.ylim()[1] * 0.9, desc, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()


from scipy.stats import kruskal
import scikit_posthocs as sp
import statsmodels.stats.multitest as smm

# Kruskal-Wallis Test: This non-parametric test checks if there are statistically significant 
# differences between the medians of three or more independent samples. 
# It's a good choice when your data doesn't meet the assumptions of ANOVA, such as normal distribution. 
# The test statistic (stat) and the p-value (p) are printed. If p is less than 0.05,
# it suggests that at least one sample median is different from the others.

#  Post-hoc Analysis: If the Kruskal-Wallis test indicates significant differences (p < 0.05), 
# you proceed with post-hoc pairwise comparisons to determine which specific groups differ from each other. 
# The posthoc_dunn function is used here, adjusting p-values for multiple comparisons using the Bonferroni method. 
# The results are printed, showing which pairs of groups have significantly different medians.

# Step 1: Kruskal-Wallis Test
stat, p = kruskal(*all_quality_filtered)
print(f'Kruskal-Wallis H-test test statistic: {stat:.3f}, p-value: {p:.5f}')

# Step 2: Post-hoc analysis if Kruskal-Wallis test is significant
if p < 0.05:
    # Post-hoc pairwise comparisons using Dunn's test
    ph = sp.posthoc_dunn(all_quality_filtered, p_adjust='bonferroni')
    print("Post-hoc pairwise comparisons (p-values adjusted for multiple comparisons):\n", ph)
else:
    print("No significant differences detected among groups.")


if False: 
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from plyfile import PlyData
    from matplotlib.cm import ScalarMappable

    # Load the .ply file
    plydata = PlyData.read('ArCoMo_Data/ArCoMo1400/output/ArCoMo1400_mesh_quality_cmparison_test2.ply')

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
    x = x - np.min(x)
    y = y - np.min(y)
    z = z - np.min(z)

    # Create a colormap ranging from green to red (reversed RdYlGn)
    cmap = cm.get_cmap('RdYlGn_r')

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=cmap(quality_normalized))

    # Add a colorbar
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(quality_abs), vmax=np.max(quality_abs)))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7]) # This creates a new axes for colorbar
    fig.colorbar(sm, cax=cbar_ax, label='Absolute vertex distance difference [mm]') # Here we associate colorbar with the new axes

    # Set the labels for x, y, and z axes
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')

    # Set the title for the plot
    ax.set_title('Vertex distances differences between reference mesh and reconstructed mesh')


    # Filter out zero values
    quality_filtered = quality_filtered = quality[quality >= 0.05]

    # Create a new figure for the histogram
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)  # Creating a subplot for the histogram
    plt.hist(quality_filtered, bins=100, color='blue', edgecolor='black')
    plt.title('Histogram of Vertex Distance Differences')
    plt.xlabel('Vertex Distance Difference [mm]')
    plt.ylabel('Number of Vertex points')

    # Boxplot
    plt.subplot(1, 2, 2)  # Creating a subplot for the boxplot
    box = plt.boxplot(quality_filtered, vert=True)  # Set vert=True for vertical boxplot
    plt.title('Boxplot of Vertex Distance Differences')
    plt.xlabel('')

    # Change y label
    plt.ylabel('Vertex Distance Difference [mm]')

    # Change x-axis labels to have no numbers
    plt.gca().set_xticklabels([])

    # Statistical measures
    min_val = min(quality_filtered)
    max_val = max(quality_filtered)
    median_val = np.median(quality_filtered)
    mean_val = np.mean(quality_filtered)
    std_dev = np.std(quality_filtered)

    # Add descriptions
    desc = f"Min: {min_val:.2f} mm\nMax: {max_val:.2f} mm\nMedian: {median_val:.2f} mm\nMean: {mean_val:.2f} mm\nStd Dev: {std_dev:.2f} mm"
    plt.text(0.7, 0.7, desc, fontsize=10, transform=plt.gca().transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

if False: 
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from plyfile import PlyData, PlyElement

    # Load PLY file
    plydata = PlyData.read('ArCoMo_Data/ArCoMo3/output/Test_mesh_quality.ply')

    # Extract vertex coordinates and quality values
    vertices = np.vstack([plydata['vertex'][prop] for prop in ['x', 'y', 'z']]).T
    quality = plydata['vertex']['quality']
    abs_quality = abs(quality)

    # Normalize quality values between 0 and 1
    normalized_quality = (abs_quality - np.min(abs_quality)) / (np.max(abs_quality) - np.min(abs_quality))

    # Plot mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices with colors based on quality
    colormap = cm.viridis  
    colors = colormap(normalized_quality)

    scatter = ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c=colors)

    # Create a colorbar as legend
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Quality')
    cbar.set_ticks([0, np.max(abs_quality)])  # Set colorbar ticks to 0 and max(quality)
    cbar.ax.set_ylim(0, np.max(abs_quality))  # Set colorbar limits

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
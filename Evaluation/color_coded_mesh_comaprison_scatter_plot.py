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
quality_filtered = quality

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
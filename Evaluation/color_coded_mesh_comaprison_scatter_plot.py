import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from plyfile import PlyData
from matplotlib.cm import ScalarMappable

# Load the .ply file
plydata = PlyData.read('ArCoMo_Data/ArCoMo3/output/Test_mesh_quality.ply')

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

plt.show()

# Create a histogram of the quality values
plt.figure()
plt.hist(quality, bins=100, color='blue', edgecolor='black')
plt.title('Histogram of Vertex Distance Differences')
plt.xlabel('Vertex Distance Difference [mm]')
plt.ylabel('Number of Vertex points')
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
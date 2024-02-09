import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from plyfile import PlyData
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the .ply file
plydata = PlyData.read('ArCoMo_Data/ArCoMo3/output/Test_mesh_quality.ply')

# Extract the vertex data
vertex = plydata['vertex']
x = vertex['x']
y = vertex['y']
z = vertex['z']
quality = vertex['quality']
quality = abs(quality)

# Extract the face data
face = plydata['face']
face_indices = np.vstack(face.data['vertex_indices'])

# Normalize the quality values to range [0, 1]
quality_normalized = (quality - np.min(quality)) / (np.max(quality) - np.min(quality))


# Define vertices
vertices = np.vstack([x, y, z]).T

# Create a colormap ranging from green to red (reversed RdYlGn)
cmap = cm.get_cmap('RdYlGn_r')

# Create a new figure for the mesh plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the mesh
polys = [vertices[face] for face in face_indices]
collection = Poly3DCollection(polys, linewidths=1, alpha=0.5, edgecolors='r', cmap=cmap)
face_colors = np.mean(quality_normalized[face_indices], axis=1)
collection.set_facecolor(cmap(face_colors))
ax.add_collection3d(collection)

# Create a colorbar as legend
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(quality), vmax=np.max(quality)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label('Quality')
cbar.set_ticks([0, np.max(quality)])  # Set colorbar ticks to 0 and max(quality)
cbar.ax.set_ylim(0, np.max(quality))  # Set colorbar limits

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay


# Load the global and detailed meshes
global_mesh = o3d.io.read_triangle_mesh("C:/Users/JL/Code/workflow/mesh_CT_model.stl")
detailed_mesh = o3d.io.read_triangle_mesh("C:/Users/JL/Code/workflow/mesh_OCT_model.stl")

# Assuming you've already registered and aligned the meshes

# Convert meshes to numpy arrays for easier manipulation
global_vertices = np.asarray(global_mesh.vertices)
detailed_vertices = np.asarray(detailed_mesh.vertices)

triangulation = Delaunay(global_vertices)

# Interpolate vertices in the overlapping region
overlap_vertices = np.intersect1d(triangulation.simplices, detailed_vertices, assume_unique=True)
interpolated_vertices = (global_vertices + detailed_vertices) / 2.0

# Create a new mesh with interpolated vertices
fused_mesh = o3d.geometry.TriangleMesh()
fused_mesh.vertices = o3d.utility.Vector3dVector(interpolated_vertices)
fused_mesh.triangles = o3d.utility.Vector3iVector(triangulation.simplices)

# Save or visualize the fused mesh
#o3d.io.write_triangle_mesh("fused_mesh.ply", fused_mesh)
o3d.visualization.draw_geometries([fused_mesh])
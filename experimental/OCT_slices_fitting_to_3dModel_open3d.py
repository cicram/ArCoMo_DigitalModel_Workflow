import cv2
import numpy as np
from PIL import Image
from stl import mesh
import open3d as o3d

OCT_start_frame_pullback = 7
OCT_end_frame_pullback = 8

input_file_pullback = "C:/Users/JL/ArCoMo_Data/ArCoMo3/OCT Pullback ArCoMo3 5mm.tif"
input_file_stationary = "C:/Users/JL/ArCoMo_Data/ArCoMo3/Stationary/ArCoMo3_oct_stat_mid_lad_pre_pci_blank.tif"
crop = 100

# Load the STL model
your_stl_model = mesh.Mesh.from_file('C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_3d_models/ArCoMo3_shell.stl')

# Create a Open3D TriangleMesh for the STL model
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(your_stl_model.vectors.reshape(-1, 3))
mesh_o3d.triangles = o3d.utility.Vector3iVector(np.arange(len(your_stl_model.vectors) * 3).reshape(-1, 3))

point_cloud = o3d.geometry.PointCloud()

# Load images
image_list = []
with Image.open(input_file_pullback) as im_pullback:
    for page_pullback in range(OCT_start_frame_pullback, OCT_end_frame_pullback, 1):
        # Load an image using OpenCV
        im_pullback.seek(page_pullback)  # Move to the current page (frame)
        image_pullback = np.array(im_pullback.convert('RGB'))  # Convert PIL image to NumPy array
        image_pullback = image_pullback[:, :, ::-1].copy()  # Crop image at xxx pixels from top

        # Create an Open3D PointCloud representing the image
        height, width, _ = image_pullback.shape
        x = np.arange(width)
        y = np.arange(height)
        z = np.full_like(y, x)  # Assign Z-coordinate

        # Convert image coordinates to 3D points
        points = np.column_stack((x, y, z))
        point_cloud.points = o3d.utility.Vector3dVector(points)

# Visualize the 3D model and the images
o3d.visualization.draw_geometries([mesh_o3d, point_cloud])
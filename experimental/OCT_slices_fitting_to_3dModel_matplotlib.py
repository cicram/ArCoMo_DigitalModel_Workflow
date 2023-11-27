import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import PolyCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from io import open

OCT_start_frame_pullback = 4
OCT_end_frame_pullback = 8

input_file_pullback = "C:/Users/JL/ArCoMo_Data/ArCoMo3/OCT Pullback ArCoMo3 5mm.tif"
input_file_stationary = "C:/Users/JL/ArCoMo_Data/ArCoMo3/Stationary/ArCoMo3_oct_stat_mid_lad_pre_pci_blank.tif"
crop = 100

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Load images
with Image.open(input_file_pullback) as im_pullback:
    for page_pullback in range(OCT_start_frame_pullback, OCT_end_frame_pullback, 1):
        # Load an image using OpenCV
        im_pullback.seek(page_pullback)  # Move to the current page (frame)
        image_pullback = np.array(im_pullback.convert('RGB'))  # Convert PIL image to NumPy array
        image_pullback = image_pullback[:-crop, :, ::-1].copy()  # Crop image at xxx pixels from top

        gray_image = cv2.cvtColor(image_pullback, cv2.COLOR_BGR2GRAY)

        # Get image dimensions
        height, width = gray_image.shape

        # Create a meshgrid for 3D plotting
        x = np.arange(- width/2, width/2, 1)
        y = np.arange(- height/2, height/2, 1)
        x, y = np.meshgrid(x, y)
        skipping = 1
        x = x[::skipping, ::skipping] / 98
        y = y[::skipping, ::skipping] / 98


        # Create a 3D plot
        
        z = np.ones_like(x) * 1700

        surf = ax.plot_surface(x, y, z, facecolors=plt.cm.gray(gray_image[::skipping, ::skipping]), rstride=5, cstride=5, antialiased=True)


def read_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line[2:].split())))
            elif line.startswith('f '):
                faces.append([int(vertex.split('/')[0]) - 1 for vertex in line[2:].split()])

    return np.array(vertices), np.array(faces)

def plot_obj(file_path, ax):
    vertices, faces = read_obj(file_path)

    # Plot the vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', marker='o')

    # Triangulate the faces and plot them
    for face in faces:
            vertices_face = vertices[face]
            poly3d = [[tuple(vertices_face[i])] for i in range(len(vertices_face))]
            poly3d_collection = Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1)
            ax.add_collection3d(poly3d_collection)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

obj_file_path = 'C:/Users/JL/NX_parts/ArCoMo3_3d.obj'
plot_obj(obj_file_path, ax)
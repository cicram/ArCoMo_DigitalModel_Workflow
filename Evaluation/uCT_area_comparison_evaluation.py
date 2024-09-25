import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
import os
from plyfile import PlyData
import pyvista as pv
from scipy.spatial import KDTree
import trimesh
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon
import csv 
import pandas as pd
from scipy.stats import linregress, spearmanr
import json
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon

def interpolate_points(arr, points):
    original_points = np.arange(len(arr))
    target_points = np.linspace(0, len(arr) - 1, points)

    # Interpolate to exactly 100 points
    f = interp1d(original_points, arr, kind='linear')
    interpolated_array = f(target_points)

    return interpolated_array

def compute_area(points, centerline_point):
    # Fit a spline to the points
    tck, u = splprep(np.array(points).T, s=0, per=True)

    # Fit a spline to the points
    tck, u = splprep(np.array(points).T, s=0, per=True)
    
    # Evaluate the spline at many points to form a polygon
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_spline, y_spline = splev(u_new, tck, der=0)
    
    # Create a Polygon object from the spline points
    polygon = Polygon(np.column_stack((x_spline, y_spline)))
    
    # Compute the area within the polygon using Shapely
    area_within_circle = polygon.area
    
    return area_within_circle, x_spline, y_spline

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

class InteractivePointFilter:
    def __init__(self, points, centerline_point):
        self.points = points
        self.centerline_point = centerline_point
        self.selected_points = None
        self.filtered_points = self.points
        # Create a plot with the points
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter(points[:, 0], points[:, 1], s=30)
        self.ax.scatter(self.centerline_point[0], self.centerline_point[1], color="red")

        # Initialize the PolygonSelector
        self.selector = PolygonSelector(self.ax, self.onselect, useblit=True)
        
        # Show the plot
        plt.show()

    def onselect(self, vertices):
        """
        This function is called when the polygon is completed.
        It filters out points inside the polygon.
        """
        path = Path(vertices)
        
        # Check which points are inside the polygon
        inside = path.contains_points(self.points[:, :2])

        # Filter out points inside the polygon
        self.selected_points = self.points[inside]
        self.filtered_points = self.points[~inside]
        
        # Update the plot to show only filtered points
        self.scatter.set_offsets(self.filtered_points[:, :2])

        self.fig.canvas.draw()
###########################################################################################################

ArCoMo_number = "1"
ArCoMo_number_gt = "100"
# SELECT OCT PART WITH INDEXES
#start_idx = 300 #Ar900: 105 #Ar1500: 90 #Ar1300: 170 #Ar1100: 300 #Ar1000: 260 #Ar300: 620, Ar200: 150, Ar100: 58
#end_idx = 600 #Ar900: 405 #Ar1500: 390 #Ar1300: 480 #Ar1100: 600 #Ar1000: 560 #Ar300: 950, Ar200: 410, Ar100: 266

ply_file_12_bar = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_12bar_segmentation/balloon_12bar_Ballon.ply'
ply_file_18_bar = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_18bar_segmentation/balloon_18bar_Ballon.ply'
ply_file_4_bar = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/balloon_4bar_Calc.ply'
ply_file_16_bar = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_16bar_segmentation/balloon_16bar_Ballon.ply'
ply_file_8_bar = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_8bar_segmentation/balloon_8bar_Ballon.ply'
ply_file_20_bar = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_20bar_segmentation/balloon_20bar_Ballon.ply'
centerline_file = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_12bar_segmentation/centerline.json'


if True:
    # Read in index values
    main_branch_start_idx = None
    oct_start = None
    oct_end = None
    oct_registration = None
    centerline_resampling_distance = 0.2

    all_quality_filtered = []  # List to store filtered quality data from all files

    
    z_spacing = 0.0172
    start_cut = 100
    end_cut = 500

    cutting_points = np.linspace(z_spacing*start_cut, z_spacing*end_cut, 20)
        
    # Load the .ply file
    print("load mesh")

    plydata = PlyData.read(ply_file_4_bar)

    # Extract the vertex data
    vertex = plydata['vertex']
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    mesh = trimesh.load_mesh(ply_file_4_bar)

    print("mesh loaded")

    # Create a KDTree for efficient nearest neighbor search
    mesh_points = np.vstack((x, y, z)).T
    kd_tree = KDTree(mesh_points)

    radius = 1
    data_model = []
    data_model.append(["z-height", "Area", "Filter radius"])
    data_gt = []
    data_gt.append(["z-height", "Area", "Filter radius"])

    points_3d = []
    plt_idx = 0
    interp_points = 20
    are_threshold = 0.2
    start_idx = 10
    end_idx = 20
    area_all = []
    colors_all = []
    filtered_points_all = []
    centerline_points_2d = []

    # Loop over each centerline point to extract and plot the 2D points
    for idx, val in enumerate(cutting_points):
        flag_except = 0
        try:
            # Get the coordinates of the points on the centerline
            cut_point = np.array([5, 5, cutting_points[idx]])
            
            normal = [0, 0, 1] #centerline_points[idx] - centerline_points[idx-1]
            
            # Get the Path3D object of the intersection
            path3D = mesh.section(plane_origin=cut_point, plane_normal=normal)

            # Extract the points from the Path3D object
            points = np.array([path3D.vertices[i] for entity in path3D.entities for i in entity.points])
            
            # Filter points within the radius
            point_filter = InteractivePointFilter(points, cut_point) 
            filtered_points = point_filter.filtered_points
            filtered_points_all.append(filtered_points)
           
            points_2d = []
            for point in filtered_points:
                points_2d.append([point[0], point[1]])

            area, x_spline_points, y_spline_points = compute_area(points_2d, cut_point)
            area_all.append(area)

            for x, y in zip(x_spline_points, y_spline_points):
                points_3d.append([x, y, cutting_points[idx]])
            
            centerline_points_2d.append([cut_point[0], cut_point[1], idx])
            
            print(area)

        except: 
            flag_except = 1
            
        if flag_except:
            data_model.append([cutting_points[idx], np.nan, radius])
        else:
            data_model.append([cutting_points[idx], area, radius])


    ####################################################
    # Plot the original mesh, centerline, and clipping planes
    p = pv.Plotter()

    for idx, fil_point in enumerate(filtered_points_all):
        if idx >= 1 and idx % 13 == 0:
            polydata3 = pv.PolyData(fil_point)
            p.add_mesh(polydata3, color="yellow")
            cut_point = np.array([5, 5, cutting_points[idx]])
            normal = [0, 0, 1]
            plane_polydata = pv.Plane(center=cut_point, direction=normal, i_size=20, j_size=20, i_resolution=100, j_resolution=100)
            p.add_mesh(plane_polydata, color="red", opacity=0.5)
    
    # Add original mesh
    p.add_mesh(mesh, color="green", opacity=0.3, show_edges=True)

    p.show()


centerline_points_2d_file = 'C:/Users/JL/Code/uCT/centerline.csv'

if False:
    with open(centerline_points_2d_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(centerline_points_2d)

spline_points_18bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_18bar_segmentation/output_spline_points_18_bar.csv'
spline_points_12bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_12bar_segmentation/output_spline_points_12_bar.csv'
spline_points_4bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_spline_points_4_bar.csv'
spline_points_8bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_8bar_segmentation/output_spline_points_8_bar.csv'
spline_points_16bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_16bar_segmentation/output_spline_points_16_bar.csv'
spline_points_20bar_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_20bar_segmentation/output_spline_points_20_bar.csv'

if False:
    with open(spline_points_4bar_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(points_3d)


bar_18_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_18bar_segmentation/output_areas_18_bar.csv'
bar_12_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_12bar_segmentation/output_areas_12_bar.csv'
bar_4_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_4bar_segmentation/output_areas_4_bar.csv'
bar_8_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_8bar_segmentation/output_areas_8_bar.csv'
bar_16_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_16bar_segmentation/output_areas_16_bar.csv'
bar_20_csv = 'C:/Users/JL/Code/uCT/BIO18_LAD_ballon_20bar_segmentation/output_areas_20_bar.csv'

# Save areas to csv file
if False:
    with open(bar_4_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_model)


######################################################################################################################################
########################### Area visualisation (linear regression) ######################################################################
######################################################################################################################################


df_18bar = pd.read_csv(bar_18_csv)
df_12bar = pd.read_csv(bar_12_csv)
df_4bar = pd.read_csv(bar_4_csv)

colors = ['black', 'blue', 'green']

fig = plt.figure()
plt.plot(df_4bar['z-height'], df_4bar['Area'], marker='o', linestyle='-', color=colors[0], label='4 bar')
plt.plot(df_12bar['z-height'], df_12bar['Area'], marker='o', linestyle='-', color=colors[1], label='12 bar')
plt.plot(df_18bar['z-height'], df_18bar['Area'], marker='o', linestyle='-', color=colors[2], label='18 bar')

plt.legend()
plt.xlabel('z-height')
plt.ylabel('Area')
plt.title('z-height vs Area')
plt.grid(True)

plt.show()

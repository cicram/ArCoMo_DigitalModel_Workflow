import numpy as np
import pyvista as pv
import pymeshfix
from scipy.spatial.distance import cdist
import csv 
import trimesh

def parse_point_cloud_centerline(file_path):
    flag_header = False
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                if parts[0] == "[Main" and flag_header:
                    flag_header = False
                if flag_header:
                    px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                    data.append((px, py, pz))
                if parts[0] == "Px" and parts[1] == "Py" and parts[2] == "Pz":
                    flag_header = True

    data = np.array(data)
    return data

def resample_center_line(center_line, z_distance):
    # Desired distance between resampled points
    interval_distance = z_distance

    # Initialize variables
    resampled_points = [center_line[0]]  # Start with the first point
    current_position = center_line[0]
    current_index = 0

    while current_index < len(center_line) - 1:
        next_position = center_line[current_index + 1]
        direction_vector = next_position - current_position
        # get length between points
        direction_length = np.linalg.norm(direction_vector)

        # Check if we need to insert points along the segment, if segment is finished, do not interpolate anymore (distance > xxx)
        if direction_length <= 4.0:
            while direction_length >= interval_distance:
                # Calculate the next resampled point and add it
                t = interval_distance / direction_length
                new_point = current_position + t * direction_vector
                resampled_points.append(new_point)

                # Update the current position and remaining length
                current_position = new_point
                direction_vector = next_position - current_position
                direction_length = np.linalg.norm(direction_vector)
        else:
            # Skip interpolation and move to the next point
            resampled_points.append(next_position)
            current_position = next_position

        # Move to the next point
        current_index += 1

    # Add the last point of the original centerline
    resampled_points.append(center_line[-1])

    # Convert the resampled points to a NumPy array
    resampled_points = np.array(resampled_points)

    return resampled_points

def filter_points_within_radius(points, center_point, radius):
    """
    Filter points that are within a certain radius from a center point.
    """
    # Calculate the distance of each point from the center point
    distances = np.linalg.norm(points - center_point, axis=1)
    
    # Create a mask to identify points within the radius
    mask = distances <= radius
    
    # Extract the filtered points using the mask
    filtered_points = points[mask]
    
    return filtered_points

def project_point_to_plane(point, origin, normal):
    point = np.array(point)
    origin = np.array(origin)
    normal = np.array(normal)
    return point - np.dot((point - origin), normal) * normal

def compute_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area

# Load mesh
#mesh_file = "ArCoMo_Data/ArCoMo3/output/ArCoMo3_shell.stl"
mesh_file = "C:/Users/JL/Model_evaluation/ArCoMo300/_Correction.ply"
title_csv = "C:/Users/JL/Model_evaluation/AreaVolumeResults/ArCoMo1400_Colored_Qaulity_PureCT.csv"

# Intersect the mesh with the plane
mesh = trimesh.load_mesh(mesh_file)

# Load centerline
centerline_file = "ArCoMo_Data/ArCoMo3/ArCoMo3_centerline.txt"
centerline_points = parse_point_cloud_centerline(centerline_file)

centerline_points = resample_center_line(centerline_points, 0.2)

radius = 3

data = []
data.append(["Centerline IDX", "Area", "Filter radius"])

for idx, val in enumerate(centerline_points):
    if idx == 650:     
        flag_except = 0
        try:
            # Get the coordinates of the points on the centerline
            cut_point_3 = centerline_points[idx]

            normal_3 = centerline_points[idx] - centerline_points[idx-1]

            plane_origin_3 = cut_point_3

            plane_polydata_3 = pv.Plane(center=plane_origin_3, direction=normal_3, i_size=20, j_size=20, i_resolution=100, j_resolution=100)

            # Get the Path3D object of the intersection
            path3D = mesh.section(plane_origin=plane_origin_3, plane_normal=normal_3)

            # Extract the points from the Path3D object
            points = np.array([path3D.vertices[i] for entity in path3D.entities for i in entity.points])
            
            filtered_points = filter_points_within_radius(points, cut_point_3, 3)

            points_2d = [project_point_to_plane(p, cut_point_3, normal_3)[:2] for p in filtered_points]

            area = compute_area(points_2d)
            print(area)
        except: 
            flag_except = 1
            
        if flag_except:
            data.append([idx, np.nan, radius])
        else:
            data.append([idx, area, radius])

if False:
    # Write data to CSV file
    with open(title_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if True:
    import matplotlib.pyplot as plt
    x = [point[0] for point in points_2d]
    y = [point[1] for point in points_2d]

    # Plot the points
    plt.plot(x, y)
    plt.show()
    print(compute_area(points_2d))

    # Create a PolyData object from the points
    polydata = pv.PolyData(filtered_points)


    ####################################################
    # Plot the original mesh, centerline, and both clipping planes
    p = pv.Plotter()

    # Add the intersection
    p.add_mesh(polydata, color="yellow")

    # Add original mesh
    p.add_mesh(mesh, color="green", opacity=0.3, show_edges=True)

    # Add clipped mesh

    # Add centerline
    p.add_points(centerline_points[idx], color="blue", point_size=5)

    # Add first clipping plane
    p.add_mesh(plane_polydata_3, color="red", opacity=0.5)

    # Add second clipping plane

    p.show()

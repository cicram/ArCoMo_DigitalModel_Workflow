import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist
import pyvista as pv
from pymeshfix import MeshFix
import csv

def point_to_line_distance(p0, p1, p):
    """
    Calculate the distance between a point and a line segment.
    """
    v = p1 - p0
    w = p - p0
    c1 = np.dot(w, v)
    c2 = np.dot(v, v)
    if c1 <= 0:
        return np.linalg.norm(p - p0)
    if c2 <= c1:
        return np.linalg.norm(p - p1)
    b = c1 / c2
    pb = p0 + b * v
    return np.linalg.norm(p - pb)

# Function to remove points from mesh that are not within a spatial radius from the line segment
def filter_points_within_radius(mesh, centerline_points, cut_point_1, cut_point_2, radius):
    # Compute the direction vector of the line segment between cut_point_1 and cut_point_2
    line_direction = cut_point_2 - cut_point_1
    # Normalize the direction vector
    line_direction /= np.linalg.norm(line_direction)
    
    # Calculate the distance of each point in the mesh from the line segment
    distances = np.array([point_to_line_distance(cut_point_1, cut_point_2, p) for p in mesh.points])
    
    # Create a mask to identify points within the spatial radius
    mask = distances <= radius
    
    # Extract the filtered points from the mesh using the mask
    filtered_mesh = mesh.extract_points(mask)
    
    return filtered_mesh

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

# Load your mesh from a .ply file
#mesh_file = "ArCoMo_Data/ArCoMo3/output/ArCoMo3_shell.stl"
mesh_file = "C:/Users/JL/Model_evaluation/ArCoMo300/ArCoMo300_Colored_Qaulity_ICP_Correction.ply"

mesh = pv.read(mesh_file)

# Load centerline
centerline_file = "ArCoMo_Data/ArCoMo3/ArCoMo3_centerline.txt"
centerline_points = parse_point_cloud_centerline(centerline_file)

centerline_points = resample_center_line(centerline_points, 0.2)

distance = 15 # = 3mm
radius = 3

data = []
data.append(["Centerline IDX", "Volume", "Distance", "Filter radius"])
for idx, val in enumerate(centerline_points):
    print(idx)
    if idx > distance and idx < (len(centerline_points) - distance - 1):
        flag_except = 0
        try: 
            # Step 1: Identify the points on the centerline for the cutting planes
            cut_point_index_2 = idx  # Index of the first point on the centerline
            cut_point_index_1 = idx + distance  # Index of the second point on the centerline

            # Get the coordinates of the points on the centerline
            cut_point_1 = centerline_points[cut_point_index_1]
            cut_point_2 = centerline_points[cut_point_index_2]

            # Compute the normal vectors for the clipping planes
            normal_1 = centerline_points[cut_point_index_1] - centerline_points[cut_point_index_1-1]
            normal_2 = -normal_1

            # Step 2: Define the first cutting plane using the first cut point and computed normal
            plane_origin_1 = cut_point_1
            plane_normal_1 = normal_1

            # Create the first plane polydata using the origin and normal vector of the first plane
            plane_polydata_1 = pv.Plane(center=plane_origin_1, direction=plane_normal_1)

            # Step 3: Define the second cutting plane using the second cut point and computed normal
            plane_origin_2 = cut_point_2
            plane_normal_2 = normal_2

            # Create the second plane polydata using the origin and normal vector of the second plane
            plane_polydata_2 = pv.Plane(center=plane_origin_2, direction=plane_normal_2)

            # Clip the mesh with both planes to obtain the volume of interest
            clipped_mesh_1 = mesh.clip_surface(plane_polydata_1)
            clipped_mesh_2 = clipped_mesh_1.clip_surface(plane_polydata_2)

            filtered_mesh = filter_points_within_radius(clipped_mesh_2, centerline_points, cut_point_1, cut_point_2, radius)

            # Extract the surface of the UnstructuredGrid to get a PolyData object
            filtered_polydata = filtered_mesh.extract_surface()

            # Convert the pyvista mesh to a format that MeshFix can understand
            vertices = filtered_polydata.points
            faces = filtered_polydata.faces.reshape((-1, 4))[:, 1:]

            # Create a MeshFix object
            fixer = MeshFix(vertices, faces)

            # Repair the mesh
            fixer.repair()

            # Fill the holes
            #fixer.fill_holes()

            # Get the repaired mesh
            repaired = fixer.mesh

            # Compute the volume of the original mesh
            original_volume = mesh.volume

            # Compute the volume of the filtered mesh
            filtered_mesh_volume = repaired.volume
            print("Filtered mesh volume: {:.3f}".format(filtered_mesh_volume))

        except: 
            flag_except = 1
        
        if flag_except:
            data.append([idx, np.nan, distance, radius])
        else:
            data.append([idx, filtered_mesh_volume, distance, radius])
if True:
    # Write data to CSV file
    title_csv = 'data_volume2.csv'
    with open(title_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if False: 
    print("Original mesh volume: {:.3f}".format(original_volume))
    print("Filtered mesh volume: {:.3f}".format(filtered_mesh_volume))

    # Plot the original mesh, centerline, and both clipping planes
    p = pv.Plotter()

    # Add original mesh
    p.add_mesh(mesh, color="green", opacity=0.3, show_edges=True)


    # Add second clipping plane
    p.add_mesh(repaired, color="red", opacity=0.8, show_edges=True)

    # Add clipped mesh
    #p.add_mesh(filtered_mesh, color="blue", opacity=0.5, show_edges=True)

    # Add centerline
    p.add_points(centerline_points, color="green", point_size=5)

    # Add first clipping plane
    p.add_mesh(plane_polydata_1, color="yellow", opacity=0.8)

    # Add second clipping plane
    p.add_mesh(plane_polydata_2, color="black", opacity=0.8)

    p.show()

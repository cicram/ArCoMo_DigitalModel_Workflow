import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist
import pyvista as pv
from pymeshfix import MeshFix
import csv
import matplotlib.pyplot as plt

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

def compute_volumes(centerline_points, mesh):
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
                normal_1 = centerline_points[cut_point_index_1] - centerline_points[cut_point_index_1 - 1]
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

                filtered_mesh = filter_points_within_radius(clipped_mesh_2, centerline_points_gt, cut_point_1,
                                                            cut_point_2, radius)

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
                # fixer.fill_holes()

                # Get the repaired mesh
                repaired = fixer.mesh

                # Compute the volume of the original mesh
                original_volume = mesh.volume

                # Compute the volume of the filtered mesh
                filtered_mesh_volume = repaired.volume
                #print("Filtered mesh volume: {:.3f}".format(filtered_mesh_volume))

            except:
                flag_except = 1

            if flag_except:
                data.append([idx, np.nan, distance, radius])
            else:
                data.append([idx, filtered_mesh_volume, distance, radius])

    return data
    
# List of mesh files
mesh_file_path_gt = "C:/Users/JL/Model_evaluation/GroundTruth/ArCoMo14_innershell_ply.ply"
mesh_file_path_ct = "C:/Users/JL/Model_evaluation/phantom_CT/ArCoMo1400/ArCoMo1400_CT.ply"

# List of CSV titles
csv_file_gt = "C:/Users/JL/Model_evaluation/phantom_CT/ArCoMo1400/volumes_result_gt.csv"
csv_file_ct = "C:/Users/JL/Model_evaluation/phantom_CT/ArCoMo1400/volumes_result_ct.csv"

mesh_gt = pv.read(mesh_file_path_gt)
mesh_ct = pv.read(mesh_file_path_ct)

# Load centerline
centerline_file_gt = "ArCoMo_Data/ArCoMo14/ArCoMo14_centerline.txt"
centerline_file_ct = "C:/Users/JL/Model_evaluation/phantom_CT/ArCoMo1400/ArCoMo1400_centerline.txt"

centerline_points_gt = parse_point_cloud_centerline(centerline_file_gt)
centerline_points_ct = parse_point_cloud_centerline(centerline_file_ct)

centerline_points_gt = resample_center_line(centerline_points_gt, 0.2)
#centerline_points_ct = resample_center_line(centerline_points_ct, 0.2)

distance = 15  # = 3mm
radius = 3

#data_gt = compute_volumes(centerline_points_gt, mesh_gt)
#data_ct = compute_volumes(centerline_points_ct, mesh_ct)


# Function to display the points and allow selection
def select_centerline_point(centerline_points, scale_range_flag):
    selected_index_final = None

    def on_pick(event):
        global selected_index
        ind = event.ind[0]
        selected_index = ind

        # Update colors
        colors = ['blue'] * len(centerline_points)
        colors[selected_index] = 'red'
        sc = ax.scatter(centerline_points[:, 0], centerline_points[:, 1], centerline_points[:, 2], c=colors, picker=True)

        plt.draw()
        print(f"Selected index: {selected_index}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if scale_range_flag:
        max_range = np.array([centerline_points_ct[:, 0].max() - centerline_points_ct[:, 0].min(), 
                            centerline_points_ct[:, 1].max() - centerline_points_ct[:, 1].min(), 
                            centerline_points_ct[:, 2].max() - centerline_points_ct[:, 2].min()]).max() / 2.0

        mid_x = (centerline_points_ct[:, 0].max() + centerline_points_ct[:, 0].min()) * 0.5
        mid_y = (centerline_points_ct[:, 1].max() + centerline_points_ct[:, 1].min()) * 0.5
        mid_z = (centerline_points_ct[:, 2].max() + centerline_points_ct[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # Plot the points
    sc = ax.scatter(centerline_points[:, 0], centerline_points[:, 1], centerline_points[:, 2], c='blue', picker=True)

    # Connect the pick event to the on_pick function
    fig.canvas.mpl_connect('pick_event', on_pick)

    # Show the plot
    plt.show()

    # Output the selected index
    print(selected_index_final)
    return selected_index_final


select_centerline_point(centerline_points_gt, 0)
select_centerline_point(centerline_points_ct, 1)

# plot
if False:
    # Extracting volume values
    volumes_gt = [row[1] for row in data_gt[1:]]
    volumes_ct = [row[1] for row in data_ct[1:]]

    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(volumes_gt, marker='o', linestyle='-', label='GT Volumes')
    plt.plot(volumes_ct, marker='s', linestyle='-', label='CT Volumes')
    plt.title('Volume Comparison')
    plt.xlabel('Index')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.show()

if False:
    # Write data to CSV file
    with open(csv_file_gt, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_gt)

    with open(csv_file_ct, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_ct)


if False: 
    print("Original mesh volume: {:.3f}".format(original_volume))
    print("Filtered mesh volume: {:.3f}".format(filtered_mesh_volume))

    # Plot the original mesh, centerline, and both clipping planes
    p = pv.Plotter()

    # Add original mesh
    p.add_mesh(mesh, color="green", opacity=0.3, show_edges=True)


    # Add second clipping plane
    p.add_mesh(repaired, color="red", opacity=1, show_edges=True)

    # Add clipped mesh
    #p.add_mesh(filtered_mesh, color="blue", opacity=0.5, show_edges=True)

    # Add centerline
    p.add_points(centerline_points[idx], color="green", point_size=5)

    # Add first clipping plane
    #p.add_mesh(plane_polydata_1, color="yellow", opacity=0.8)

    # Add second clipping plane
    #p.add_mesh(plane_polydata_2, color="black", opacity=0.8)

    p.show()

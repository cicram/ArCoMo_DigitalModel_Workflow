import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from plyfile import PlyData
import pyvista as pv
from scipy.spatial import KDTree
import trimesh
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon

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

def interpolate_points(arr, points):
    original_points = np.arange(len(arr))
    target_points = np.linspace(0, len(arr) - 1, points)

    # Interpolate to exactly 100 points
    f = interp1d(original_points, arr, kind='linear')
    interpolated_array = f(target_points)

    return interpolated_array

def project_point_to_plane(point, origin, normal):
    point = np.array(point)
    origin = np.array(origin)
    normal = np.array(normal)
    return point - np.dot((point - origin), normal) * normal

def compute_area(points):
    # Fit a spline to the points
    tck, u = splprep(np.array(points).T, s=0, per=True)

    # Evaluate the spline at many points to form a polygon
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_spline, y_spline = splev(u_new, tck, der=0)
    # x_plt = [point[0] for point in points]
    # y_plt = [point[1] for point in points]
    # plt.plot(x_spline, y_spline)
    # plt.plot(x_plt, y_plt, "x")
    # plt.show()
    # Create a Polygon object from the spline points
    polygon = Polygon(np.column_stack((x_spline, y_spline)))

    # Compute the area within the polygon using Shapely
    area_within_circle = polygon.area

    return area_within_circle

# List of .ply file paths

ArCoMo_number = "300"

ply_file = 'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_Colored_Qaulity_Overlap_Correction.ply' 

all_quality_filtered = []  # List to store filtered quality data from all files

# Load the .ply file
plydata = PlyData.read(ply_file)

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
x_shifted = x - np.min(x)
y_shifted = y - np.min(y)
z_shifted = z - np.min(z)

# Create a colormap ranging from green to red (reversed RdYlGn)
cmap = cm.get_cmap('RdYlGn_r')

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x_shifted, y_shifted, z_shifted, c=cmap(quality_normalized))

# Add a colorbar
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(quality_abs), vmax=np.max(quality_abs)))
sm.set_array([])
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
fig.colorbar(sm, cax=cbar_ax, label='Absolute vertex distance difference [mm]')

# Set the labels for x, y, and z axes
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')

# Set the title for the plot
ax.set_title('Vertex distances differences')

# Filter out quality values below 0.05
abs_quality = abs(quality)
quality_filtered = abs_quality[abs_quality >= 0.05]
all_quality_filtered.append(quality_filtered)  # Add filtered quality data to the list

plt.show()  # Show all histogram and scatter plots

###########################################################
# Create 2d color coded plot
# Intersect the mesh with the plane
mesh = trimesh.load_mesh(ply_file)

# Load centerline
centerline_file = "ArCoMo_Data/ArCoMo3/ArCoMo3_centerline.txt"
centerline_points = parse_point_cloud_centerline(centerline_file)

centerline_points = resample_center_line(centerline_points, 0.2)

# Create a KDTree for efficient nearest neighbor search
mesh_points = np.vstack((x, y, z)).T
kd_tree = KDTree(mesh_points)

radius = 2

end_idx = 950
start_idx = 620
plt_idx = 0
interp_points = 20
are_threshold = 0.2
area_all = []
colors_all = []
filtered_points_all = []
# Loop over each centerline point to extract and plot the 2D points
for idx, val in enumerate(centerline_points):
    if idx >= start_idx and idx <= end_idx:
        try:
            # Get the coordinates of the points on the centerline
            cut_point = centerline_points[idx]
            
            normal = centerline_points[idx] - centerline_points[idx-1]
            
            # Get the Path3D object of the intersection
            path3D = mesh.section(plane_origin=cut_point, plane_normal=normal)

            # Extract the points from the Path3D object
            points = np.array([path3D.vertices[i] for entity in path3D.entities for i in entity.points])
            
            # Filter points within the radius
            filtered_points = filter_points_within_radius(points, cut_point, radius)

            if idx == start_idx:
                plane_polydata_start = pv.Plane(center=cut_point, direction=normal, i_size=20, j_size=20, i_resolution=100, j_resolution=100)
                filtered_points_start = filtered_points
            if idx == end_idx:
                plane_polydata_end = pv.Plane(center=cut_point, direction=normal, i_size=20, j_size=20, i_resolution=100, j_resolution=100)
                filtered_points_end = filtered_points

            points_2d = [project_point_to_plane(p, cut_point, normal)[:2] for p in filtered_points]
            x_plt = [point[0] for point in filtered_points]
            y_plt = [point[1] for point in filtered_points]
            
            area = compute_area(points_2d)
            print(area)
            if area > are_threshold:
                area_all.append(area)
                filtered_points_all.append(filtered_points)
            
                # Find the closest mesh point to each filtered point to get the quality value
                distances, indices = kd_tree.query(filtered_points)

                filtered_points_quality = quality_normalized[indices]

                filtered_points_quality_interpolated = interpolate_points(filtered_points_quality, interp_points)

                colors = cmap(filtered_points_quality_interpolated)
                colors_all.append(colors)
        except: 
            pass

# Normalize area
area_all = np.array(area_all)
area_all_normalized = (area_all - np.min(area_all)) / (np.max(area_all) - np.min(area_all))
# Create a figure for the 2D plot
fig, ax = plt.subplots()
# Plot each square
for idx, area in enumerate(area_all_normalized):
    colors = colors_all[idx]
    for i, color in enumerate(colors):
        #square = plt.Rectangle((i*area-interp_points/2*area, plt_idx), area, 1, color=color)
        square = plt.Rectangle((i-interp_points/2, plt_idx), 1, 1, color=color)

        ax.add_patch(square)
    plt_idx = plt_idx + 1

# Set the limits of the plot
ax.set_xlim(-interp_points/2, interp_points/2)
ax.set_ylim(0, plt_idx)

# Remove the axes for a cleaner look
ax.axis('off')
# Add a colorbar to the 2D plot
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(quality_abs), vmax=np.max(quality_abs)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Absolute vertex distance difference [mm]')

# Set labels and title for the 2D plot
ax.set_xlabel('Index')
ax.set_ylabel('Centerline Path Index')
ax.set_title('2D Projection of Vertex Distance Differences (Indexed)')

# Show the 2D plot
plt.show()


# Create a PolyData object from the points
polydata2 = pv.PolyData(filtered_points_end)
polydata = pv.PolyData(filtered_points_start)

####################################################
# Plot the original mesh, centerline, and both clipping planes
p = pv.Plotter()

for fil_point in filtered_points_all:
    polydata3 = pv.PolyData(fil_point)
    p.add_mesh(polydata3, color="yellow")

# Add the intersection
p.add_mesh(polydata, color="yellow")
p.add_mesh(polydata2, color="yellow")

# Add original mesh
p.add_mesh(mesh, color="green", opacity=0.3, show_edges=True)

# Add clipped mesh

# Add first clipping plane
p.add_mesh(plane_polydata_start, color="red", opacity=0.5)
p.add_mesh(plane_polydata_end, color="blue", opacity=0.5)

# Add second clipping plane

p.show()
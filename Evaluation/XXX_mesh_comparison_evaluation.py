import numpy as np
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

def create_folder(folder_path):
    # Creates a folder if it doesn't exist.

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

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
    # Convert inputs to numpy arrays
    point = np.array(point)
    origin = np.array(origin)
    normal = np.array(normal)
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Compute the projection of the point onto the plane
    return point - np.dot((point - origin), normal) * normal

def interpolate_points(arr, points):
    original_points = np.arange(len(arr))
    target_points = np.linspace(0, len(arr) - 1, points)

    # Interpolate to exactly 100 points
    f = interp1d(original_points, arr, kind='linear')
    interpolated_array = f(target_points)

    return interpolated_array

def compute_area(points):
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
    
    # Plotting the points, spline, and polygon
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]
    
    if False:
        plt.figure(figsize=(8, 8))
        plt.plot(x_spline, y_spline, label='Fitted Spline', color='blue')
        plt.plot(x_points, y_points, 'x', label='Original Points', color='red')
        plt.fill(x_spline, y_spline, alpha=0.3, color='blue')
        
        # Annotate the area in the center
        center_x, center_y = polygon.centroid.x, polygon.centroid.y
        plt.text(center_x, center_y, f'Area: {area_within_circle:.2f} mm²', fontsize=12, ha='center')
        
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.legend()
        plt.title('Spline Fit and Enclosed Area')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    
    return area_within_circle

def convert_to_2d(points_3d, origin, normal):
    if False:
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='blue', label='Original 3D Points')
        plt.show()
    # Find two vectors in the plane
    normal = normal / np.linalg.norm(normal)
    # Vector in the plane (not parallel to normal)
    v1 = np.array([1, 0, 0]) if normal[0] == 0 else np.array([0, 1, 0])
    # Basis vector u in the plane
    u = np.cross(normal, v1)
    u /= np.linalg.norm(u)
    # Basis vector v in the plane
    v = np.cross(normal, u)
    
    # Project each 3D point to 2D in the plane's coordinate system
    points_2d = []
    for point in points_3d:
        point_proj = project_point_to_plane(point, origin, normal)
        # Calculate 2D coordinates in the plane's basis
        d = point_proj - origin
        point_2d = [np.dot(d, u), np.dot(d, v)]
        points_2d.append(point_2d)
    
    # Plotting the points, spline, and polygon
    x_points = [point[0] for point in points_2d]
    y_points = [point[1] for point in points_2d]
    #plt.plot(x_points, y_points, 'x', label='Original Points', color='red')
    #plt.show()

    return points_2d


###########################################################################################################
# CHANGE NUMBER HERE AND FOLDER LOCATION
ArCoMo_numbers = ["900", "1000", "1100", "1300", "1500"]
ArCoMo_numbers_gt = ["9", "10", "11", "13", "15"]

for itr in range(len(ArCoMo_numbers)):
    ArCoMo_number = ArCoMo_numbers[itr]
    ArCoMo_number_gt = ArCoMo_numbers_gt[itr]
    # SELECT OCT PART WITH INDEXES
    #start_idx = 300 #Ar900: 105 #Ar1500: 90 #Ar1300: 170 #Ar1100: 300 #Ar1000: 260 #Ar300: 620, Ar200: 150, Ar100: 58
    #end_idx = 600 #Ar900: 405 #Ar1500: 390 #Ar1300: 480 #Ar1100: 600 #Ar1000: 560 #Ar300: 950, Ar200: 410, Ar100: 266


    ply_file = 'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_Colored_Qaulity_Overlap_Correction.ply'
    ply_file_gt = 'C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/ArCoMo_Data/ArCoMo' + ArCoMo_number_gt + '/output_ground_truth/ArCoMo' + ArCoMo_number_gt + '_ground_truth_mesh.ply'
    gt_csv = 'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number_gt + '_areas.csv'
    model_csv = 'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_areas.csv'
    analysis_output_csv = 'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_areas_statisitcal_analysis.xlsx'
    linear_regression_results_csv = 'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_areas_linear_regression.xlsx'
    mesh_results_csv = 'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + 'statistical_results_mesh_gemoetry.csv'
    centerline_file = 'C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/ArCoMo_Data/ArCoMo' + ArCoMo_number_gt + '/ArCoMo' + ArCoMo_number_gt + '_centerline.txt'
    registration_start_idx_file = 'C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/ArCoMo_Data/ArCoMo' + ArCoMo_number + '/output/ArCoMo' + ArCoMo_number + '_centerline_registration_points.txt'
    start_end_idx_file = 'C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/ArCoMo_Data/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_oct_frames_info.txt'
    folder_path_img = 'C:/Users/JL/Model_evaluation/ArCoMo' + ArCoMo_number + '/plots' 


    if False:
        ply_file = 'C:/Users/siege/Universitaet Bern/Ilic, Marc Sascha (STUDENTS) - Dokumente/ArCoMo/Workflow_3D_reconstruction/Model_evaluation/ArCoMo100/ArCoMo100_Colored_Qaulity_Overlap_Correction.ply'
        ply_file_gt = 'C:/Users/siege/Universitaet Bern/Ilic, Marc Sascha (STUDENTS) - Dokumente/ArCoMo/Workflow_3D_reconstruction/Gound_truth_meshes/ArCoMo1_ground_truth.ply'
        gt_csv = 'C:/Users/siege/Universitaet Bern/Ilic, Marc Sascha (STUDENTS) - Dokumente/ArCoMo/Workflow_3D_reconstruction/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number_gt + '_areas.csv'
        model_csv = 'C:/Users/siege/Universitaet Bern/Ilic, Marc Sascha (STUDENTS) - Dokumente/ArCoMo/Workflow_3D_reconstruction/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_areas.csv'
        analysis_output_csv = 'C:/Users/siege/Universitaet Bern/Ilic, Marc Sascha (STUDENTS) - Dokumente/ArCoMo/Workflow_3D_reconstruction/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_areas_statisitcal_analysis.xlsx'
        linear_regression_results_csv = 'C:/Users/siege/Universitaet Bern/Ilic, Marc Sascha (STUDENTS) - Dokumente/ArCoMo/Workflow_3D_reconstruction/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + '_areas_linear_regression.xlsx'
        mesh_results_csv = 'C:/Users/siege/Universitaet Bern/Ilic, Marc Sascha (STUDENTS) - Dokumente/ArCoMo/Workflow_3D_reconstruction/Model_evaluation/ArCoMo' + ArCoMo_number + '/ArCoMo' + ArCoMo_number + 'statistical_results_mesh_gemoetry.csv'
        centerline_file = 'C:/Users/siege/OneDrive - Universitaet Bern/Documents/ArCoMo_workflow/ArCoMo_DigitalModel_Workflow/ArCoMo_Data\ArCoMo' + ArCoMo_number_gt + '/ArCoMo' + ArCoMo_number_gt + '_centerline.txt'

        folder_path = 'C:/Users/siege/Universitaet Bern/Ilic, Marc Sascha (STUDENTS) - Dokumente/ArCoMo/Workflow_3D_reconstruction/Model_evaluation/ArCoMo'  + ArCoMo_number + "/ArCoMo" + ArCoMo_number + "_"

    # Read in index values
    main_branch_start_idx = None
    oct_start = None
    oct_end = None
    oct_registration = None

    # Read the file
    with open(registration_start_idx_file, 'r') as file:
        for line in file:
            if line.startswith('Registration_main_branch_start_idx'):
                # Extract the number after the colon
                main_branch_start_idx = int(line.split(':')[1].strip())
                break

    # Read the file
    with open(start_end_idx_file, 'r') as file:
        for line in file:
            if line.startswith('oct_start'):
                oct_start = int(line.split(':')[1].strip())
            elif line.startswith('oct_end'):
                oct_end = int(line.split(':')[1].strip())
            elif line.startswith('oct_registration'):
                oct_registration = int(line.split(':')[1].strip())

    start_idx = main_branch_start_idx - (oct_end - oct_registration)
    end_idx = main_branch_start_idx + (oct_registration- oct_start)
    # Print the extracted value
    print("Start idx:", start_idx)
    print("End idx:", end_idx)

    ###########################################################################################################
    create_folder(folder_path_img)

    folder_path_img = folder_path_img + "/ArCoMo" + ArCoMo_number + "_"

        ######################################################################################################################################
    ########################### 3D COLOR CODED SCATTER PLOT ##############################################################################
    ######################################################################################################################################
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
    # ax.set_title('Vertex distances differences')

    # Filter out quality values below 0.05
    abs_quality = abs(quality)
    quality_filtered = abs_quality[abs_quality >= 0.05]
    all_quality_filtered.append(quality_filtered)  # Add filtered quality data to the list

    plt.savefig(folder_path_img + 'vert_diff_3D.svg', format='svg')
    plt.savefig(folder_path_img + 'vert_diff_3D.png', format='png')
    #plt.show()  # Show all histogram and scatter plots


    # Calculate statistical measures
    min_val = np.min(quality_filtered)
    max_val = np.max(quality_filtered)
    median_val = np.median(quality_filtered)
    mean_val = np.mean(quality_filtered)
    std_dev = np.std(quality_filtered)


    # Prepare data to be written to CSV
    data = [
        ["Measure", "Value"],
        ["Min", f"{min_val:.2f}"],
        ["Max", f"{max_val:.2f}"],
        ["Median", f"{median_val:.2f}"],
        ["Mean", f"{mean_val:.2f}"],
        ["Std Dev", f"{std_dev:.2f}"]
    ]

    # Save the description to a csv file
    with open(mesh_results_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    ######################################################################################################################################
    ########################### 2D COLOR CODED SCATTER PLOT ##############################################################################
    ######################################################################################################################################

    ######### MODEL #########################

    mesh = trimesh.load_mesh(ply_file)

    # Load centerline
    centerline_points = parse_point_cloud_centerline(centerline_file)

    centerline_points = resample_center_line(centerline_points, 0.2)

    # Create a KDTree for efficient nearest neighbor search
    mesh_points = np.vstack((x, y, z)).T
    kd_tree = KDTree(mesh_points)

    radius = 3
    data_model = []
    data_model.append(["Centerline IDX", "Area", "Filter radius"])
    data_gt = []
    data_gt.append(["Centerline IDX", "Area", "Filter radius"])

    plt_idx = 0
    interp_points = 20
    are_threshold = 0.2

    area_all = []
    colors_all = []
    filtered_points_all = []
    # Loop over each centerline point to extract and plot the 2D points
    for idx, val in enumerate(centerline_points):
        if idx >= start_idx and idx <= end_idx:
            flag_except = 0
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

                x_plt = [point[0] for point in filtered_points]
                y_plt = [point[1] for point in filtered_points]
                
                points_2d = convert_to_2d(filtered_points, cut_point, normal)

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
                flag_except = 1
                
            if flag_except:
                data_model.append([idx, np.nan, radius])
            else:
                data_model.append([idx, area, radius])

    # Normalize area
    area_all = np.array(area_all)
    area_all_normalized = (area_all - np.min(area_all)) / (np.max(area_all) - np.min(area_all))
    # Create a figure for the 2D plot
    fig, ax = plt.subplots()
    # Plot each square
    for idx, area in enumerate(area_all_normalized):
        colors = colors_all[idx]
        for i, color in enumerate(colors):
            #square = plt.Rectangle((i*area-interp_points/2*area, plt_idx), area, 1, color=color) #use this if you want to include the are into the 2d plot
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
    plt.savefig(folder_path_img + 'vert_diff_2D.svg', format='svg')
    plt.savefig(folder_path_img + 'vert_diff_2D.png', format='png')
    #plt.show()



    # Create a PolyData object from the points
    polydata2 = pv.PolyData(filtered_points_end)
    polydata = pv.PolyData(filtered_points_start)

    ####################################################
    # Plot the original mesh, centerline, and both clipping planes
    p = pv.Plotter()

    for fil_point in filtered_points_all:
        polydata3 = pv.PolyData(fil_point)
        p.add_mesh(polydata3, color="yellow")

    # Add the intersectione
    p.add_mesh(polydata, color="yellow")
    p.add_mesh(polydata2, color="yellow")

    # Add original mesh
    p.add_mesh(mesh, color="green", opacity=0.3, show_edges=True)

    # Add clipped mesh

    # Add first clipping plane
    p.add_mesh(plane_polydata_start, color="red", opacity=0.5)
    p.add_mesh(plane_polydata_end, color="blue", opacity=0.5)

    # Add second clipping plane

    #p.show()

    ######### GROUND TRUTH #########################
    mesh = trimesh.load_mesh(ply_file_gt)

    # Load centerline
    centerline_points = parse_point_cloud_centerline(centerline_file)

    centerline_points = resample_center_line(centerline_points, 0.2)

    # Create a KDTree for efficient nearest neighbor search
    mesh_points = np.vstack((x, y, z)).T
    kd_tree = KDTree(mesh_points)
    area_all = []
    colors_all = []
    filtered_points_all = []
    for idx, val in enumerate(centerline_points):
        if idx >= start_idx and idx <= end_idx:
            flag_except = 0
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

                x_plt = [point[0] for point in filtered_points]
                y_plt = [point[1] for point in filtered_points]
                
                points_2d = convert_to_2d(filtered_points, cut_point, normal)

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
                flag_except = 1
                
            if flag_except:
                data_gt.append([idx, np.nan, radius])
            else:
                data_gt.append([idx, area, radius])

    # Save areas to csv file
    with open(model_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_model)
    with open(gt_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_gt)



    ######################################################################################################################################
    ########################### Area visualisation (linear regression) ######################################################################
    ######################################################################################################################################


    df_gt = pd.read_csv(gt_csv)
    df_model = pd.read_csv(model_csv)

    colors = ['black', 'blue']

    fig = plt.figure()
    plt.plot(df_gt['Centerline IDX'], df_gt['Area'], marker='o', linestyle='-', color=colors[0], label='Ground turth')
    plt.plot(df_model['Centerline IDX'], df_model['Area'], marker='o', linestyle='-', color=colors[1], label='Model')
    plt.legend()
    plt.xlabel('Centerline IDX')
    plt.ylabel('Area')
    plt.title('Centerline IDX vs Area')
    plt.grid(True)

    plt.savefig(folder_path_img + 'area_diff.svg', format='svg')
    #plt.show()

    
    diff_area = np.abs(df_gt['Area']- df_model['Area'])
    area_gt = df_gt['Area']
    area_gt_zscore = (area_gt-np.mean(area_gt))/np.std(area_gt)
    mla_gt = np.min(area_gt)
    area_model = df_model['Area']
    area_model_zscore = (area_model-np.mean(area_model))/np.std(area_model)
    mla_model = np.min(area_model)
    diff_zscore = np.abs(area_gt_zscore-area_model_zscore)

    spearman_c, spearman_p = spearmanr(area_model_zscore, area_gt_zscore)

    rel_mla_err = (np.abs(mla_model-mla_gt)/mla_gt)*100
    print(f"MLA GT: {mla_gt:.4f}")
    print(f"MLA Model: {mla_model:.4f}")
    print(f"MLA error: {rel_mla_err:.4f} %")



    fig = plt.figure()
    plt.plot(df_gt['Centerline IDX'], area_gt_zscore, marker='o', linestyle='-', color=colors[0], label='Ground turth')
    plt.plot(df_model['Centerline IDX'], area_model_zscore, marker='o', linestyle='-', color=colors[1], label='Model')
    plt.legend()
    plt.xlabel('Centerline IDX')
    plt.ylabel('z-score (Lumen area)')
    plt.title('Centerline IDX vs z-score')
    plt.grid(True)

    plt.savefig(folder_path_img + 'zscore_diff_3D.svg', format='svg')
    #plt.show()

    # Bland-Altman plot
    mean_area = (df_gt['Area'] + df_model['Area']) / 2

    # Create the Bland-Altman plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_area, diff_area, alpha=0.5)
    plt.axhline(np.mean(diff_area), color='red', linestyle='--')
    plt.axhline(np.mean(diff_area) + 1.96 * np.std(diff_area), color='blue', linestyle='--')
    plt.axhline(np.mean(diff_area) - 1.96 * np.std(diff_area), color='blue', linestyle='--')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean of Ground Truth and Model Area')
    plt.ylabel('Absolute Difference between Ground Truth and Model Area')
    plt.grid(True)

    plt.savefig(folder_path_img + 'bland_alt.svg', format='svg')
    #plt.show()

    # Statistical Analysis
    analysis_data = []

    mean_abs_error = np.mean(diff_area)
    median_abs_error = np.median(diff_area)
    mean_std_error = np.std(diff_area)
    max_abs_error = np.max(diff_area)
    mean_squared_error = np.mean((area_model - area_gt) ** 2)
    correlation_coefficient = np.corrcoef(area_model, area_gt)[0, 1]
    # spearman_c, spearman_p = spearmanr(area_model, area_gt)

    mean_zscore_error = np.mean(diff_zscore)
    std_zscore_error = np.std(diff_zscore)

    analysis_data.append(["Overlap", max_abs_error, mean_abs_error, mean_std_error,
                        median_abs_error, mean_squared_error, spearman_c,
                            mean_zscore_error, std_zscore_error])

    print(f"Spearman rank correlation coefficient: {spearman_c:.4f}")
    print(f"P-value: {spearman_p:.4f}")

    # Create a DataFrame for the statistical analysis data
    analysis_df = pd.DataFrame(analysis_data, columns=['Method', 'Max Absolute Error', 'Mean Absolute Error', 'Mean STD Error', 
                                                    'Median Absolute Error', 'Mean Squared Error', 'Spearman Coefficient',
                                                    'Mean z-score diff','Std z-score diff'])

    # Save the DataFrame to an Excel file
    analysis_df.to_excel(analysis_output_csv, index=False)

    # LINEAR REGRESSION
    slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = linregress(area_gt_zscore, area_model_zscore)

    # Calculate R-squared
    r_squared_1 = r_value_1 ** 2

    # Create a DataFrame for linear regression values
    data = {
        'Slope': [slope_1],
        'Intercept': [intercept_1],
        'R-value': [r_value_1],
        'P-value': [p_value_1],
        'Standard Error': [std_err_1],
        'R-value-squared': [r_squared_1]

    }

    df_linear_regression = pd.DataFrame(data, index=['Overlap'])

    # Save linear regression values to an Excel file
    df_linear_regression.to_excel(linear_regression_results_csv)

    # Plotting
    plt.figure(figsize=(10, 8))

    # Plotting data points and fitted lines with matching colors
    plt.plot(area_gt, area_model, 'o', color=colors[1], label='Image correlation')
    plt.plot(area_gt, slope_1 * area_gt + intercept_1, '--', color=colors[1], label=f'Fitted Line (slope={slope_1:.2f}, intercept={intercept_1:.2f})')

    # Plotting the 45-degree line for reference
    min_val = min(min(area_gt), min(area_model))
    max_val = max(max(area_gt), max(area_model))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='45° Reference line')

    plt.xlabel('Ground Truth Area')
    plt.ylabel('Measured Area')
    plt.title('Measured Areas with Fitted Lines to Ground Truth')
    plt.legend()

    plt.savefig(folder_path_img + 'lin_reg.svg', format='svg')
    #plt.show()

from workflow_OCT_lumen_extraction_visualization import oct_lumen_extraction
from workflow_center_line_registration_visualization import center_line_registration
from workflow_visual_pointcloud_editing_VTK_point_visualization import point_cloud_visual_editing
from workflow_center_line_smooting_gui_visualization import PointCloudSmoothingVisualizer
from workflow_center_line_registration_point_selection_GUI_visualization import PointCloudRegistrationPointSelectionVisualizer
import open3d as o3d


def parse_my_points(file_path):
    data = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into three values
            parts = line.strip().split()

            # Ensure there are three values on each line
            if len(parts) == 3:
                # Parse the values as floats and append them to the respective lists
                px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                data.append((px, py, pz))
    return np.array(data[0])

if __name__ == "__main__":
    color1 = (0, 255, 0)  # Green circle
    color2 = (192, 220, 192)  # Circle dots color

    # OCT image start and stop frame
    OCT_start_frame = 300
    OCT_end_frame = 350

    # Initialize the z-coordinate for the first image
    z_distance = 0.2  # Increment by 0.2 or 0.1mm

    # smoothing kernel size and threshold value
    smoothing_kernel_size = 15  # Adjust as needed
    threshold_value = 100

    # Image crop-off
    crop = 157 #157 for view 10mm #130 for view 7mm  # pixels

    # Define the conversion factor: 1 millimeter = 102 pixels
    conversion_factor = 1 / 103.0

    # Input file paths
    input_file_OCT = 'workflow_data/OCT.tif'
    input_file_centerline = "workflow_data/centerline.txt"
    # Ulit images
    letter_x_mask_path = "workflow_utils/image_X.jpg"

    # Displaying and saving options
    save_file = False
    display_results = False
    save_images_for_controll = False 

    # Extract oct lumen contour and align them
    if True:
        oct_lumen_extractor = oct_lumen_extraction()
        OCT_registration_frame = oct_lumen_extractor.find_registration_frame(letter_x_mask_path, input_file_OCT, crop, color1, color2, display_results)
        registration_point = oct_lumen_extractor.get_registration_point(color1, color2, input_file_OCT, crop, OCT_registration_frame, display_results, z_distance, save_file, conversion_factor)
        my_aligned_images = oct_lumen_extractor.process_tif_file(crop, input_file_OCT, OCT_end_frame, OCT_start_frame, z_distance, conversion_factor, save_file, color1, color2, smoothing_kernel_size, threshold_value, display_results, registration_point, OCT_registration_frame, save_images_for_controll)
        with open("workflow_processed_data_output/my_aligned_points.txt", "w") as file:
            # Iterate through each array in the list
            for point_array in my_aligned_images:
                for point in point_array:
                    file.write(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}\n")

    # Parse OCT frames
    center_line_registrator = center_line_registration()
    grouped_OCT_frames = center_line_registrator.parse_OCT_lumen_point_cloud("workflow_processed_data_output/output_point_cloud.txt")

    # Make smoother bifurication curves
    pc_smoother = PointCloudSmoothingVisualizer(input_file_centerline)
    smoothed_pc_centerline = pc_smoother.pc_centerline
    if False:
        with open("workflow_processed_data_output/centerline_resmapled.txt", 'w') as file:
            for point in smoothed_pc_centerline:
                file.write(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}\n")

    # Resample center line
    resampled_pc_centerline = center_line_registrator.resample_center_line(smoothed_pc_centerline, display_results, z_distance)

    # Compute center line vectors, that point from the current point to the next point
    centerline_vectors = center_line_registrator.find_centerline_vectors(resampled_pc_centerline, display_results)

    # Load marked registration point in CT
    registration_point_CT = center_line_registrator.parse_registration_point_CT("workflow_data/CT_registration_point.txt")

    # Load marked registration point in OCT
    registration_point_OCT = center_line_registrator.parse_registration_point_OCT("workflow_processed_data_output/aligned_OCT_registration_point.txt")

    # Adapt z value
    centerline_registration_point_selector = PointCloudRegistrationPointSelectionVisualizer(resampled_pc_centerline, registration_point_CT)
    centerline_registration_start = centerline_registration_point_selector.selected_point_index

    print(centerline_registration_start)

    # REMOVE !!!!!!!!!!!!!!!!!

    OCT_registration_frame = 203
    oct_lumen_rotation_matrix, rotated_registration_point_OCT = center_line_registrator.get_oct_lumen_rotation_matrix(resampled_pc_centerline, centerline_registration_start, grouped_OCT_frames, registration_point_OCT, registration_point_CT, OCT_registration_frame, z_distance, display_results)

    # rotate OCT_frames
    rotated_grouped_OCT_frames = center_line_registrator.rotate_frames(grouped_OCT_frames, oct_lumen_rotation_matrix)
    import numpy as np
    print("start parsing")

    my_aligned_images_parsed = np.loadtxt("workflow_processed_data_output/my_aligned_points.txt")
    print("stop parsing")


    my_aligned_images_rotated = np.dot(my_aligned_images_parsed, oct_lumen_rotation_matrix.T)
    import matplotlib.pyplot as plt
    if False:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Show the plot
        x_filtered = []
        y_filtered = []
        z_filtered = []
        for itr in range(len(my_aligned_images)):
            for frame_points in my_aligned_images[itr]:
                x_filtered.append(frame_points[0])
                y_filtered.append(frame_points[1])
                z_filtered.append(frame_points[2])
        ax.scatter(x_filtered[::120], y_filtered[::120], z_filtered[::120], c="blue", marker='o')
        ax.set_xlabel('Px')
        ax.set_ylabel('Py')
        ax.set_zlabel('Pz')
        #plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Show the plot
        x_filtered = []
        y_filtered = []
        z_filtered = []
        for itr in range(len(my_aligned_images_rotated)):
            for frame_points in my_aligned_images_rotated[itr]:
                x_filtered.append(frame_points[0])
                y_filtered.append(frame_points[1])
                z_filtered.append(0.2*itr)
        ax.scatter(x_filtered[::120], y_filtered[::120], z_filtered[::120], c="blue", marker='o')
        x_filtered = []
        y_filtered = []
        z_filtered = []
        for z_level, frame_points in grouped_OCT_frames.items():
            if z_level > 56.3:
                for i, data in enumerate(frame_points):
                        x_filtered.append(frame_points[i][0])
                        y_filtered.append(frame_points[i][1])
                        z_filtered.append(frame_points[i][2])
        ax.scatter(x_filtered[::30], y_filtered[::30], z_filtered[::30], c="red", marker='o')
        ax.set_xlabel('Px')
        ax.set_ylabel('Py')
        ax.set_zlabel('Pz')
        #plt.show()


    if display_results:
        #------------------------------------------#
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Show the plot
        x_filtered = []
        y_filtered = []
        z_filtered = []
        for z_level, frame_points in grouped_OCT_frames.items():
            for i, data in enumerate(frame_points):
                x_filtered.append(frame_points[i][0])
                y_filtered.append(frame_points[i][1])
                z_filtered.append(frame_points[i][2])
        ax.scatter(x_filtered[::30], y_filtered[::30], z_filtered[::30], c="blue", marker='o')
        ax.set_xlabel('Px')
        ax.set_ylabel('Py')
        ax.set_zlabel('Pz')
        plt.show()
        #------------------------------------------#


    display_results = True
    #register frames onto centerline
    center_line_registrator.register_OCT_frames_onto_centerline(my_aligned_images_rotated, rotated_grouped_OCT_frames, centerline_registration_start, centerline_vectors,
                                        resampled_pc_centerline, OCT_registration_frame, z_distance, rotated_registration_point_OCT, save_file, display_results)

    # Create CT point cloud from ply file
    # Load the .PLY file
    mesh = o3d.io.read_triangle_mesh("workflow_data/CT_ply.ply")

    # Decimate the mesh (optional)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)

    # Create a regular point cloud from the mesh with a smaller voxel size
    voxel_size = 0.05  # Adjust this value to control point density (smaller values = denser point cloud)
    pcd = mesh.sample_points_uniformly(number_of_points=int(mesh.get_surface_area() / voxel_size))

    o3d.io.write_point_cloud("workflow_data/CT_pointcloud_resampled.ply", pcd)
    point_cloud = o3d.io.read_point_cloud("workflow_data/CT_pointcloud_resampled.ply")

    # Extract the points as a numpy array
    points = pcd.points

    # Define the output file name
    output_file = "workflow_data/CT_pointcloud.txt"

    # Write the points to the text file
    with open(output_file, "w") as file:
        for point in points:
            file.write(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}\n")

    # Visual point cloud editing:
    point_cloud_visual_editior = point_cloud_visual_editing()
    file_path_1 = "workflow_data/CT_pointcloud.txt"
    file_path_2 = "workflow_processed_data_output/saved_registered_splines.txt"
    point_cloud_visual_editior.run_editor(file_path_1, file_path_2)
    point_cloud_save = point_cloud_visual_editior.fused_point_cloud
    
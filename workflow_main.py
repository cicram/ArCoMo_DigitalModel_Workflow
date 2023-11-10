from workflow_OCT_lumen_extraction import oct_lumen_extraction
from workflow_center_line_registration import center_line_registration
from workflow_visual_pointcloud_editing_VTK_point import point_cloud_visual_editing
from workflow_center_line_smooting_gui import PointCloudSmoothingVisualizer

if __name__ == "__main__":
    color1 = (0, 255, 0)  # Green circle
    color2 = (192, 220, 192)  # Circle dots color

    # Initialize the z-coordinate for the first image
    z_distance = 0.1  # Increment by 0.2 or 0.1mm

    # smoothing kernel size and threshold value
    smoothing_kernel_size = 15  # Adjust as needed
    threshold_value = 100

    # Image crop-off
    crop = 157 #for view 10mm #130 for view 7mm  # pixels

    # Define the conversion factor: 1 millimeter = 98 pixels
    conversion_factor = 1 / 98.0

    # Input file path
    input_file = 'phantom_data/ArCoMoPhantom-2.tif'

    # Registration frame number
    carina_point_frame = 373 - 1 # previsou 121

    # Displaying and saving options
    save_file = False
    display_results = False
    save_images_for_controll = False 

    # Extract oct lumen contour and align them
    oct_lumen_extractor = oct_lumen_extraction()
    if False:
        registration_point = oct_lumen_extractor.get_registration_point(color1, color2, input_file, crop, carina_point_frame, display_results, z_distance, save_file, conversion_factor)
        oct_lumen_extractor.process_tif_file(crop, input_file, z_distance, conversion_factor, save_file, color1, color2, smoothing_kernel_size, threshold_value, display_results, registration_point, carina_point_frame, save_images_for_controll)

        # Parse OCT frames
        center_line_registrator = center_line_registration()
        grouped_OCT_frames = center_line_registrator.parse_OCT_lumen_point_cloud("workflow_processed_data_output/output_point_cloud.txt")

        # Parse center line point cloud
        #pc_centerline = center_line_registrator.parse_point_cloud_centerline(, display_results)

        # Make smoother bifurication curves
        pc_smoother = PointCloudSmoothingVisualizer("workflow_data/centerline.txt")
        smoothed_pc_centerline = pc_smoother.pc_centerline

        # remove !!!!
        import numpy as np
        import matplotlib.pyplot as plt
        smoothed_pc_centerline = np.loadtxt("phantom_data/centerline.txt")
        sidearm_centerline = np.loadtxt("phantom_data/second_curve.txt")

        # Resample point cloud
        resampled_pc_centerline = center_line_registrator.resample_center_line(smoothed_pc_centerline, display_results, z_distance)

        # Create a 3D scatter plot for the original points
        # Compute center line vectors, that point from the current point to the next point
        centerline_vectors = center_line_registrator.find_centerline_vectors(resampled_pc_centerline, display_results)

        # Find marked registration point in CT
        registration_point_CT = center_line_registrator.parse_registration_point("phantom_data/registration_point_CT.txt", display_results)

        # Find marked registration point in OCT
        registration_point_OCT = center_line_registrator.parse_registration_point("workflow_processed_data_output/aligned_OCT_registration_point.txt", display_results)

        OCT_registration_frame = 373 - 1  #121
        # Adapt z value
        centerline_registration_start = center_line_registrator.find_closest_point_index(resampled_pc_centerline, registration_point_CT)
        if display_results:
            x_orig = sidearm_centerline[:, 0]
            y_orig = sidearm_centerline[:, 1]
            z_orig = sidearm_centerline[:, 2]
            x_resampled = resampled_pc_centerline[:, 0]
            y_resampled = resampled_pc_centerline[:, 1]
            z_resampled = resampled_pc_centerline[:, 2]
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_resampled, y_resampled, z_resampled, c="yellow", marker='o')
            ax.scatter(x_orig, y_orig, z_orig, c="black", marker='o')
            ax.scatter(resampled_pc_centerline[centerline_registration_start, 0], resampled_pc_centerline[centerline_registration_start, 1], resampled_pc_centerline[centerline_registration_start, 2], c="red", marker='x')

            ax.scatter(registration_point_CT[0], registration_point_CT[1], registration_point_CT[2], c="blue", marker='x')

            ax.set_xlabel('Px')
            ax.set_ylabel('Py')
            ax.set_zlabel('Pz')
            plt.show()

        oct_lumen_rotation_matrix, rotated_registration_point_OCT = center_line_registrator.get_oct_lumen_rotation_matrix(resampled_pc_centerline, centerline_registration_start, grouped_OCT_frames, registration_point_OCT, registration_point_CT, OCT_registration_frame, z_distance, display_results)

        # rotate OCT_frames
        rotated_grouped_OCT_frames = center_line_registrator.rotate_frames(grouped_OCT_frames, oct_lumen_rotation_matrix)

        if display_results:
            #------------------------------------------#
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
        center_line_registrator.register_OCT_frames_onto_centerline(rotated_grouped_OCT_frames, centerline_registration_start, centerline_vectors,
                                            resampled_pc_centerline, OCT_registration_frame, z_distance, rotated_registration_point_OCT, save_file, display_results)

    # Visual point cloud editing:
    point_cloud_visual_editior = point_cloud_visual_editing()
    file_path_1 = "phantom_data/noisy_downsampled_point_cloud_with_branch.txt"
    file_path_2 = "workflow_processed_data_output/saved_registered_splines.txt"
    point_cloud_visual_editior.run_editor(file_path_1, file_path_2)
    point_cloud_save = point_cloud_visual_editior.fused_point_cloud
    
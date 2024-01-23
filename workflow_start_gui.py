import tkinter as tk
from tkinter import ttk
import os
from workflow_OCT_lumen_extraction_new import oct_lumen_extraction
from workflow_center_line_registration import center_line_registration
from workflow_visual_pointcloud_editing_VTK_point import point_cloud_visual_editing
from workflow_center_line_smooting_gui import PointCloudSmoothingVisualizer
from workflow_center_line_registration_point_selection_GUI import PointCloudRegistrationPointSelectionVisualizer
import open3d as o3d
import tkinter as tk
import numpy as np


class OCTAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Start")
        self.save_intermediate_steps = False
        self.display_intermediate_results = False
        self.oct_start = 1
        self.oct_registration_frame = 1
        self.oct_end = 100
        self.arcomo_number = 0

        self.create_widgets()

    def create_widgets(self):
        # Labels
        self.label_oct_registration_frame = ttk.Label(self.master, text="OCT Registration Frame:")
        self.label_oct_start = ttk.Label(self.master, text="OCT Start Frame:")
        self.label_oct_end = ttk.Label(self.master, text="OCT End Frame:")
        self.label_arcomo_number = ttk.Label(self.master, text="ArCoMo Number:")

        # Entry Widgets
        self.entry_oct_registration_frame = ttk.Entry(self.master)
        self.entry_oct_start = ttk.Entry(self.master)
        self.entry_oct_end = ttk.Entry(self.master)
        self.entry_arcomo_number = ttk.Entry(self.master)

        # Checkboxes
        self.save_intermediate_steps_var = tk.IntVar()
        self.check_save_intermediate_steps = ttk.Checkbutton(self.master,
                                                             text="Save Intermediate Steps",
                                                             variable=self.save_intermediate_steps_var)

        self.display_intermediate_results_var = tk.IntVar()
        self.check_display_intermediate_results = ttk.Checkbutton(self.master,
                                                                 text="Display Intermediate Results",
                                                                 variable=self.display_intermediate_results_var)

        # Button
        self.run_button = ttk.Button(self.master, text="Run Analysis", command=self.run_analysis)
        self.get_oct_frames_info_button = ttk.Button(self.master, text="Get OCT Frames Info", command=self.get_oct_frames_info)

        # Text Widget for Instructions
        instructions_text = "Enter the ArCoMo Number and click 'Get OCT Frames Info'.\n" \
                            "Check vlaues, modify them if you want, mark checkboxes if you want intermediate results and the hit 'Run Analysis'"
        self.instructions_label = ttk.Label(self.master, text="Instructions:")
        self.instructions_text_widget = tk.Text(self.master, height=5, width=40, wrap=tk.WORD, state=tk.NORMAL)
        self.instructions_text_widget.insert(tk.END, instructions_text)
        #self.instructions_text_widget.configure(state=tk.DISABLED)  # Make the text widget read-only
        self.instructions_label.grid(row=8, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.instructions_text_widget.grid(row=9, column=0, columnspan=2, padx=10, pady=5, sticky="w")


        # Layout
        self.label_arcomo_number.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_arcomo_number.grid(row=0, column=1, padx=10, pady=5)

        self.get_oct_frames_info_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.label_oct_start.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.entry_oct_start.grid(row=2, column=1, padx=10, pady=5)

        self.label_oct_end.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.entry_oct_end.grid(row=3, column=1, padx=10, pady=5)

        self.label_oct_registration_frame.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.entry_oct_registration_frame.grid(row=4, column=1, padx=10, pady=5)

        self.check_save_intermediate_steps.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.check_display_intermediate_results.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.run_button.grid(row=7, column=0, columnspan=2, pady=10)

    def add_instruction_text(self, instruction_text):
        self.instructions_text_widget.configure(state=tk.NORMAL)
        self.instructions_text_widget.delete(1.0, tk.END)  # Delete existing text, starting from line 1
        self.instructions_text_widget.insert(1.0, instruction_text)
        self.instructions_text_widget.configure(state=tk.DISABLED)  # Make the text widget read-only
        self.master.update_idletasks()  # Force GUI update

    def get_oct_frames_info(self):
        # Retrieve ArCoMo Number
        arcomo_number = self.entry_arcomo_number.get()

        if not arcomo_number.isdigit():
            # Display an error message or handle invalid input
            return

        arcomo_number = int(arcomo_number)

        # Construct the file path for oct_frames_info.txt
        file_path = f"ArCoMo_Data/ArCoMo{arcomo_number}/ArCoMo{arcomo_number}_oct_frames_info.txt"

        # Check if the file exists
        if os.path.exists(file_path):
            # Read information from the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    key, value = map(str.strip, line.split(':'))
                    key = key.lower()

                    if key == 'oct_start':
                        self.entry_oct_start.delete(0, tk.END)
                        self.entry_oct_start.insert(0, value)
                    elif key == 'oct_end':
                        self.entry_oct_end.delete(0, tk.END)
                        self.entry_oct_end.insert(0, value)
                    elif key == 'oct_registration':
                        self.entry_oct_registration_frame.delete(0, tk.END)
                        self.entry_oct_registration_frame.insert(0, value)

        else:
            # Display an error message or handle the case where the file does not exist
            return
        
    def run_analysis(self):
        # Retrieve values from entry widgets and checkboxes
        self.oct_registration_frame = self.entry_oct_registration_frame.get()
        self.oct_start = self.entry_oct_start.get()
        self.oct_end = self.entry_oct_end.get()
        self.arcomo_number = self.entry_arcomo_number.get()
        self.save_intermediate_steps = self.save_intermediate_steps_var.get()
        self.display_intermediate_results = self.display_intermediate_results_var.get()

        # Run processing
        self.run_processing()


    def run_processing(self):
        color1 = (0, 255, 0)  # Green circle
        color2 = (192, 220, 192)  # Circle dots color

        #ArCoMo number
        arcomo_number = int(self.arcomo_number)

        # OCT image start and stop and registration frame
        OCT_start_frame = int(self.oct_start)
        OCT_end_frame = int(self.oct_end)
        OCT_registration_frame = int(self.oct_registration_frame)

        # Displaying and saving options
        save_file = self.save_intermediate_steps
        display_results = self.display_intermediate_results
        save_images_for_controll = False 

        # Initialize the z-coordinate for the first image
        z_distance = 0.2  # Increment by 0.2 or 0.1mm

        # smoothing kernel size and threshold value
        smoothing_kernel_size = 15  # Adjust as needed
        threshold_value = 100

        # Image crop-off
        crop = 157 #157 for view 10mm #130 for view 7mm  # pixels

        # Define the conversion factor: 1 millimeter = 103 pixels
        conversion_factor = 1 / 103.0

        # Input file paths
        input_file_OCT = 'ArCoMo_Data/ArCoMo' + str(arcomo_number) + '/ArCoMo' + str(arcomo_number) +'_oct.tif'
        input_file_centerline = 'ArCoMo_Data/ArCoMo' + str(arcomo_number) + '/ArCoMo' + str(arcomo_number) +'_centerline.txt'
        input_file_ct_registration_point = 'ArCoMo_Data/ArCoMo' + str(arcomo_number) + '/ArCoMo' + str(arcomo_number) + '_CT_registration.txt'
        input_file_ct_mesh = 'ArCoMo_Data/ArCoMo' + str(arcomo_number) + '/ArCoMo' + str(arcomo_number) + '_CT.ply'
        path_fused_point_cloud = 'ArCoMo_Data/ArCoMo' + str(arcomo_number) + '/output/ArCoMo' + str(arcomo_number) + '_fused_point_cloud.xyz'


        # Update instructions
        instruction_text = "Select with the mouse (click) the wanted registration point (carina point) in the OCT slide."
        self.add_instruction_text(instruction_text)

        # Extract oct lumen contour and align them
        oct_lumen_extractor = oct_lumen_extraction()
        registration_point = oct_lumen_extractor.get_registration_point(color1, color2, input_file_OCT, crop, OCT_registration_frame, display_results, z_distance, save_file, conversion_factor)
        
        
        # Update instructions
        instruction_text = "OCT frame extraction and rotation is performed, please wait and check resulting plot"
        self.add_instruction_text(instruction_text)

        oct_point_cloud, registration_point_OCT = oct_lumen_extractor.process_tif_file(crop, input_file_OCT, OCT_end_frame, OCT_start_frame, z_distance, conversion_factor, save_file, color1, color2, smoothing_kernel_size, threshold_value, display_results, registration_point, OCT_registration_frame, save_images_for_controll)

        # Parse OCT frames
        center_line_registrator = center_line_registration()
        grouped_OCT_frames = center_line_registrator.restructure_OCT_lumen_point_cloud(oct_point_cloud)

        # If parsing from file
        #grouped_OCT_frames = center_line_registrator.parse_OCT_lumen_point_cloud("workflow_processed_data_output/output_point_cloud.txt")

        # Load marked registration point in CT
        registration_point_CT = center_line_registrator.parse_registration_point_CT(input_file_ct_registration_point)

        # Make smoother bifurication curves
        # Update instructions
        instruction_text = "Make a smooth centerline, by fitting a spline. "
        self.add_instruction_text(instruction_text)
        pc_smoother = PointCloudSmoothingVisualizer(input_file_centerline, registration_point_CT)
        smoothed_pc_centerline = pc_smoother.pc_centerline

        # Resample center line
        resampled_pc_centerline = center_line_registrator.resample_center_line(smoothed_pc_centerline, display_results, z_distance)

        # Compute center line vectors, that point from the current point to the next point
        centerline_vectors = center_line_registrator.find_centerline_vectors(resampled_pc_centerline, display_results)

        # Get registration hight
        # Update instructions
        instruction_text = "Select the registration height on the centerline (should be on hight of bifurication)"
        self.add_instruction_text(instruction_text)
        centerline_registration_point_selector = PointCloudRegistrationPointSelectionVisualizer(resampled_pc_centerline, registration_point_CT)
        centerline_registration_start = centerline_registration_point_selector.selected_point_index

        oct_lumen_rotation_matrix, rotated_registration_point_OCT = center_line_registrator.get_oct_lumen_rotation_matrix(resampled_pc_centerline, centerline_registration_start, grouped_OCT_frames, registration_point_OCT, registration_point_CT, OCT_registration_frame, z_distance, display_results)

        # rotate OCT_frames
        rotated_grouped_OCT_frames = center_line_registrator.rotate_frames(grouped_OCT_frames, oct_lumen_rotation_matrix, display_results)

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


        #register frames onto centerline
        saved_registered_splines = center_line_registrator.register_OCT_frames_onto_centerline(rotated_grouped_OCT_frames, centerline_registration_start, centerline_vectors, 
                                                                                            resampled_pc_centerline, OCT_registration_frame, z_distance, rotated_registration_point_OCT, save_file, display_results)

        # Create CT point cloud from ply file

        # Update instructions
        instruction_text = "Delete overlapping point form the pointcloud, then save it."
        self.add_instruction_text(instruction_text)

        # Load the .PLY file
        mesh = o3d.io.read_triangle_mesh(input_file_ct_mesh)

        # Decimate the mesh (optional)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)

        # Create a regular point cloud from the mesh with a smaller voxel size
        voxel_size = 0.05  # Adjust this value to control point density (smaller values = denser point cloud)
        pcd = mesh.sample_points_uniformly(number_of_points=int(mesh.get_surface_area() / voxel_size))

        # Extract the points as a numpy array
        ct_points_ = pcd.points

        if save_file:
            # Define the output file name
            output_file_ct_points = "temp/CT_pointcloud.txt"

            # Write the points to the text file
            with open(output_file_ct_points, "w") as file:
                for point in ct_points_:
                    file.write(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}\n")

        ct_points = []
        for point in ct_points_:
            ct_points.append([point[0], point[1], point[2]])
        ct_points = np.array(ct_points)

        oct_points = []
        for spline in saved_registered_splines:
            spline = spline[::2]
            for point in spline:
                oct_points.append([point[0], point[1], point[2]])

        oct_points = np.array(oct_points)

        # Visual point cloud editing:
        point_cloud_visual_editior = point_cloud_visual_editing()
        point_cloud_visual_editior.run_editor(ct_points, oct_points)

        # Save the fused point cloud to a text file
        point_cloud_save = point_cloud_visual_editior.fused_point_cloud
        np.savetxt(path_fused_point_cloud, point_cloud_save, fmt='%f %f %f')

        # Update instructions
        instruction_text = "Continue with the software Meshlab. Import the outputfile.xyz (from ArCoMoX/output/) into the software and open the instructions for the following steps. You can close this window now."
        self.add_instruction_text(instruction_text)

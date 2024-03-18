from tkinter import ttk
import os
from workflow_stent_calc_OCT_extraction import oct_extraction
from workflow_stent_calc_center_line_registration import center_line_registration
from workflow_stent_calc_visual_pointcloud_editing_VTK_point import point_cloud_visual_editing
from workflow_stent_calc_center_line_smooting_gui import PointCloudSmoothingVisualizer
from workflow_stent_calc_center_line_registration_point_selection_GUI import PointCloudRegistrationPointSelectionVisualizer
from workflow_stent_calc_image_visualization_on_model import OctImageVisualizier

import open3d as o3d
import tkinter as tk
import numpy as np

# Constants
STENT_AND_CALC = 3
STENT = 2
CALC = 1
BASIC = 0
IMAGE = "Image"
ICP = "ICP"
OVERLAP = "Overlap"

class OCTAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Start")
        self.save_intermediate_steps = False
        self.display_intermediate_results = False
        self.include_stent = False
        self.include_calc = False
        self.oct_start = 1
        self.oct_registration_frame = 1
        self.oct_end = 100
        self.arcomo_number = 0
        self.processing_info = 0 # 0 is basic processing, 1 is calc processing, 2 is stent processing, 3 is full processing
        self.axial_twist_correction_method = "" # 0 is basic processing, 1 is calc processing, 2 is stent processing, 3 is full processing


        # Parameters
        self.color1 = (0, 255, 0)  # Green circle
        self.color2 = (192, 220, 192)  # Circle dots color
        # Initialize the z-coordinate for the first image
        self.z_distance = 0.2  # Increment by 0.2 or 0.1mm

        # smoothing kernel size and threshold value
        self.smoothing_kernel_size = 15  # Adjust as needed
        self.threshold_value = 100

        # Image crop-off
        self.crop_top = 157 #157 for view 10mm #130 for view 7mm  # pixels
        self.crop_bottom = 0 

        # Define the conversion factor: 1 millimeter = 103 pixels
        self.conversion_factor = 1 / 103.0
        self.create_widgets()

        # Define hight and width
        self.image_hight = 1024 - self.crop_top - self.crop_bottom
        self.image_withd = 1024

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

        # Dropdown Box for Correction Method
        self.correction_method_var = tk.StringVar()
        self.correction_method_dropdown = ttk.Combobox(self.master, textvariable=self.correction_method_var, 
                                                        values=["Image", "ICP", "Overlap"])
        self.correction_method_var.set("Image")
        
        # Checkboxes
        self.save_intermediate_steps_var = tk.IntVar()
        self.check_save_intermediate_steps = ttk.Checkbutton(self.master,
                                                             text="Save Intermediate Steps",
                                                             variable=self.save_intermediate_steps_var)

        self.display_intermediate_results_var = tk.IntVar()
        self.check_display_intermediate_results = ttk.Checkbutton(self.master,
                                                                 text="Display Intermediate Results",
                                                                 variable=self.display_intermediate_results_var)
        
        self.include_stent_var = tk.IntVar()
        self.check_include_stent = ttk.Checkbutton(self.master, text="Include stent", variable=self.include_stent_var)

        self.include_calc_var = tk.IntVar()
        self.check_include_calc = ttk.Checkbutton(self.master, text="Include calc", variable=self.include_calc_var)
        
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
        self.instructions_label.grid(row=10, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.instructions_text_widget.grid(row=11, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.label_correction_method = ttk.Label(self.master, text="Axial Twist Correction Method:")

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
        
        self.label_correction_method.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.correction_method_dropdown.grid(row=5, column=1, padx=10, pady=5)

        self.check_include_calc.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.check_include_stent.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.check_save_intermediate_steps.grid(row=8, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.check_display_intermediate_results.grid(row=9, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.run_button.grid(row=10, column=0, columnspan=2, pady=10)

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
        self.include_calc = self.include_calc_var.get()
        self.include_stent = self.include_stent_var.get()
        self.axial_twist_correction_method = self.correction_method_var.get()
                #ArCoMo number
        self.arcomo_number = int(self.arcomo_number)

        # OCT image start and stop and registration frame
        self.OCT_start_frame = int(self.oct_start)
        self.OCT_end_frame = int(self.oct_end)
        self.OCT_registration_frame = int(self.oct_registration_frame)

        # Displaying and saving options
        self.save_file = self.save_intermediate_steps
        self.display_results = self.display_intermediate_results
        self.save_images_for_controll = False 

        # Input file paths
        self.input_file_OCT = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) +'_oct.tif'
        self.input_file_OCT_blank = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) +'_oct_blank.tif'
        self.input_file_centerline = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) +'_centerline.txt'
        self.input_file_ct_registration_point = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) + '_CT_registration.txt'
        self.input_file_ct_mesh = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) + '_CT.ply'
        self.path_fused_point_cloud = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/output/ArCoMo' + str(self.arcomo_number) + '_fused_point_cloud.xyz'


        if self.include_calc and self.include_stent:
            # Run full workflow
            print("full workflow")
            self.input_file_OCT = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) +'_oct_stent_lumen.tif'
            self.input_file_OCT_stent = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) +'_oct_stent_stent.tif'
            self.input_file_OCT_blank = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) +'_oct_stent_blank.tif'
            self.path_point_cloud_calc = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/output/ArCoMo' + str(self.arcomo_number) + '_point_cloud_calc.xyz'
            self.path_segmented_calc = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/output/ArCoMo' + str(self.arcomo_number) + '_segmented_calc.xyz'
            self.crop_bottom = 150
            self.image_hight = 1024 - self.crop_top - self.crop_bottom
            processing_info = STENT_AND_CALC
            self.run_processing(processing_info)

        elif self.include_stent and not self.include_calc:
            # Run workflow just with stent
            print("stent workflow")
            self.input_file_OCT = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) +'_oct_stent_lumen.tif'
            self.input_file_OCT_stent = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) +'_oct_stent_stent.tif'
            self.input_file_OCT_blank = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/ArCoMo' + str(self.arcomo_number) +'_oct_stent_blank.tif'
            self.path_point_cloud_stent = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/output/ArCoMo' + str(self.arcomo_number) + '_point_cloud_stent.xyz'
            self.crop_bottom = 150
            self.image_hight = 1024 - self.crop_top - self.crop_bottom
            processing_info = STENT
            self.run_processing(processing_info)

        elif not self.include_stent and self.include_calc:
            # Run workflow just with calc
            print("calc workflow")
            self.path_point_cloud_calc = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/output/ArCoMo' + str(self.arcomo_number) + '_point_cloud_calc.xyz'
            self.path_segmented_calc = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/output/ArCoMo' + str(self.arcomo_number) + '_segmented_calc.xyz'
            processing_info = CALC
            self.run_processing(processing_info)

        else:
            # Run basic workflow
            print("basic workflow")
            processing_info = BASIC
            self.run_processing(processing_info)


    def run_processing(self, processing_info):

        # Update instructions
        instruction_text = "Select with the mouse (click) the wanted registration point (carina point) in the OCT slide."
        self.add_instruction_text(instruction_text)

        # Get registration point in oct
        oct_extractor = oct_extraction()
        registration_point_OCT = oct_extractor.get_registration_point(self.input_file_OCT, self.crop_top, self.crop_bottom, self.OCT_start_frame, self.OCT_registration_frame, self.display_results, self.z_distance,
                                                                   self.save_file, self.conversion_factor)

        # Update instructions
        instruction_text = "OCT lumen extraction and rotation is performed, please wait and check resulting plot"
        self.add_instruction_text(instruction_text)

        # Get oct lumen contours
        oct_lumen_contours = oct_extractor.get_lumen_contour(self.crop_top, self.crop_bottom, self.input_file_OCT, self.OCT_end_frame, self.OCT_start_frame, self.OCT_registration_frame,
                                                             self.z_distance, self.conversion_factor, self.save_file, self.color1, self.color2, 
                                                             self.smoothing_kernel_size, self.threshold_value, self.display_results, self.save_images_for_controll)

        # Get rotation correction matrix
        if self.axial_twist_correction_method == IMAGE:
            oct_rotation_angles = oct_extractor.get_rotation_matrix(self.input_file_OCT_blank, self.OCT_start_frame, self.OCT_end_frame) 

        if self.axial_twist_correction_method == ICP:
            oct_rotation_angles = oct_extractor.get_rotation_matrix_ICP(oct_lumen_contours, self.z_distance) 

        if self.axial_twist_correction_method == OVERLAP:
            oct_rotation_angles = oct_extractor.get_rotation_matrix_overlap(oct_lumen_contours, self.z_distance, self.crop_top, self.crop_bottom, self.conversion_factor) 
        

        import  matplotlib.pyplot as plt
        plt.plot(oct_rotation_angles, label="Axial rotation angles")  # Add a label to your plot for the legend
        plt.xlabel("OCT pullback image")  # Add an x-axis label
        plt.ylabel("Rotation angle [°]")  # Add a y-axis label
        plt.legend()  # Display the legend
        plt.show()
        # Saving the data to a .txt file
        path_axial_correction = 'ArCoMo_Data/ArCoMo' + str(self.arcomo_number) + '/output/ArCoMo' + str(self.arcomo_number) + 'axial_angle_correction'  + str(self.axial_twist_correction_method)  +'.xyz'

        with open(path_axial_correction, "w") as file:
            for angle in oct_rotation_angles:
                file.write(f"{angle}\n")

        # Align oct frames and registration point
        oct_lumen_point_cloud = oct_extractor.frames_alignment(oct_lumen_contours, oct_rotation_angles, self.z_distance, self.image_hight, self.image_withd, self.conversion_factor)
        registration_point_OCT = oct_extractor.frames_alignment(registration_point_OCT, oct_rotation_angles, self.z_distance, self.image_hight, self.image_withd, self.conversion_factor)
     
        # Restructure frames
        center_line_registrator = center_line_registration()
        grouped_OCT_lumen = center_line_registrator.restructure_point_clouds(oct_lumen_point_cloud, self.OCT_start_frame, self.OCT_end_frame, self.z_distance)

        ############################
        unrotated_grouped_OCT_lumen = center_line_registrator.restructure_point_clouds(oct_lumen_contours, self.OCT_start_frame, self.OCT_end_frame, self.z_distance)

        ############################

        # Get oct calc contours
        if processing_info == CALC or processing_info == STENT_AND_CALC:
            instruction_text = "OCT calc extraction is performed. Instructions: Draw path with left mouse click then release. Press 'C' to accept drawn path. Press 'R' to restart. Press 'S' to save path."
            self.add_instruction_text(instruction_text)
            calc_contours = oct_extractor.get_calc_contours(self.path_segmented_calc, self.input_file_OCT_blank, self.OCT_start_frame, self.OCT_end_frame, self.z_distance, self.conversion_factor, self.crop_top, self.crop_bottom)
            # Align frames
            oct_calc_point_cloud = oct_extractor.frames_alignment(calc_contours, oct_rotation_angles, self.z_distance, self.image_hight, self.image_withd, self.conversion_factor)
            # Restructure frames
            grouped_calc = center_line_registrator.restructure_point_clouds(oct_calc_point_cloud, self.OCT_start_frame, self.OCT_end_frame, self.z_distance)


        # Get oct stent contours
        if processing_info == STENT or processing_info == STENT_AND_CALC:
            instruction_text = "OCT stent extraction is performed"
            self.add_instruction_text(instruction_text)
            stent_contours = oct_extractor.get_stent_contours(self.input_file_OCT_stent ,self.OCT_start_frame, self.OCT_end_frame, self.crop_top, self.crop_bottom, self.z_distance, self.conversion_factor)
            # Align frames
            oct_stent_point_cloud = oct_extractor.frames_alignment(stent_contours, oct_rotation_angles, self.z_distance,self.image_hight, self.image_withd, self.conversion_factor)
            # Restructure frames     
            grouped_stent = center_line_registrator.restructure_point_clouds(oct_stent_point_cloud, self.OCT_start_frame, self.OCT_end_frame, self.z_distance)


        # Load marked registration point in CT
        registration_point_CT = center_line_registrator.parse_registration_point_CT(self.input_file_ct_registration_point)

        # Make smoother bifurication curves of centerline
        instruction_text = "Make a smooth centerline, by fitting a spline. "
        self.add_instruction_text(instruction_text)
        pc_smoother = PointCloudSmoothingVisualizer(self.input_file_centerline, registration_point_CT)
        smoothed_pc_centerline = pc_smoother.pc_centerline

        # Resample center line
        resampled_pc_centerline = center_line_registrator.resample_center_line(smoothed_pc_centerline, self.display_results, self.z_distance)

        # Compute center line vectors, that point from the previous centerline-point to the next centerline-point
        centerline_vectors = center_line_registrator.find_centerline_vectors(resampled_pc_centerline, self.display_results)

        # Get registration points and compute roation matrix
        instruction_text = "Select the registration height on the centerline (should be on hight of bifurication). Red for main branch, Blue for side branch, use toggle button to switch"
        self.add_instruction_text(instruction_text)

        centerline_registration_point_selector = PointCloudRegistrationPointSelectionVisualizer(resampled_pc_centerline, registration_point_CT)
        centerline_registration_start = centerline_registration_point_selector.selected_point_index_red
        selected_registration_point_CT = np.array(centerline_registration_point_selector.selected_registration_point_CT)
        oct_lumen_rotation_matrix, rotated_registration_point_OCT = center_line_registrator.get_oct_lumen_rotation_matrix(centerline_registration_point_selector.selected_point_index_blue, self.OCT_start_frame, registration_point_CT, resampled_pc_centerline, centerline_registration_start, grouped_OCT_lumen, 
                                                                                                                          registration_point_OCT, selected_registration_point_CT, self.OCT_registration_frame, self.z_distance, self.display_results)

        if self.display_results or False:
            #------------------------------------------#
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Show the plot
            x_filtered = []
            y_filtered = []
            z_filtered = []
            for i, frame_points in enumerate(resampled_pc_centerline):
                if i != centerline_registration_start:
                    if i != centerline_registration_point_selector.selected_point_index_blue:
                        x_filtered.append(frame_points[0])
                        y_filtered.append(frame_points[1])
                        z_filtered.append(frame_points[2])
            ax.scatter(x_filtered, y_filtered, z_filtered, c="black", marker='o', s=8)
            #ax.scatter(rotated_registration_point_OCT[0], rotated_registration_point_OCT[1], rotated_registration_point_OCT[2], c="yellow", marker='o', s=3)
            ax.scatter(selected_registration_point_CT[0], selected_registration_point_CT[1], selected_registration_point_CT[2], c="grey", marker='o', s=20)
            ax.scatter(resampled_pc_centerline[centerline_registration_start][0], resampled_pc_centerline[centerline_registration_start][1], resampled_pc_centerline[centerline_registration_start][2], c="red", marker='o', s=20)
            ax.scatter(resampled_pc_centerline[centerline_registration_point_selector.selected_point_index_blue][0], resampled_pc_centerline[centerline_registration_point_selector.selected_point_index_blue][1], resampled_pc_centerline[centerline_registration_point_selector.selected_point_index_blue][2], c="blue", marker='o', s=20)

            # Extract the coordinates of the two points
            x_values = [resampled_pc_centerline[centerline_registration_start][0], 
                        resampled_pc_centerline[centerline_registration_point_selector.selected_point_index_blue][0]]
            y_values = [resampled_pc_centerline[centerline_registration_start][1], 
                        resampled_pc_centerline[centerline_registration_point_selector.selected_point_index_blue][1]]
            z_values = [resampled_pc_centerline[centerline_registration_start][2], 
                        resampled_pc_centerline[centerline_registration_point_selector.selected_point_index_blue][2]]

            # Plot the line connecting the two points
            ax.plot(x_values, y_values, z_values, c='black')

            from matplotlib.lines import Line2D

            legend_elements = [Line2D([0], [0], marker='o', color='w', label='Center-line', markerfacecolor='black', markersize=10),
                                Line2D([0], [0], marker='o', color='w', label='Registration point CT', markerfacecolor='grey', markersize=10),
                                Line2D([0], [0], marker='o', color='w', label='Selected center-line point branch of interest', markerfacecolor='red', markersize=10),
                                Line2D([0], [0], marker='o', color='w', label='Selected center-line point side branch', markerfacecolor='blue', markersize=10)]

            ax.legend(handles=legend_elements, loc='best')

            # Remove grid
            ax.grid(False)

            # Remove axes
            ax.set_axis_off()

            # Set background color to white
            ax.set_facecolor('white')
            plt.show()
            #------------------------------------------#

        # rotate OCT_frames
        rotated_grouped_OCT_lumen = center_line_registrator.rotate_frames(grouped_OCT_lumen, oct_lumen_rotation_matrix, self.display_results)

        if processing_info == STENT or processing_info == STENT_AND_CALC:
            rotated_grouped_OCT_stent = center_line_registrator.rotate_frames(grouped_stent, oct_lumen_rotation_matrix, self.display_results)

        if processing_info == CALC or processing_info == STENT_AND_CALC:
            rotated_grouped_OCT_calc = center_line_registrator.rotate_frames(grouped_calc, oct_lumen_rotation_matrix, self.display_results)
      
        if self.display_results or False:
            #------------------------------------------#
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Show the plot
            x_filtered = []
            y_filtered = []
            z_filtered = []
            for frame_points in oct_lumen_contours:
                for i, data in enumerate(frame_points):
                    if i // 1000 % 20 == 0:
                        x_filtered.append(data[0])
                        y_filtered.append(data[1])
                        z_filtered.append(data[2])
            ax.scatter(x_filtered[::20], y_filtered[::20], z_filtered[::20], c="blue", marker='o')
            x_filtered = []         
            y_filtered = []
            z_filtered = []
            for frame_points in oct_lumen_point_cloud:
                for i, data in enumerate(frame_points):
                    if i // 1000 % 20 == 0:
                        x_filtered.append(data[0])
                        y_filtered.append(data[1])
                        z_filtered.append(data[2])
            ax.scatter(x_filtered[::20], y_filtered[::20], z_filtered[::20], c="red", marker='o')
            max_z = max(z_filtered)

            # Plot the line marking the Z-axis
            place = 512*self.conversion_factor
            ax.plot([place, place], [place, place], [0, max_z], c='black')
            ax.set_xlabel('Px')
            ax.set_ylabel('Py')
            ax.set_zlabel('Pz')
            x_filtered = []
            y_filtered = []
            z_filtered = []
        
            from matplotlib.lines import Line2D

            legend_elements = [Line2D([0], [0], marker='o', color='w', label='OCT lumen contours unaligned', markerfacecolor='blue', markersize=10),
                                Line2D([0], [0], marker='o', color='w', label='OCT lumen contours aligned', markerfacecolor='red', markersize=10)]
            ax.legend(handles=legend_elements, loc='best')

            # Remove grid
            ax.grid(False)

            # Remove axes
            ax.set_axis_off()

            # Set background color to white
            ax.set_facecolor('white')
            plt.show()
            #------------------------------------------#

        #######################################################################
        #image_visualizer = OctImageVisualizier()
        #image_visualizer.visualize_images(self.input_file_OCT, centerline_vectors, rotated_grouped_OCT_lumen, centerline_registration_start, resampled_pc_centerline, oct_lumen_rotation_matrix, oct_rotation_angles, self.OCT_start_frame, self.OCT_end_frame, self.OCT_registration_frame, self.crop_bottom, self.crop_top, self.conversion_factor, self.z_distance)
        ####################################################################### 

        #register frames onto centerline
        if processing_info == BASIC:
            registered_oct_lumen = center_line_registrator.register_OCT_frames_onto_centerline(rotated_grouped_OCT_lumen, centerline_registration_start, centerline_vectors,
                                                                                                resampled_pc_centerline, self.OCT_registration_frame, self.OCT_start_frame, self.z_distance, rotated_registration_point_OCT, self.save_file, self.display_results)
        if processing_info == STENT or processing_info == STENT_AND_CALC:
            registered_oct_lumen, registered_oct_stent = center_line_registrator.register_OCT_frames_onto_centerline_stent(rotated_grouped_OCT_lumen, rotated_grouped_OCT_stent,centerline_registration_start, centerline_vectors,
                                                                                                resampled_pc_centerline, self.OCT_registration_frame, self.OCT_start_frame, self.z_distance, rotated_registration_point_OCT, self.save_file, self.display_results)
        if processing_info == CALC or processing_info == STENT_AND_CALC:
            registered_oct_lumen, registered_oct_calc = center_line_registrator.register_OCT_frames_onto_centerline_calc(rotated_grouped_OCT_lumen, rotated_grouped_OCT_calc, centerline_registration_start, centerline_vectors,
                                                                                                resampled_pc_centerline, self.OCT_registration_frame, self.OCT_start_frame, self.z_distance, rotated_registration_point_OCT, self.save_file, self.display_results)

        if self.display_results or False:
            #------------------------------------------#
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Show the plot
            x_filtered = []
            y_filtered = []
            z_filtered = []
            colors = []

            target = len(registered_oct_lumen)/500/2
            for i, frame_points in enumerate(registered_oct_lumen):
                if i // 500 % 3 == 0:
                    x_filtered.append(frame_points[0])
                    y_filtered.append(frame_points[1])
                    z_filtered.append(frame_points[2])
                    if target*500 - 750 <= i < target*500 + 750:
                        colors.append('blue')
                    else:
                        colors.append('red')

            ax.scatter(x_filtered, y_filtered, z_filtered, c=colors, marker='o', s=2)
            ax.set_xlabel('Px')
            ax.set_ylabel('Py')
            ax.set_zlabel('Pz')
            x_filtered = []
            y_filtered = []
            z_filtered = []
            for i, frame_points in enumerate(resampled_pc_centerline):
                if (i+2) % 3 == 0:
                    x_filtered.append(frame_points[0])
                    y_filtered.append(frame_points[1])
                    z_filtered.append(frame_points[2])
            ax.scatter(x_filtered, y_filtered, z_filtered, c="black", marker='o', s=2)
            #ax.plot(x_filtered, y_filtered, z_filtered, c="black")  # This line connects the points

            from matplotlib.lines import Line2D

            legend_elements = [Line2D([0], [0], marker='o', color='w', label='OCT lumen contours', markerfacecolor='red', markersize=10),
                                Line2D([0], [0], marker='o', color='w', label='Center-line', markerfacecolor='black', markersize=10), 
                                Line2D([0], [0], marker='o', color='w', label='OCT registration lumen contour', markerfacecolor='blue', markersize=10)]
            ax.legend(handles=legend_elements, loc='best')

            # Remove grid
            ax.grid(False)

            # Remove axes
            ax.set_axis_off()

            # Set background color to white
            ax.set_facecolor('white')
            plt.show()
            #------------------------------------------#

        # Update instructions
        instruction_text = "Delete overlapping point form the pointcloud, then save it."
        self.add_instruction_text(instruction_text)

        # Load the .PLY file
        mesh = o3d.io.read_triangle_mesh(self.input_file_ct_mesh)

        # Decimate the mesh (optional)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)   #100000

        # Create a regular point cloud from the mesh with a smaller voxel size
        voxel_size = 0.05  # Adjust this value to control point density (smaller values = denser point cloud)
        pcd = mesh.sample_points_uniformly(number_of_points=int(mesh.get_surface_area() / voxel_size))

        # Extract the points as a numpy array
        ct_points_ = pcd.points

        ct_points = []
        for point in ct_points_:
            ct_points.append([point[0], point[1], point[2]])
        ct_points = np.array(ct_points)

        # Visual point cloud editing:
        point_cloud_visual_editior = point_cloud_visual_editing()
        point_cloud_visual_editior.run_editor(ct_points, registered_oct_lumen)

        # Save the fused point cloud to a text file
        point_cloud_save = point_cloud_visual_editior.fused_point_cloud
        np.savetxt(self.path_fused_point_cloud, point_cloud_save, fmt='%f %f %f')
        if processing_info == CALC or processing_info == STENT_AND_CALC:
            np.savetxt(self.path_point_cloud_calc, registered_oct_calc, fmt='%f %f %f')
        if processing_info == STENT or processing_info == STENT_AND_CALC:
            np.savetxt(self.path_point_cloud_stent, registered_oct_stent, fmt='%f %f %f')

        # Update instructions
        instruction_text = "Continue with the software Meshlab. Import the outputfile.xyz (from ArCoMoX/output/) into the software and open the instructions for the following steps. You can close this window now."
        self.add_instruction_text(instruction_text)

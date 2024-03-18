from scipy.interpolate import splprep, splev
import os
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import rotate
from PIL import Image
import cv2
import numpy as np
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import cv2 as cv
import numpy as np
import math
from dijkstar import Graph, find_path

from workflow_stent_calc_segmentation_calc import ContourDrawer

class oct_extraction:
    def __init__(self):
        self.oct_registration_point_x = 0
        self.oct_registration_point_y = 0
        self.retPt = []
        self.choice = "Use Existing"

    # Function to filter by RGB color
    def filter_color(self, input_image, color1, color2):
        # Create masks for the two colors
        lower_color1 = np.array([color1[2], color1[1], color1[0]], dtype=np.uint8)
        upper_color1 = np.array([color1[2], color1[1], color1[0]], dtype=np.uint8)
        # Little offset given to ensure color of line-dots are included
        lower_color2 = np.array([color2[2] - 1, color2[1] - 1, color2[0] - 1], dtype=np.uint8)
        upper_color2 = np.array([color2[2] + 1, color2[1] + 1, color2[0] + 1], dtype=np.uint8)

        mask1 = cv2.inRange(input_image, lower_color1, upper_color1)
        mask2 = cv2.inRange(input_image, lower_color2, upper_color2)

        # Combine the masks to create a binary mask
        binary_mask = cv2.bitwise_or(mask1, mask2)
        ##### PHANTOM MODEL ############################3
        if False:
            color = [255, 255, 255]
            lower_color2 = np.array([color[2] - 1, color[1] - 1, color[0] - 1], dtype=np.uint8)
            upper_color2 = np.array([color[2] + 1, color[1] + 1, color[0] + 1], dtype=np.uint8)
            mask2 = cv2.inRange(input_image, lower_color2, upper_color2)
            # Combine the masks to create a binary mask
            binary_mask = cv2.bitwise_or(mask1, mask2)
        ##### PHANTOM MODEL ############################3

        return binary_mask

    # Function to filter by RGB color
    def filter_color_one(self, input_image, color):
        # Create masks for the two colors
        color = np.array([color[2], color[1], color[0]], dtype=np.uint8)


        binary_mask = cv2.inRange(input_image, color, color)
        # Combine the masks to create a binary mask

        return binary_mask

    # Function to smooth the image and create an intensity threshold mask (thin line removal)
    def smooth_and_threshold(self, image, smoothing_kernel_size, threshold_value):
        # Smooth the image using Gaussian blur
        blurred = cv2.GaussianBlur(image, (smoothing_kernel_size, smoothing_kernel_size), 0)

        # Create an intensity threshold mask
        _, mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

        return mask


    def compute_transformation_matrix(self, source_points_orig, source_points):

        # Compute the translation by finding the difference between the centroids.
        centroid_orig = np.mean(source_points_orig, axis=0)
        centroid_transformed = np.mean(source_points, axis=0)
        translation = centroid_transformed - centroid_orig

        # Calculate the rotation matrix using Procrustes analysis.
        H = np.dot(source_points_orig.T, source_points)
        U, _, Vt = np.linalg.svd(H)
        rotation_matrix = np.dot(U, Vt)

        # Create the 2x3 transformation matrix.
        transformation_matrix = np.column_stack((rotation_matrix, translation))

        return transformation_matrix


    def letter_A_removal(self, binary_mask, letter_mask_path):
        # Load the binary mask and the letter 'A' mask
        letter_mask = cv2.imread(letter_mask_path, cv2.IMREAD_GRAYSCALE)

        # Find the position of 'A' in the binary mask
        result = cv2.matchTemplate(binary_mask, letter_mask, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Determine the position of the letter 'A' and its size
        letter_width, letter_height = letter_mask.shape[::-1]
        top_left = max_loc
        bottom_right = (top_left[0] + letter_width, top_left[1] + letter_height)

        # Remove 'A' from the binary mask
        binary_mask_removed = binary_mask.copy()
        cv2.rectangle(binary_mask_removed, top_left, bottom_right, 0, -1)  # Fills the region with black

        return binary_mask_removed


    def close_binary_mask(self, binary_mask, display_images):
        # Find contours in the binary mask.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 1:
            return None
        # Assuming there's only one contour.
        contour = contours[0]

        # Calculate the convex hull of the contour.
        convex_hull = cv2.convexHull(contour)

        # Create a new binary mask with the closed contour.
        closed_mask = np.zeros_like(binary_mask)
        cv2.drawContours(closed_mask, [convex_hull], -1, 255, thickness=cv2.FILLED)

        if display_images:
            self.plt_cv2_images('Closed mask image', closed_mask)

        return closed_mask


    def spline_fitting(self, binary_mask, display_images, save_images_for_controll, page):
        # Apply morphological operations to close gaps in the binary mask.
        kernel = np.ones((30, 30), np.uint8)
        closed_mask_ = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        closed_mask = self.close_binary_mask(closed_mask_, display_images)
        # Check if there is a contour, if not, go back
        if closed_mask is None:
            return None

        # Find contours in the closed binary mask.
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Assuming there's only one contour.
        contour = contours[0]

        # Convert the contour points to NumPy array.
        contour_points = np.array(contour)[:, 0, :]

        # Fit a spline to the contour points.
        # Perfect matching spline
        tck1, _ = splprep(contour_points.T, s=0, per=True)
        # Smoothed spline
        tck2, _ = splprep(contour_points.T, s=1000, per=True)

        # Evaluate the spline to get points on the fitted ellipse.
        u = np.linspace(0, 1, 1000)
        x1, y1 = splev(u, tck1)
        x2, y2 = splev(u, tck2)

        fitted_contour = list(zip(x2, y2))
        
        if display_images:
            # Create an all black image with the same size as the binary mask image
            black_image = np.zeros_like(binary_mask)

            # Draw the fitted contour on the black image
            cv2.drawContours(black_image, [np.array(fitted_contour, dtype=np.int32)], -1, (255, 255, 255), 2)

            # Display the black image with the fitted contour
            cv2.imshow("Black Image with Fitted Contour", black_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()     
        return fitted_contour


    def plt_cv2_images(self, title, image):
            cv2.imshow(title, image)
            cv2.waitKey(0)  # Wait for a key press to move to the next image
            cv2.destroyAllWindows()  # Close the image window


    def frames_alignment_calc(self, contours, rotation_matrix, z_offset, height, width, conversion_factor):
        rotated_contours = []

        # Calculate the center of rotation
        center_y = (width / 2) * conversion_factor
        center_x = (height / 2) * conversion_factor
        
        # adapt rotation matrix
        extended_rotation_matrix = [angle for angle in rotation_matrix for _ in range(2)]


        for current_contour in contours: 
            current_contour = np.array(current_contour)
            
            idx = int(current_contour[0][2]/z_offset)

            rot_angle = extended_rotation_matrix[idx]  # Convert the list to a numpy array
            if current_contour is not None:

                    # Translate to the center of rotation
                current_contour[:, 0] -= center_x
                current_contour[:, 1] -= center_y

                c, s = math.cos(np.radians(rot_angle)), math.sin(np.radians(rot_angle))
            
                # 2D rotation matrix for x-y plane
                rot_xy = np.array([[c, -s],
                                [s, c]])
                
                # Apply rotation only to x-y coordinates
                current_contour_xy_rotated = np.dot(current_contour[:, :2], rot_xy.T)
                
                # Translate back to the original position
                current_contour_xy_rotated[:, 0] += center_x
                current_contour_xy_rotated[:, 1] += center_y

                # Combine rotated x-y coordinates with original z-values
                current_contour_rotated = np.column_stack((current_contour_xy_rotated, current_contour[:, 2]))
                
                rotated_contours.append(current_contour_rotated)
    
        return np.array(rotated_contours)
    
    def frames_alignment(self, contours, rotation_matrix, z_offset, height, width, conversion_factor):
        rotated_contours = []

        # Calculate the center of rotation
        center_y = (width / 2) * conversion_factor
        center_x = (height / 2) * conversion_factor

        # Case for rotation point, would not work with other code, due to for condition
        if len(contours) == 3: # for registration point
                current_contour = np.array(contours)
                idx = int(current_contour[2]/z_offset)
                rot_angle = rotation_matrix[idx]  # Convert the list to a numpy array
                
                if current_contour is not None:
                    # Translate to the center of rotation
                    current_contour[0] -= center_x
                    current_contour[1] -= center_y

                    c, s = math.cos(np.radians(rot_angle)), math.sin(np.radians(rot_angle))

                    # 2D rotation matrix for x-y plane
                    rot_xy = np.array([[c, -s],
                                    [s, c]])
                    
                    # Apply rotation only to x-y coordinates
                    current_contour_xy_rotated = np.dot(current_contour[0:2], rot_xy.T)
                    
                    # Translate back to the original position
                    current_contour_xy_rotated[0] += center_x
                    current_contour_xy_rotated[1] += center_y
        
                    # Combine rotated x-y coordinates with original z-values
                    current_contour_rotated = [current_contour_xy_rotated[0], current_contour_xy_rotated[1], current_contour[2]]

                    
                    rotated_contours.append(current_contour_rotated)
        
        else:
            for current_contour in contours: # for stents
                current_contour = np.array(current_contour)
                if len(current_contour) == 3:
                    idx = int(current_contour[2]/z_offset)

                    rot_angle = rotation_matrix[idx]  # Convert the list to a numpy array
                    if current_contour is not None:

                        # Translate to the center of rotation
                        current_contour[0] -= center_x
                        current_contour[1] -= center_y

                        c, s = math.cos(np.radians(rot_angle)), math.sin(np.radians(rot_angle))
                        
                        # 2D rotation matrix for x-y plane
                        rot_xy = np.array([[c, -s],
                                        [s, c]])
                        
                    
                    # Apply rotation only to x-y coordinates
                    current_contour_xy_rotated = np.dot(current_contour[0:2], rot_xy.T)
                    
                    # Translate back to the original position
                    current_contour_xy_rotated[0] += center_x
                    current_contour_xy_rotated[1] += center_y

                    # Combine rotated x-y coordinates with original z-values
                    current_contour_rotated = [current_contour_xy_rotated[0], current_contour_xy_rotated[1], current_contour[2]]
                        
                    rotated_contours.append(current_contour_rotated)
                
                else: # for normal contours
                    idx = int(current_contour[0][2]/z_offset)
                    rot_angle = rotation_matrix[idx]  # Convert the list to a numpy array
                    if current_contour is not None:

                         # Translate to the center of rotation
                        current_contour[:, 0] -= center_x
                        current_contour[:, 1] -= center_y

                        c, s = math.cos(np.radians(rot_angle)), math.sin(np.radians(rot_angle))
                    
                        # 2D rotation matrix for x-y plane
                        rot_xy = np.array([[c, -s],
                                        [s, c]])
                        
                        # Apply rotation only to x-y coordinates
                        current_contour_xy_rotated = np.dot(current_contour[:, :2], rot_xy.T)
                        
                        # Translate back to the original position
                        current_contour_xy_rotated[:, 0] += center_x
                        current_contour_xy_rotated[:, 1] += center_y

                        # Combine rotated x-y coordinates with original z-values
                        current_contour_rotated = np.column_stack((current_contour_xy_rotated, current_contour[:, 2]))
                        
                        rotated_contours.append(current_contour_rotated)
        
        return np.array(rotated_contours)
    

    def click(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates
        if event == cv.EVENT_LBUTTONDOWN:
            self.retPt = [(x, y)]
        elif event == cv.EVENT_LBUTTONUP:
            self.retPt.append((x, y))
        # record the ending (x, y) coordinates
            
    def fit_spline_to_path(self, path, z_value, conversion_factor):
            # Extract x and y coordinates from the path
            concatenated_path = [point for subpath in path for point in subpath]
            path_x, path_y = zip(*concatenated_path)
            plt.plot(path_x, path_y)
            plt.show()
            try:
                # Fit a spline to the path
                tck, _ = splprep([path_x, path_y], s=1000, per=True)

                # Evaluate the spline to get points on the fitted spline
                u = np.linspace(0, 1, 500)  # Adjust the number of points as needed
                x, y = splev(u, tck)
                z_values = [z_value] * len(x)
                plt.plot(x, y)
                plt.show()
                # Return the fitted spline as a list of points
                return list(zip(x*conversion_factor, y*conversion_factor, z_values))
            except: 
                print("could not fit spline")
                return []


    def apply_gaussian_blur(self, image):
        return cv.GaussianBlur(image, (5, 5), 0)


    def segment_calc_live_wire(self, input_file, OCT_start_frame, OCT_end_frame, crop_top, z_distance, conversion_factor):
        saved_paths = []
        saved_splines = []
        with Image.open(input_file) as im:
            for page in range(105, 106, 1):
                next_slide = False
                path_total = []
                path = []
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                image_flipped = np.flipud(image)
                height, width, channels = image_flipped.shape
                open_cv_image = image_flipped[0: height-crop_top, :, ::-1].copy()  # Crop image at xxx pixels from top


                # Apply Gaussian blur to the image
                #img_filtered = apply_gaussian_blur(img_orig)
                #cv.imwrite("smoothed_image.jpg", img_filtered)

                img = open_cv_image

                final = img.copy()
                img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                vertex = img.copy()
                h,w = img.shape[1::-1]

                graph = Graph(undirected=True)

                # Iterating over an image and avoiding boundaries
                for i in range (1, w-1):
                    for j in range(1, h-1):
                        G_x = float(vertex[i,j]) - float(vertex[i,j+1])    # Center - right
                        G_y = float(vertex[i,j]) - float(vertex[i+1, j])   # Center - bottom
                        G = np.sqrt((G_x)**2 + (G_y)**2)
                        theeta = 0.0
                        if (G_x > 0 or G_x < 0):
                            theeta = math.atan(G_y/G_x)
                        # Theeta is rotated in clockwise direction (90 degrees) to align with edge
                        theeta_a = theeta + math.pi/2
                        G_x_a = abs(G * math.cos(theeta_a)) + 0.00001
                        G_y_a = abs(G * math.sin(theeta_a)) + 0.00001
                        
                        # Strongest Edge will have lowest weights
                        W_x = 1/G_x_a
                        W_y = 1/G_y_a
                        
                        # Assigning weights
                        graph.add_edge((i,j), (i,j+1), W_x) # W_x is given to right of current vertex
                        graph.add_edge((i,j), (i+1,j), W_y) # W_y is given to bottom of current vertex

                # Opens image select the points using mouse and press c to done
                cv.namedWindow("image")
                while True:
                    while True:
                        cv.setMouseCallback("image", self.click)
                        cv.imshow("image", final)
                        key = cv.waitKey(2) & 0xFF

                        if key == ord("c"):
                            # Continue with the existing path selection logic
                            if len(self.retPt) > 0:
                                break

                        elif key == ord("r") or key == ord("R"):
                            # Reset path if the "R" key is pressed
                            final = img.copy()
                            path_total = []
                            path = []

                        elif key == ord("s") or key == ord("S"):
                            # Save the path if the "S" key is pressed
                            if len(path_total) > 1:
                                # Fit a spline to the path and save the 500 points
                                fitted_spline = self.fit_spline_to_path(path_total, (page-OCT_start_frame)*z_distance, conversion_factor)
                                saved_splines.append(fitted_spline)
                                print("Spline saved!")
                            next_slide = True
                            break
                    if next_slide:
                        cv.destroyWindow('image')
                        break
                    # Gets the start and ending points in image format
                    startPt = (self.retPt[0][1], self.retPt[0][0])
                    endPt = (self.retPt[1][1], self.retPt[1][0])

                    # Find_path[0] returns nodes it traveled for the shortest path
                    path = find_path(graph, startPt, endPt)[0]
                    path_total.append(path)
                    if path is None:
                        break

                    # Turn those visited nodes to white
                    for i in range(len(path)):
                        final[path[i][0], path[i][1]] = 255

                    cv.imshow('ImageWindow', final)
                    cv.waitKey(0)
                    cv.destroyWindow('ImageWindow')

                    # Clear retPt for the next iteration
                    self.retPt = []
        
        return saved_splines
    

    def close_dialog(self, dialog, choice):
        print(f"You chose: {choice}")
        self.choice = choice
        dialog.destroy()
        self.window.destroy()  


    def custom_dialog(self):
        self.window = tk.Tk()  
        self.window.withdraw()  

        # Create a new top-level window
        dialog = tk.Toplevel(self.window) 

        # Add your message
        tk.Label(dialog, text="A segmentation for this case already exists. What do you want to do?").pack()

        # Add your buttons
        tk.Button(dialog, text="Use Existing", command=lambda: self.close_dialog(dialog, "Use Existing")).pack()
        tk.Button(dialog, text="Edit", command=lambda: self.close_dialog(dialog, "Edit")).pack()
        tk.Button(dialog, text="New", command=lambda: self.close_dialog(dialog, "New")).pack()

        self.window.mainloop()


    def get_calc_contours(self, path_segmented_calc, input_file, OCT_start_frame, OCT_end_frame, z_space, conversion_factor, crop_top, crop_bottom):
        # Check if there is already a calc file, if yes, ask the user if he wants to use this or if he wants to do a new segmentation
        existing_contours = []
        if os.path.exists(path_segmented_calc):
            # Ask the user if they want to use the existing file, perform a new segmentation, or edit the existing file
            choice_existing = messagebox.askquestion("File Exists", "A segmentation for this case already exists. Do you want to use this file for segmentation?",
                                            icon='warning')
            
            if choice_existing == 'yes':
                # Parse calc
                results = []
                with open(path_segmented_calc, 'r') as file:
                    for line in file:
                        # Parse each line and convert it back to the original data structure
                        path = eval(line.strip())
                        results.append(path)
                calc_contours = results

            elif choice_existing == 'no':
                edit_choice = messagebox.askquestion("Edit or start a new", "Do you want to edit the exisintg segmentation, if no you will start a new?",
                                            icon='warning')
                if edit_choice == 'yes':
                    # Edit existing calc
                    with open(path_segmented_calc, 'r') as file:
                        for line in file:
                            # Parse each line and convert it back to the original data structure
                            path = eval(line.strip())
                            existing_contours.append(path)

                    drawer = ContourDrawer()
                    calc_contours = drawer.segment_calc(input_file, OCT_start_frame, OCT_end_frame, crop_top, crop_bottom, z_space, conversion_factor, existing_contours)

                    # Save the file
                    with open(path_segmented_calc, 'w') as file:
                        for path in calc_contours:
                            # Write each path to the file
                            file.write(str(path) + '\n')
            
                elif edit_choice == 'no':
                    # Segment new calc
                    drawer = ContourDrawer()
                    calc_contours = drawer.segment_calc(input_file, OCT_start_frame, OCT_end_frame, crop_top, crop_bottom, z_space, conversion_factor, existing_contours)

                    # Save the file
                    with open(path_segmented_calc, 'w') as file:
                        for path in calc_contours:
                            # Write each path to the file
                            file.write(str(path) + '\n')

        else:
            # File doesn't exist, perform a new segmentation
            drawer = ContourDrawer()
            calc_contours = drawer.segment_calc(input_file, OCT_start_frame, OCT_end_frame, crop_top, crop_bottom, z_space, conversion_factor, existing_contours)
            
            # Save the file
            with open(path_segmented_calc, 'w') as file:
                for path in calc_contours:
                    # Write each path to the file
                    file.write(str(path) + '\n')

        # Add more frames, so interframes distance is smaller
        #calc_contours_augmented = self.process_frames(calc_contours, z_space)
        return calc_contours

    def process_frames(self, frames_list, z_distance):
        # Initialize a new list to store the modified frames
        modified_frames = []

        # Iterate through each frame in the original list
        for frame in frames_list:
            # Extract x, y, and z values from the frame
            x_values = [x[0] for x in frame]
            y_values = [y[1] for y in frame]
            z_values = [z[2] for z in frame]

           # Copy the original frame and append it to the modified_frames list
            modified_frames.append(list(zip(x_values, y_values, z_values)))

            # Copy the original frame and increment the z-value by z_value/3 and append it
            modified_frames.append(list(zip(x_values, y_values, [z + z_distance/2 for z in z_values])))

            # Copy the original frame and increment the z-value by 2 * z_value/3 and append it
            #modified_frames.append(list(zip(x_values, y_values, [z + 2 * z_distance/3 for z in z_values])))

        return modified_frames     

    def get_stent_contours(self, input_file, OCT_start_frame, OCT_end_frame, crop_top, crop_bottom, z_space, conversion_factor):
        color = [255, 255, 255]
        points = []
        with Image.open(input_file) as im:
            for page in range(OCT_start_frame, OCT_end_frame, 1):
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                image_flipped = np.flipud(image)
                height, width, channels = image_flipped.shape
                open_cv_image = image_flipped[crop_bottom: height-crop_top, :, ::-1].copy()  # Crop image at xxx pixels from top

                # Apply the color filter
                binary_mask = self.filter_color_one(open_cv_image, color)
                
            # Find contours in the binary mask
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours is not None:
                    # Create lists to store x and y coordinates of centroids
                    centroid_x_list = []
                    centroid_y_list = []
                    # Iterate through each contour
                    for contour in contours:
                        # Calculate the centroid of the contour
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            centroid_x = int(M["m10"] / M["m00"])
                            centroid_y = int(M["m01"] / M["m00"])

                            # Append centroid coordinates to lists
                            centroid_x_list.append(centroid_x)
                            centroid_y_list.append(centroid_y)
                            points.append([centroid_x*conversion_factor, centroid_y*conversion_factor, (page - OCT_start_frame)*z_space])
        
        return points
    

    def get_lumen_contour(self, crop_top, crop_bottom, input_file, OCT_end_frame, OCT_start_frame, OCT_registration_frame, z_offset, conversion_factor, save_file, color1, color2, smoothing_kernel_size,
                        threshold_value, display_images, save_images_for_controll):
        z_coordinate = 0
        point_cloud = []
        with Image.open(input_file) as im:
            for page in range(OCT_start_frame, OCT_end_frame, 1):
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                image_flipped = np.flipud(image)
                height, width, channels = image_flipped.shape
                open_cv_image = image_flipped[crop_bottom: height-crop_top, :, ::-1].copy()  # Crop image at xxx pixels from top

                # Apply the color filter
                binary_mask = self.filter_color(open_cv_image, color1, color2)
                if display_images:
                    self.plt_cv2_images('Color filtered image', binary_mask)

                # Remove letter A filter
                binary_mask_A_removed = self.letter_A_removal(binary_mask, letter_mask_path="workflow_utils/Image_a.jpg")
                if display_images:
                    self.plt_cv2_images('Color filtered image', binary_mask_A_removed)

                # Apply smoothing and thresholding (thin line removal)
                threshold_mask = self.smooth_and_threshold(binary_mask_A_removed, smoothing_kernel_size, threshold_value)
                if display_images:
                    self.plt_cv2_images('Thresholded image', threshold_mask)

                # fit a spline to the contour
                current_contour = self.spline_fitting(threshold_mask, display_images, save_images_for_controll, page)
                
                if current_contour is not None:
                    aligned_source_points = np.array(current_contour)

                    
                    # Create a 3D point cloud with incremental z-coordinate and convert into mm unit.
                    point_cloud_current = [(xi * conversion_factor, yi * conversion_factor, z_coordinate) for xi, yi in
                                        aligned_source_points]                 
                    
                    point_cloud.append(point_cloud_current)

                z_coordinate += z_offset  # Increment z-coordinate

        return point_cloud
    
    # Function to handle mouse clicks
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'Registration point selected at (x={x}, y={y})')
            self.oct_registration_point_x = x
            self.oct_registration_point_y = y
             # Close the OpenCV window
            cv2.destroyAllWindows()


    def get_registration_point(self, input_file, crop_top, crop_bottom,  OCT_start_frame, OCT_registration_frame, display_images, z_offset, save_file, conversion_factor):
        with Image.open(input_file) as im:
            im.seek(OCT_registration_frame)  # Move to marked page (frame)
            image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
            image_flipped = np.flipud(image)
            height, width, channels = image_flipped.shape
            open_cv_image = image_flipped[crop_bottom: height-crop_top, :, ::-1].copy()

            # Display the cropped image
            cv2.imshow('Select registration point (left mouse click)', open_cv_image)            

            # Set the mouse callback function
            cv2.setMouseCallback('Select registration point (left mouse click)', self.mouse_callback)
            # Wait for the user to click on the image
            cv2.waitKey()

        # Convert into mm unit
        registration_point = [self.oct_registration_point_x*conversion_factor, self.oct_registration_point_y*conversion_factor, (OCT_registration_frame-OCT_start_frame)*z_offset]
        return registration_point
    

    def get_rotation_matrix(self, input_file, OCT_start_frame, OCT_end_frame):
        rotation_angles = []
        rotation_total = 0
        previous_image = None
        with Image.open(input_file) as im:
            for page in range(OCT_start_frame, OCT_end_frame, 1):
                im.seek(page)
                image = np.array(im.convert('RGB'))
                image_flipped = image #np.flipud(image)
                height, width, channels = image_flipped.shape
                open_cv_image = image_flipped[300: height-300, 300: width-300, ::-1].copy()

                h, w, _ = open_cv_image.shape
                if previous_image is not None:
                    im1 = Image.fromarray(open_cv_image)            
                    im2 = Image.fromarray(previous_image)

                    mse = []
                    for i in range(-30, 30):
                        im2_rot = im2.rotate(i/10)
                        mse.append(self.rmsdiff(im1,im2_rot))

                    print((mse.index(min(mse))-30)/10)
                    rotation_total += (mse.index(min(mse))-30)/10

                previous_image = open_cv_image
                rotation_angles.append(rotation_total)
        
        return np.array(rotation_angles)

    def rmsdiff(self, x, y):
        """Calculates the root mean square error (RSME) between two images"""
        errors = np.asarray(ImageChops.difference(x, y)) / 255
        return math.sqrt(np.mean(np.square(errors)))

###############################################################################################
    def get_rotation_matrix_ICP(self, oct_lumen_contours, z_distance):
        previous_contour = None
        rotations = []
        total_rotations = []
        rotation_total = 0
        rotation = 0
        z_previous = 0
        flag = True
        for current_contour in oct_lumen_contours:
            z_current = current_contour[0][2]
            while True:
                z_diff = z_current - z_previous
                current_contour = [(x[0], x[1]) for x in current_contour]
                
                if (z_diff) < (z_distance + 0.01):
                    if previous_contour is not None:
                        flag = False
                        # Perform ICP alignment.
                        rotation = self.icp_alignment(current_contour, previous_contour)
                        previous_contour = current_contour
                        rotation_total += rotation
                        rotations.append(rotation)
                        total_rotations.append(rotation_total)
                        z_previous = z_current
                        break
                    else:
                        previous_contour = current_contour
                        rotation = 0
                        rotation_total += rotation
                        rotations.append(rotation)
                        total_rotations.append(rotation_total)
                        break
                else:
                    z_previous += z_distance
                    rotation = 0
                    rotation_total += rotation
                    rotations.append(rotation)
                    total_rotations.append(rotation_total)
            
        return np.array(total_rotations)
    
    # Estimate rotation and translations
    def point_based_matching(self, point_pairs):
        """
        This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
        by F. Lu and E. Milios.

        :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
        :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
        """

        x_mean = 0
        y_mean = 0
        xp_mean = 0
        yp_mean = 0
        n = len(point_pairs)

        if n == 0:
            return None, None, None

        for pair in point_pairs:

            (x, y), (xp, yp) = pair

            x_mean += x
            y_mean += y
            xp_mean += xp
            yp_mean += yp

        x_mean /= n
        y_mean /= n
        xp_mean /= n
        yp_mean /= n

        s_x_xp = 0
        s_y_yp = 0
        s_x_yp = 0
        s_y_xp = 0
        for pair in point_pairs:

            (x, y), (xp, yp) = pair

            s_x_xp += (x - x_mean)*(xp - xp_mean)
            s_y_yp += (y - y_mean)*(yp - yp_mean)
            s_x_yp += (x - x_mean)*(yp - yp_mean)
            s_y_xp += (y - y_mean)*(xp - xp_mean)

        rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
        translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
        translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

        return rot_angle, translation_x, translation_y
    

    def icp_alignment(self, points, reference_points, max_iterations=100, distance_threshold=100, convergence_translation_threshold=1e-3,
            convergence_rotation_threshold=1e-4, point_pairs_threshold=10, verbose=False):
        """
        An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
        of N 2D (reference) points.

        :param reference_points: the reference point set as a numpy array (N x 2)
        :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2)
        :param max_iterations: the maximum number of iteration to be executed
        :param distance_threshold: the distance threshold between two points in order to be considered as a pair
        :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                                transformation to be considered converged
        :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                                to be considered converged
        :param point_pairs_threshold: the minimum number of point pairs the should exist
        :param verbose: whether to print informative messages about the process (default: False)
        :return: the transformation history as a list of numpy arrays containing the rotation (R) and translation (T)
                transformation in each iteration in the format [R | T] and the aligned points as a numpy array M x 2
        """
        max_rotation_flag = False
        all_points = []
        transformation_history = []
        iteration_array = []
        translation_x_mm = []
        translation_y_mm = []
        distance_between_points = []
        total_rotation_degrees = 0
        total_trans_x = 0
        total_trans_y = 0
        rotation_degrees = []
        final_rotation = np.array([[ 1, 0], [ 0,  1]])
        source_points_orig = np.copy(points)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)
        
        for iter_num in range(max_iterations):
            if verbose:
                print('------ iteration', iter_num, '------')

            closest_point_pairs = []  # list of point correspondences for closest point rule

            distances, indices = nbrs.kneighbors(points)
            for nn_index in range(len(distances)):
                if distances[nn_index][0] < distance_threshold:
                    closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

            # if only few point pairs, stop process
            if verbose:
                print('number of pairs found:', len(closest_point_pairs))
            if len(closest_point_pairs) < point_pairs_threshold:
                if verbose:
                    print('No better solution can be found (very few point pairs)!')
                break

            # compute translation and rotation using point correspondences
            closest_rot_angle, closest_translation_x, closest_translation_y = self.point_based_matching(closest_point_pairs)
            if closest_rot_angle is not None:
                if verbose:
                    print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                    print('Translation:', closest_translation_x, closest_translation_y)
                        
            if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
                print("No better solution can be found!")
                if verbose:
                    print('No better solution can be found!')
                break
            
            # transform 'points' (using the calculated rotation and translation)
            c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
            rot = np.array([[c, -s],
                            [s, c]])
            aligned_points = np.dot(points, rot.T)
            aligned_points[:, 0] += closest_translation_x 
            aligned_points[:, 1] += closest_translation_y

            # update 'points' for the next iteration
            points = aligned_points
            all_points.append(np.copy(points))
            # update transformation history
            transformation_history.append(np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))
            
            final_rotation = np.dot(final_rotation, rot)
            total_trans_x += closest_translation_x
            total_trans_y += closest_translation_y
            total_rotation_degrees += np.degrees(closest_rot_angle)

            # Compute distance between points
            distance_between_points.append(np.mean(distances))

            iteration_array.append(iter_num)

            # Compute the rotation angle in radians.
            rotation_radians = np.arctan2(rot[1, 0], rot[0, 0])

            # Convert rotation angle to degrees.
            rotation_angle_degrees = np.degrees(rotation_radians)
            rotation_degrees.append(rotation_angle_degrees)

            # Calculate translation in millimeters.
            translation_x_mm.append(closest_translation_x)
            translation_y_mm.append(closest_translation_y)
            
            # check convergence
            if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                    and (abs(closest_translation_x) < convergence_translation_threshold) \
                    and (abs(closest_translation_y) < convergence_translation_threshold):
                
                print('Converged!')

                if False:
                    # Plotting outside the loop
                    for i, points in enumerate(all_points):
                        plt.plot(reference_points[:,0], reference_points[:,1], color="green")
                        plt.plot(points[:, 0], points[:, 1], color="blue")
                        plt.title(f'Iteration: {i + 1}')
                        plt.xlabel('X-axis')
                        plt.ylabel('Y-axis')
                        plt.show(block=False)
                        plt.pause(1)
                        # Clear the current figure to remove the previous points
                        plt.clf()
                    plt.show()
                if verbose:
                    print('Converged!')
                
                # Stop the ICP process
                break
        
        return total_rotation_degrees
################################################################################################
    
    def get_rotation_matrix_overlap(self, oct_lumen_contours, z_distance, crop_top, crop_bottom, conversion_factor):
        height = (1024 - crop_top - crop_bottom)
        width = 1024
        center_x = width/2
        center_y = width/2
        previous_contour = None
        rotations = []
        total_rotations = []
        rotation_total = 0
        rotation = 0
        z_previous = 0
        prnitng_overlaps = []
        prnitng_angle = []

        for current_contour in oct_lumen_contours:
            max_overlap = 0
            z_current = current_contour[0][2]
            while True:
                z_diff = z_current - z_previous
                current_contour = [(x[0], x[1]) for x in current_contour]
                
                if (z_diff) < (z_distance + 0.01):
                    if previous_contour is not None:
                        # Perform overlap measurements.
                        for angle in range(-30, 31):
                            overlap = self.calculate_overlap(current_contour, previous_contour, angle/10, center_x, center_y, height, width, conversion_factor)
                            prnitng_overlaps.append(overlap)
                            prnitng_angle.append(angle/10)
                            if overlap > max_overlap:
                                max_overlap = overlap
                                rotation = angle/10
                        overlap = self.calculate_overlap(current_contour, previous_contour, rotation, center_x, center_y, height, width, conversion_factor, True)
                        print(f'rotation = {rotation}')
                        previous_contour = current_contour
                        rotation_total += rotation
                        rotations.append(rotation)
                        total_rotations.append(rotation_total)
                        z_previous = z_current
                        break
                    else:
                        previous_contour = current_contour
                        rotation = 0
                        rotation_total += rotation
                        rotations.append(rotation)
                        total_rotations.append(rotation_total)
                        break
                else:
                    z_previous += z_distance
                    rotation = 0
                    rotation_total += rotation
                    rotations.append(rotation)
                    total_rotations.append(rotation_total)
            if False:
                plt.plot(prnitng_angle, prnitng_overlaps ,label="Overlap of contours")
                plt.plot(rotation, max_overlap , "x", label="Max overlap")

                plt.xlabel("Rotation angle [°]")  # Add an x-axis label
                plt.ylabel("Overlapping area [pixels]")  # Add a y-axis label
                plt.legend()  # Display the legend
                plt.show()
        return np.array(total_rotations)
        

    def calculate_overlap(self, current_contour, previous_contour, rotation_angle, center_x, center_y, height, width, conversion_factor, plot=False):
        # Rotate the current contour around the fixed center point (512, 512)
        rotated_contour = self.rotate_contour(current_contour, center_x, center_y, rotation_angle, conversion_factor)
        previous_contour = [(int(x/conversion_factor), int(y/conversion_factor)) for x, y in previous_contour]

        if False:
            x = []
            y = []
            for point in rotated_contour:
                x.append(point[0])
                y.append(point[1])

            plt.plot(x, y)
            x = []
            y = []
            for point in previous_contour:
                x.append(point[0])
                y.append(point[1])

            plt.plot(x, y)
            plt.plot(center_x, center_y, "x")
            plt.show()

        # Create blank images to store the intersection
        image_current = np.zeros((1024, 1024), dtype=np.uint8)
        image_previous = np.zeros((1024, 1024), dtype=np.uint8)
        
        
        rotated_contour = np.array(rotated_contour, dtype=np.int32)
        previous_contour = np.array(previous_contour, dtype=np.int32)

        # Draw the contours on the images
        cv2.drawContours(image_current, [rotated_contour], 0, 255, -1)
        cv2.drawContours(image_previous, [previous_contour], 0, 255, -1)

         # Find the intersection (common area)
        intersection = cv2.bitwise_and(image_current, image_previous)

        # Calculate the area of the intersection
        white_pixel_count = cv2.countNonZero(intersection)
        return white_pixel_count

    def rotate_contour(self, contour, center_x, center_y, angle_degrees, conversion_factor):
        # Convert angle to radians
        angle_radians = np.deg2rad(angle_degrees)

        # Create a rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1)

        # Apply rotation to each point in the contour
        rotated_contour = []
        for point in contour:
            rotated_point = np.dot(rotation_matrix, [point[0]/conversion_factor, point[1]/conversion_factor, 1])
            rotated_contour.append((rotated_point[0], rotated_point[1]))

        return [(int(x), int(y)) for x, y in rotated_contour]
###############################################################################################
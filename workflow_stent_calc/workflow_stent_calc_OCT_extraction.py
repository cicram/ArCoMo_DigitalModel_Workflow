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

class oct_extraction:
    def __init__(self):
        self.oct_registration_point_x = 0
        self.oct_registration_point_y = 0
        self.retPt = []

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


    def close_binary_mask(self, binary_mask):
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

        return closed_mask


    def spline_fitting(self, binary_mask, display_images, save_images_for_controll, page):
        # Apply morphological operations to close gaps in the binary mask.
        kernel = np.ones((30, 30), np.uint8)
        closed_mask_ = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        closed_mask = self.close_binary_mask(closed_mask_)
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

        return fitted_contour


    def plt_cv2_images(self, title, image):
            cv2.imshow(title, image)
            cv2.waitKey(0)  # Wait for a key press to move to the next image
            cv2.destroyAllWindows()  # Close the image window


    def frames_alignment(self, contours, rotation_matrix, z_offset):
        rotated_contours = []

        # Case for rotation point, would not work with other code, due to for condition
        if len(contours) == 3:
                current_contour = np.array(contours)
                idx = int(current_contour[2]/z_offset)
                rot_angle = rotation_matrix[idx]  # Convert the list to a numpy array
                if current_contour is not None:
                    c, s = math.cos(np.radians(rot_angle)), math.sin(np.radians(rot_angle))
                    
                    # 2D rotation matrix for x-y plane
                    rot_xy = np.array([[c, -s],
                                    [s, c]])
                    
                    # Apply rotation only to x-y coordinates
                    current_contour_xy_rotated = np.dot(current_contour[0:2], rot_xy.T)
                    
                    # Combine rotated x-y coordinates with original z-values
                    current_contour_rotated = [current_contour_xy_rotated[0], current_contour_xy_rotated[1], current_contour[2]]

                    
                    rotated_contours.append(current_contour_rotated)
        
        #For Normal rotation
        else:
            for current_contour in contours:
                current_contour = np.array(current_contour)
                if len(current_contour) == 3:
                    idx = int(current_contour[2]*z_offset)

                    rot_angle = rotation_matrix[idx]  # Convert the list to a numpy array
                    if current_contour is not None:
                        c, s = math.cos(np.radians(rot_angle)), math.sin(np.radians(rot_angle))
                        
                        # 2D rotation matrix for x-y plane
                        rot_xy = np.array([[c, -s],
                                        [s, c]])
                        
                                            # Apply rotation only to x-y coordinates
                    current_contour_xy_rotated = np.dot(current_contour[0:2], rot_xy.T)
                    
                    # Combine rotated x-y coordinates with original z-values
                    current_contour_rotated = [current_contour_xy_rotated[0], current_contour_xy_rotated[1], current_contour[2]]
                        
                    rotated_contours.append(current_contour_rotated)
                
                else:
                    idx = int(current_contour[0][2]*z_offset)

                    rot_angle = rotation_matrix[idx]  # Convert the list to a numpy array
                    if current_contour is not None:
                        c, s = math.cos(np.radians(rot_angle)), math.sin(np.radians(rot_angle))
                        
                        # 2D rotation matrix for x-y plane
                        rot_xy = np.array([[c, -s],
                                        [s, c]])
                        
                        # Apply rotation only to x-y coordinates
                        current_contour_xy_rotated = np.dot(current_contour[:, :2], rot_xy.T)
                        
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


    def segment_calc(self, input_file, OCT_start_frame, OCT_end_frame, crop_top, z_distance, conversion_factor):
        saved_paths = []
        saved_splines = []
        with Image.open(input_file) as im:
            for page in range(OCT_start_frame, 8, 1):
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

                        elif key == ord("s") or key == ord("S"):
                            # Save the path if the "S" key is pressed
                            if len(path_total) > 1:
                                # Fit a spline to the path and save the 500 points
                                fitted_spline = self.fit_spline_to_path(path_total, (page+OCT_start_frame)*z_distance, conversion_factor)
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

    def get_calc_contours(self, path_segmented_calc, input_file, OCT_start_frame, OCT_end_frame, z_space, conversion_factor, crop_top):
        # Check if there is already a calc file, if yes, ask the user if he wants to use this or if he wants to do a new segmentation

        if os.path.exists(path_segmented_calc):
            # Ask the user if they want to use the existing file or perform a new segmentation
            result = messagebox.askquestion("File Exists", "A segmentation for this case already exists. Do you want to use this file for segmentation?",
                                            icon='warning')

            if result == 'yes':
                # Parse calc
                results = []
                with open(path_segmented_calc, 'r') as file:
                    for line in file:
                        # Parse each line and convert it back to the original data structure
                        path = eval(line.strip())
                        results.append(path)
                calc_contours = results
            
            else:
                # Segment new calc
                calc_contours = self.segment_calc(input_file, OCT_start_frame, OCT_end_frame, crop_top, z_space, conversion_factor)
                # save the file
                with open(path_segmented_calc, 'w') as file:
                            for path in calc_contours:
                                # Write each path to the file
                                file.write(str(path) + '\n')
        
        else:
            # File doesn't exist, perform a new segmentation
            calc_contours = self.segment_calc(input_file, OCT_start_frame, OCT_end_frame, crop_top, z_space, conversion_factor)
            # save the file
            with open(path_segmented_calc, 'w') as file:
                for path in calc_contours:
                    # Write each path to the file
                    file.write(str(path) + '\n')

        return calc_contours
    

    def get_stent_contours(self, input_file, OCT_start_frame, OCT_end_frame, z_space, conversion_factor):
        color = [255, 255, 255]
        crop = 200
        crop2 = 900
        points = []
        with Image.open(input_file) as im:
            for page in range(OCT_start_frame, OCT_end_frame, 1):
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                open_cv_image = image[crop:crop2, :, ::-1].copy()  # Crop image at xxx pixels from top

                # Apply the color filter
                binary_mask = self.filter_color_one(open_cv_image, color)
                
            # Find contours in the binary mask
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                        points.append([centroid_x*conversion_factor, centroid_y*conversion_factor, page*z_space])

        if False:
            # Restrcutre data accoring to z-value hight
            grouped_data = {}
            
            for item in points:
                key = round(item[2], 4)  # Rounding to handle potential floating-point precision issues
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(tuple(item))
            
            result = [grouped_data[key] for key in sorted(grouped_data)]
        
        return points
    

    def get_lumen_contour(self, crop, input_file, OCT_end_frame, OCT_start_frame, OCT_registration_frame, z_offset, conversion_factor, save_file, color1, color2, smoothing_kernel_size,
                        threshold_value, display_images, save_images_for_controll):
        z_coordinate = 0
        point_cloud = []
        with Image.open(input_file) as im:
            for page in range(OCT_start_frame, OCT_end_frame, 1):
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                image_flipped = np.flipud(image)
                height, width, channels = image_flipped.shape
                open_cv_image = image_flipped[0: height-crop, :, ::-1].copy()  # Crop image at xxx pixels from top

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
                
                if current_contour is None:
                    print("Contour is none")

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


    def get_registration_point(self, input_file, crop, OCT_start_frame, OCT_registration_frame, display_images, z_offset, save_file, conversion_factor):
        with Image.open(input_file) as im:
            im.seek(OCT_registration_frame)  # Move to marked page (frame)
            image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
            image_flipped = np.flipud(image)
            height, width, channels = image_flipped.shape
            open_cv_image = image_flipped[0: height-crop, :, ::-1].copy()

            # Display the cropped image
            cv2.imshow('Select registration point (left mouse click)', open_cv_image)            

            # Set the mouse callback function
            cv2.setMouseCallback('Select registration point (left mouse click)', self.mouse_callback)
            # Wait for the user to click on the image
            cv2.waitKey()

        # Convert into mm unit
        registration_point = [self.oct_registration_point_x, self.oct_registration_point_y, (OCT_registration_frame-OCT_start_frame)*z_offset]
        return registration_point
    

    def get_rotation_matrix(self, input_file):
        rotation_angles = []
        rotation_total = 0
        previous_image = None
        with Image.open(input_file) as im:
            for page in range(1, 280, 1):
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

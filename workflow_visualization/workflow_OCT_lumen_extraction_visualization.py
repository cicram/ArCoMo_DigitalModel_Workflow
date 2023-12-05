from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import rotate
from PIL import Image

class oct_lumen_extraction:

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


    def euclidean_distance(self, point1, point2):
        """
        Euclidean distance between two points.
        :param point1: the first point as a tuple (a_1, a_2, ..., a_n)
        :param point2: the second point as a tuple (b_1, b_2, ..., b_n)
        :return: the Euclidean distance
        """
        a = np.array(point1)
        b = np.array(point2)

        return np.linalg.norm(a - b, ord=2)


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


    def icp_alignment(self, first_contour, points, reference_points, transformation_matrix_previous, display_images, registration_point, image, page, gray_image, z_coordinate, max_iterations=100, distance_threshold=100, convergence_translation_threshold=1e-3,
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
        # Rough initial alignment
        if transformation_matrix_previous is not None and False:
            # Pad the 3D point cloud with a column of 1's.
            points = np.array(points)
            #plt.plot(points[:, 0], points[:, 1], "x")
            points = np.hstack((points, np.ones((points.shape[0], 1))))

            # Perform matrix multiplication to transform the point cloud.
            points = np.dot(points, transformation_matrix_previous.T)

            # Remove the padding column and keep the transformed 3D points.
            points = points[:, :3]
            #plt.plot(points[:, 0], points[:, 1], 'o')
            #plt.show()

        registration_point_ = registration_point[0:2]
        transformation_history = []
        iteration_array = []
        translation_x_mm = []
        translation_y_mm = []
        distance_between_points = []
        height, width = gray_image.shape
        scaling = 98
        total_rotation_degrees = 0
        total_trans_x = 0
        total_trans_y = 0
        rotation_degrees = []
        final_rotation = np.array([[ 1, 0], [ 0,  1]])
        source_points_orig = np.copy(points)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)
        image_orig = np.copy(image)
        my_points__ = []
        for i in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                single_point = np.array([j/scaling, i/scaling])
                my_points__.append(single_point)
        my_points_ = np.array(my_points__)
        print("page: " + str(page) + " icp algo applied")

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
    
            updated_registration_point = np.dot(registration_point_, rot.T)
            updated_registration_point[0] += closest_translation_x
            updated_registration_point[1] += closest_translation_y

            my_points = np.dot(my_points_, rot.T)
            my_points[:, 0] += closest_translation_x/scaling
            my_points[:, 1] += closest_translation_y/scaling

            if False:
                # Image transposition
                # Rotate the image using scipy.ndimage.rotate
                rotated_image = rotate(image, np.degrees(closest_rot_angle), reshape=True)
                # Translate the image using NumPy array operations
                #translated_image = np.roll(rotated_image, (closest_translation_x, closest_translation_y), axis=(0, 1))
                
                #Image.fromarray(translated_image).show(title='Translated Image')
                image = rotated_image
                                 # Display the original and rotated image
                Image.fromarray(image_orig).show(title='Original Image')
                Image.fromarray(rotated_image).show(title='Rotated Image')

            # update 'points' for the next iteration
            points = aligned_points
            registration_point_ = updated_registration_point
            my_points_ = my_points
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

                if verbose:
                    print('Converged!')
                if display_images:
                    plt.figure(figsize=(12, 4))
                    plt.subplot(141)
                    plt.plot(iteration_array, distance_between_points)
                    #plt.plot(iteration_array[break_idx], distance_between_points[break_idx], "x")
                    plt.xlabel("Iteration")
                    plt.ylabel("Distance")
                    plt.title("Distance vs. Iteration")

                    plt.subplot(142)
                    plt.plot(iteration_array, rotation_degrees)
                    #plt.plot(iteration_array[break_idx], rotation_degrees[break_idx], "x")
                    plt.xlabel("Iteration")
                    plt.ylabel("Rotation (degrees)")
                    plt.title("Rotation vs. Iteration")

                    plt.subplot(143)
                    plt.plot(iteration_array, translation_x_mm)
                    #plt.plot(iteration_array[break_idx], translation_mm[break_idx], "x")
                    plt.xlabel("Iteration")
                    plt.ylabel("Translation x (mm)")
                    plt.title("Translation x vs. Iteration")

                    plt.subplot(144)
                    plt.plot(iteration_array, translation_y_mm)
                    #plt.plot(iteration_array[break_idx], translation_mm[break_idx], "x")
                    plt.xlabel("Iteration")
                    plt.ylabel("Translation x (mm)")
                    plt.title("Translation x vs. Iteration")

                    plt.tight_layout()
                    plt.show()
                # Stop the ICP process
                break
            
        transformation_matrix = self.compute_transformation_matrix(source_points_orig, points)
        with open("workflow_processed_data_output/image_translations/alignement_translations.txt", 'a') as file:
            file.write(f"{page} {total_trans_x:.2f} {total_trans_y:.2f} {total_rotation_degrees:.2f}\n")

        rotation_matrix_alignement = np.array([[np.cos(total_rotation_degrees), -np.sin(total_rotation_degrees), 0.0],
                [np.sin(total_rotation_degrees), np.cos(total_rotation_degrees), 0.0],
                [0.0, 0.0, 1.0]])
        translation_vector_alignment = np.array([total_trans_x/scaling, total_trans_y/scaling, 0.0])
        my_points_return = []
        for point in my_points:
            my_points_return.append(np.array([point[0], point[1], page]))
        
        return points, transformation_matrix, final_rotation, total_trans_x, total_trans_y, updated_registration_point, my_points_return


    def find_registration_frame(self, letter_x_mask_path, input_file, crop, color1, color2, display_images):
        
        letter_mask = cv2.imread(letter_x_mask_path, cv2.IMREAD_GRAYSCALE)

        # Find registration frame
        with Image.open(input_file) as im:
            for page in range(im.n_frames):
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array

                open_cv_image = image[crop:, :, ::-1].copy()  # Crop image at xxx pixels from top

                # Apply the color filter
                binary_mask = self.filter_color(open_cv_image, color1, color2)
                if display_images and False:
                    self.plt_cv2_images('Color filtered image', binary_mask)
                result = cv2.matchTemplate(binary_mask, letter_mask, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val > 0.9:
                    print(page)
                    return page
                    
        return None


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

        if display_images:
            # Plot the original binary mask, the closed mask, and the fitted spline.
            plt.figure(figsize=(12, 4))
            plt.subplot(141)
            plt.imshow(binary_mask, cmap='gray')
            plt.title('Original Binary Mask')
            plt.subplot(142)
            plt.imshow(closed_mask_, cmap='gray')
            plt.title('Morphology Binary Mask')
            plt.subplot(143)
            plt.imshow(closed_mask, cmap='gray')
            plt.title('Convexhull Binary Mask')
            plt.subplot(144)
            plt.plot(x1, y1, 'r')
            plt.plot(x2, y2, 'b')
            plt.title('Fitted Spline')
            plt.gca().invert_yaxis()  # Invert the y-axis to match typical image coordinates.
            plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio.
            plt.show()
        
        if save_images_for_controll:
            plt.figure(figsize=(12, 4))
            plt.subplot(141)
            plt.imshow(binary_mask, cmap='gray')
            plt.title('Original Binary Mask')
            plt.subplot(142)
            plt.imshow(closed_mask_, cmap='gray')
            plt.title('Morphology Binary Mask')
            plt.subplot(143)
            plt.imshow(closed_mask, cmap='gray')
            plt.title('Convexhull Binary Mask')
            plt.subplot(144)
            plt.plot(x1, y1, 'r')
            plt.plot(x2, y2, 'b')
            plt.title('Fitted Spline')
            plt.gca().invert_yaxis()  # Invert the y-axis to match typical image coordinates.
            plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio.
            # Specify the file path and name for the saved PNG image
            output_directory = "controll_contour_images"
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Create the filename including the "page" variable and save path
            output_file_path = os.path.join(output_directory, f'output_image_page_{page}.png')

            # Save the plot as a PNG image
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()


        return fitted_contour


    def plt_cv2_images(self, title, image):
            cv2.imshow(title, image)
            cv2.waitKey(0)  # Wait for a key press to move to the next image
            cv2.destroyAllWindows()  # Close the image window


    def process_tif_file(self, crop, input_file, OCT_end_frame, OCT_start_frame, z_offset, conversion_factor, save_file, color1, color2, smoothing_kernel_size,
                        threshold_value, display_images, registration_point, carina_point_frame, save_images_for_controll):
        first_contour_flag = True
        z_coordinate = 0
        previous_contour = None
        point_cloud = []
        transformation_matrix_previous = None
        my_aligned_images = []
        with Image.open(input_file) as im:
            for page in range(OCT_start_frame, OCT_end_frame, 1):
                print(page)
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                image_croped = image[crop:, :, ::-1].copy()
                gray_image = cv2.cvtColor(image_croped, cv2.COLOR_BGR2GRAY)

                # Plot the grayscale image
                if False:
                    plt.imshow(gray_image, cmap='gray')
                    plt.title('Grayscale Image')
                    plt.show()
                open_cv_image = image[crop:, :, ::-1].copy()  # Crop image at xxx pixels from top

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

                height, width = gray_image.shape
                scaling = 98
                my_points = []
                for i in range(gray_image.shape[0]):
                    for j in range(gray_image.shape[1]):
                        single_point = np.array([j/scaling, i/scaling, page])
                        my_points.append(single_point)
                aligned_image_points = my_points

                # fit a spline to the contour
                current_contour = self.spline_fitting(threshold_mask, display_images, save_images_for_controll, page)
                if page == 89:
                    current_contour = None
                if current_contour is not None:

                    # save unaligned pointcloud
                    if save_file:
                        # Calculate the centroid of the aligned points.
                        centroid_unaligned = np.mean(current_contour, axis=0)

                        # Shift the points to have the centroid at (0, 0).
                        unaligned_source_points_shifted = current_contour - centroid_unaligned

                        # Create a 3D point cloud with incremental z-coordinate and convert into mm unit.
                        unaligned_current_contour = [(xi * conversion_factor, yi * conversion_factor, z_coordinate) for xi, yi in
                                            unaligned_source_points_shifted]

                        # Write the coordinates to the text file
                        with open("workflow_processed_data_output/unaligned_OCT_frames.txt", 'a') as file:
                            for coord in unaligned_current_contour:
                                file.write(f"{coord[0]:.2f} {coord[1]:.2f} {coord[2]:.2f}\n")
                        
                    if previous_contour is not None:
                        # Perform ICP alignment.
                        aligned_source_spline, transformation_matrix, final_rotation, total_trans_x, total_trans_y, updated_registration_point, aligned_image_points = self.icp_alignment(first_contour,
                                                                                                                                                                     current_contour, previous_contour, transformation_matrix_previous, display_images, registration_point, image, page, gray_image, z_coordinate)
                        # Convert the spline points to NumPy arrays.
                        source_points = np.array(current_contour)
                        aligned_source_points = np.array(aligned_source_spline)
                        transformation_matrix_previous = transformation_matrix
                        # Apply transformation also to registration point
                        if page == (carina_point_frame):
                            if False: # old way
                                registration_point_2d = registration_point[0:2]
                                registration_point_transformed = np.dot(registration_point_2d, transformation_matrix)
                                centroid = np.mean(aligned_source_points, axis=0)
                                registration_point_transformed[0] = registration_point_transformed[0] - centroid[0] * conversion_factor
                                registration_point_transformed[1] = registration_point_transformed[1] - centroid[1] * conversion_factor

                                registration_point_transformed = np.copy(registration_point[0:2])
                                rotated_points = np.dot(registration_point_transformed, final_rotation.T)
                                transposed_x = rotated_points[0] + total_trans_x * conversion_factor
                                transposed_y = rotated_points[1] + total_trans_y * conversion_factor         
                                centroid = np.mean(aligned_source_points, axis=0)
                                registration_point_transformed[0] = transposed_x - centroid[0] * conversion_factor
                                registration_point_transformed[1] = transposed_y - centroid[1] * conversion_factor

                            centroid = np.mean(aligned_source_points, axis=0)
                            updated_registration_point[0] -= centroid[0]
                            updated_registration_point[1] -= centroid[1]
                            if save_file:
                                # Write the coordinates to the text file
                                with open("workflow_processed_data_output/aligned_OCT_registration_point.txt", 'w') as file:
                                    file.write(f"{updated_registration_point[0] * conversion_factor:.2f} {updated_registration_point[1] * conversion_factor:.2f} {registration_point[2]:.2f}\n")

                        if display_images:
                            # Plot the original and aligned splines.
                            plt.figure(figsize=(10, 6))
                            plt.plot([point[0] for point in source_points], [point[1] for point in source_points], 'b-',
                                    label='Original Source Spline')
                            plt.plot([point[0] for point in previous_contour], [point[1] for point in previous_contour], 'g-',
                                    label='Target Spline')
                            plt.plot([point[0] for point in aligned_source_points], [point[1] for point in aligned_source_points],
                                    'r-', label='Transformed Source Spline')
                            plt.legend()
                            plt.gca().invert_yaxis()  # Invert the y-axis to match typical image coordinates.
                            plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio.
                            plt.title('Original and Transformed Splines')
                            plt.xlabel('X-coordinate')
                            plt.ylabel('Y-coordinate')
                            plt.show()

                    else:
                        print("first contour")
                        aligned_source_points = np.array(current_contour)
                        first_contour = np.array(current_contour)
                    # Calculate the centroid of the aligned points.
                    centroid = np.mean(aligned_source_points, axis=0)

                    # Shift the points to have the centroid at (0, 0).
                    aligned_source_points_shifted = aligned_source_points - centroid
                    aligned_image_points_shifted = np.copy(np.array(aligned_image_points))
                    aligned_image_points_shifted[:, 0] -= centroid[0]/scaling
                    aligned_image_points_shifted[:, 1] -= centroid[1]/scaling

                    # Create a 3D point cloud with incremental z-coordinate and convert into mm unit.
                    point_cloud_current = [(xi * conversion_factor, yi * conversion_factor, z_coordinate) for xi, yi in
                                        aligned_source_points_shifted]
                    
                    if page == (carina_point_frame):
                        current_contour = np.array(current_contour)
                        plt.plot(updated_registration_point[0], updated_registration_point[1], "x")
                        plt.plot(registration_point[0]- centroid[0] , registration_point[1]- centroid[1], "o")
                        point_cloud_current = np.array(point_cloud_current)
                        aligned_source_points = np.array(aligned_source_points)
                        plt.plot(aligned_source_points[:, 0] - centroid[0], aligned_source_points[:, 1] - centroid[1], "x")
                        plt.plot(current_contour[:, 0] - centroid[0], current_contour[:, 1] - centroid[1], "x")
                        #plt.show()
                    if save_file:
                        # Write the coordinates to the text file
                        with open("workflow_processed_data_output/output_point_cloud.txt", 'a') as file:
                            for coord in point_cloud_current:
                                file.write(f"{coord[0]:.2f} {coord[1]:.2f} {coord[2]:.2f}\n")
                                                        
                    with open("workflow_processed_data_output/image_translations/center_point_shift.txt", 'a') as file:
                            file.write(f"{centroid[0]:.2f} {centroid[1]:.2f}\n")

                    point_cloud.append(point_cloud_current)
                    if first_contour_flag or True:
                        previous_contour = aligned_source_points
                        #first_contour_flag = False
                    
                    my_aligned_images.append(aligned_image_points_shifted)

                    ##############################################################
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # Show the plot
                    x_filtered = []
                    y_filtered = []
                    z_filtered = []
                    for frame_points in aligned_image_points_shifted:
                        x_filtered.append(frame_points[0])
                        y_filtered.append(frame_points[1])
                        z_filtered.append(z_coordinate)
                    ax.scatter(x_filtered[::120], y_filtered[::120], z_filtered[::120], c="blue", marker='o')
                    x_filtered = []
                    y_filtered = []
                    z_filtered = []
                    for frame_points in aligned_source_points_shifted:
                            x_filtered.append(frame_points[0]/scaling)
                            y_filtered.append(frame_points[1]/scaling)
                            z_filtered.append(z_coordinate)
                    ax.scatter(x_filtered[::30], y_filtered[::30], z_filtered[::30], c="red", marker='o')
                    ax.set_xlabel('Px')
                    ax.set_ylabel('Py')
                    ax.set_zlabel('Pz')
                    #plt.show()

                    ##############################################################
                z_coordinate += z_offset  # Increment z-coordinate

        x_filtered = []
        y_filtered = []
        z_filtered = []

        for point in point_cloud:
            x , y, z = zip(*point)
            x_filtered.append(x[::20])
            y_filtered.append(y[::20])
            z_filtered.append(z[::20])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_filtered, y_filtered, z_filtered, c="red", marker='o')
        ax.set_xlabel('Px')
        ax.set_ylabel('Py')
        ax.set_zlabel('Pz')
        #plt.show()

        return my_aligned_images

    def find_center_of_letter_X(self, binary_mask):
        # Load the binary mask and the letter 'x' mask
        letter_mask = cv2.imread("workflow_utils/image_X.jpg", cv2.IMREAD_GRAYSCALE)

        # Find the position of 'x' in the binary mask
        result = cv2.matchTemplate(binary_mask, letter_mask, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Determine the position of the letter 'x' and its size
        letter_width, letter_height = letter_mask.shape[::-1]
        top_left = max_loc
        bottom_right = (top_left[0] + letter_width, top_left[1] + letter_height)

        # Create a mask with everything except the letter 'x' set to black
        binary_mask_retained = binary_mask.copy()
        binary_mask_retained[:,:] = 0  # Set the entire mask to black
        binary_mask_retained[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255  # Set the 'x' region to white

        # Calculate the center coordinates of the letter 'x'
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = ((top_left[1] + bottom_right[1]) // 2)

        return binary_mask_retained, (center_x, center_y)



    def get_registration_point(self, color1, color2, input_file, crop, carina_point_frame, display_images, z_offset, save_file, conversion_factor):
        with Image.open(input_file) as im:
            im.seek(carina_point_frame)  # Move to marked page (frame)
            image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array

            open_cv_image = image[crop:, :, ::-1].copy()  # Crop image at xxx pixels from top

            # Apply the color filter
            binary_mask = self.filter_color(open_cv_image, color1, color2)
            if display_images:
                self.plt_cv2_images('Color filtered image', binary_mask)
            
            # Find the coordinates of the letter X
            binary_mask_x, position = self.find_center_of_letter_X(binary_mask, )
            if display_images:
                self.plt_cv2_images('Color filtered image', binary_mask_x)
        
        # Convert into mm unit
        registration_point = [position[0], position[1], carina_point_frame*z_offset]
        print(registration_point)
        return np.array(registration_point)

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os


# Function to filter by RGB color
def filter_color(input_image, color1, color2):
    # Create masks for the two colors
    lower_color1 = np.array([color1[2], color1[1], color1[0]], dtype=np.uint8)
    upper_color1 = np.array([color1[2], color1[1], color1[0]], dtype=np.uint8)
    # Little offset given to ensure color of line-dots are included
    lower_color2 = np.array([color2[2] - 10, color2[1] - 10, color2[0] - 10], dtype=np.uint8)
    upper_color2 = np.array([color2[2] + 10, color2[1] + 10, color2[0] + 10], dtype=np.uint8)

    mask1 = cv2.inRange(input_image, lower_color1, upper_color1)
    mask2 = cv2.inRange(input_image, lower_color2, upper_color2)

    # Combine the masks to create a binary mask
    binary_mask = cv2.bitwise_or(mask1, mask2)

    return binary_mask


# Function to smooth the image and create an intensity threshold mask (thin line removal)
def smooth_and_threshold(image, smoothing_kernel_size, threshold_value):
    # Smooth the image using Gaussian blur
    blurred = cv2.GaussianBlur(image, (smoothing_kernel_size, smoothing_kernel_size), 0)

    # Create an intensity threshold mask
    _, mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    return mask


def compute_transformation_matrix(source_points_orig, source_points):

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


def icp_alignment(source_spline, target_spline, transformation_matrix_previous, display_images, max_iterations=100, rotation_limit=2):
    """
    Perform Iterative Closest Point (ICP) alignment between source_spline and target_spline.

    Args:
        source_spline (list of tuples): Points of the source spline.
        target_spline (list of tuples): Points of the target spline.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        aligned_source_spline (list of tuples): Aligned points of the source spline.
        transformation_matrix (numpy.ndarray): 2x3 transformation matrix.
    """
    # Convert spline points to NumPy arrays for easier manipulation.
    source_points = np.array(source_spline)
    source_points_orig = np.array(source_spline)
    target_points = np.array(target_spline)

    distance_between_points = []
    rotation_degrees = []
    translation_mm = []
    iteration_array = []

    # Stoping criteria
    rotation_tolerance = 0.5
    translation_tolerance = 0.05

    # Initialize variables to keep track of distances for the last few iterations.
    prev_rotation_degrees = [np.inf] * 3
    prev_translation_mm = [np.inf] * 3 
    break_flag = False
    total_rotation = 0
    total_translation = 0
    total_translation_x = 0
    total_translation_y = 0

    # Rough initial alignment
    if transformation_matrix_previous is not None:
        # Pad the 3D point cloud with a column of 1's.
        plt.plot(source_points[:, 0], source_points[:, 1], 'x')
        source_points = np.hstack((source_points, np.ones((source_points.shape[0], 1))))

        # Perform matrix multiplication to transform the point cloud.
        source_points = np.dot(source_points, transformation_matrix_previous.T)

        # Remove the padding column and keep the transformed 3D points.
        source_points = source_points[:, :3]
        plt.plot(source_points[:, 0], source_points[:, 1], 'o')

    for iteration in range(max_iterations):

        # Find the nearest neighbors between source and target points.
        distance_matrix = np.linalg.norm(source_points[:, np.newaxis, :] - target_points, axis=2)
        closest_indices = np.argmin(distance_matrix, axis=1)

        # Update the source points by matching them to the closest target points.
        matched_points = target_points[closest_indices]

        # Calculate the transformation matrix using the Procrustes analysis method.
        H = np.dot(source_points.T, matched_points)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(U, Vt)
        T = np.mean(matched_points, axis=0) - np.mean(np.dot(source_points, R.T), axis=0)

        # Apply the transformation to the source points.
        source_points = np.dot(source_points, R.T) + T

        # Compute distance
        distance_x = np.sum(np.square(matched_points[0] - source_points[0]))
        distance_y = np.sum(np.square(matched_points[1] - source_points[1]))

        distance_between_points.append((distance_x + distance_y) / 2)
        iteration_array.append(iteration)

        # Compute the rotation angle in radians.
        rotation_radians = np.arctan2(R[1, 0], R[0, 0])

        # Convert rotation angle to degrees.
        rotation_angle_degrees = np.degrees(rotation_radians)
        rotation_degrees.append(rotation_angle_degrees)

        # Calculate translation in millimeters.
        translation_millimeters = np.linalg.norm(T)
        translation_mm.append(translation_millimeters)

        # Add to totoal rotation
        total_rotation += rotation_angle_degrees
        total_translation_x += T[0]
        total_translation_y += T[1]
        if iteration == 99:
            print("Iterations: " + str(max(iteration_array)) + "Total rotation: " + str(total_rotation) + "Total translation x: " + str(total_translation_x) + "Total translation y: " + str(total_translation_y))
        # Check for convergence based on the change in rotation and translation values.
        if all(abs(rotation_angle_degrees - prev_rotation) <= rotation_tolerance for prev_rotation in prev_rotation_degrees) and \
            all(abs(translation_millimeters - prev_translation) <= translation_tolerance for prev_translation in prev_translation_mm):
            break
            
        
        prev_rotation_degrees.pop(0)
        prev_rotation_degrees.append(rotation_angle_degrees)
        prev_translation_mm.pop(0)
        prev_translation_mm.append(translation_millimeters)

        if False:
            if (abs(rotation_angle_degrees - prev_rotation_degrees) <= 0.2) and \
            (abs(translation_millimeters - prev_translation_mm) <= 0.05) and not break_flag:
                break_idx = iteration
                break_flag = True
                #break

            prev_rotation_degrees = rotation_angle_degrees
            prev_translation_mm = translation_millimeters

    if display_images:
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.plot(iteration_array, distance_between_points)
        #plt.plot(iteration_array[break_idx], distance_between_points[break_idx], "x")
        plt.xlabel("Iteration")
        plt.ylabel("Distance")
        plt.title("Distance vs. Iteration")

        plt.subplot(132)
        plt.plot(iteration_array, rotation_degrees)
        #plt.plot(iteration_array[break_idx], rotation_degrees[break_idx], "x")
        plt.xlabel("Iteration")
        plt.ylabel("Rotation (degrees)")
        plt.title("Rotation vs. Iteration")

        plt.subplot(133)
        plt.plot(iteration_array, translation_mm)
        #plt.plot(iteration_array[break_idx], translation_mm[break_idx], "x")
        plt.xlabel("Iteration")
        plt.ylabel("Translation (mm)")
        plt.title("Translation vs. Iteration")

        plt.tight_layout()
        plt.show()

    aligned_source_spline = source_points.tolist()

    # Create the transformation matrix.
    transformation_matrix = compute_transformation_matrix(source_points_orig, source_points)
    plt.plot(source_points[:, 0], source_points[:, 1], 'x')
    return aligned_source_spline, transformation_matrix



def letter_A_removal(binary_mask, letter_mask_path):
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


def close_binary_mask(binary_mask):
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


def spline_fitting(binary_mask, display_images, save_images_for_controll, page):
    # Apply morphological operations to close gaps in the binary mask.
    kernel = np.ones((30, 30), np.uint8)
    closed_mask_ = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    closed_mask = close_binary_mask(closed_mask_)
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


def plt_cv2_images(title, image):
        cv2.imshow(title, image)
        cv2.waitKey(0)  # Wait for a key press to move to the next image
        cv2.destroyAllWindows()  # Close the image window


def process_tif_file(input_file, z_offset, conversion_factor, save_file, color1, color2, smoothing_kernel_size,
                     threshold_value, display_images, registration_point, carina_point_frame, save_images_for_controll):
    z_coordinate = 0
    previous_contour = None
    point_cloud = []
    transformation_matrix_previous = None
    with Image.open(input_file) as im:
        for page in range(im.n_frames):
            im.seek(page)  # Move to the current page (frame)
            image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array

            open_cv_image = image[crop:, :, ::-1].copy()  # Crop image at xxx pixels from top

            # Apply the color filter
            binary_mask = filter_color(open_cv_image, color1, color2)
            if display_images:
                plt_cv2_images('Color filtered image', binary_mask)

            # Remove letter A filter
            binary_mask_A_removed = letter_A_removal(binary_mask, letter_mask_path="workflow_utils/Image_a.jpg")
            if display_images:
                plt_cv2_images('Color filtered image', binary_mask_A_removed)

            # Apply smoothing and thresholding (thin line removal)
            threshold_mask = smooth_and_threshold(binary_mask_A_removed, smoothing_kernel_size, threshold_value)
            if display_images:
                plt_cv2_images('Thresholded image', threshold_mask)

            # fit a spline to the contour
            current_contour = spline_fitting(threshold_mask, display_images, save_images_for_controll, page)

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
                    aligned_source_spline, transformation_matrix = icp_alignment(current_contour, previous_contour, transformation_matrix_previous, display_images)
                    # Apply transformation also to registration point
                    if page == (carina_point_frame + 8):
                        print("Remove me (+8)")
                        registration_point_homo = np.array([registration_point[0], registration_point[1], 1])
                        registration_point_homo_transformed = np.dot(transformation_matrix, registration_point_homo)
                        registration_point_transformed_pixel = np.array([registration_point_homo_transformed[0], registration_point_homo_transformed[1]])
                        registration_point_transformed = [(xi * conversion_factor, yi * conversion_factor, z_coordinate) for xi, yi in registration_point_transformed_pixel]
                        if save_file:
                            # Write the coordinates to the text file
                            with open("workflow_processed_data_output/aligned_OCT_registration_point.txt", 'w') as file:
                                registration_point[2] = registration_point[2] - 0.8
                                file.write(f"{registration_point_transformed[0]:.2f} {registration_point_transformed[1]:.2f} {registration_point[2]:.2f}\n")
                    # Convert the spline points to NumPy arrays.
                    source_points = np.array(current_contour)
                    aligned_source_points = np.array(aligned_source_spline)
                    transformation_matrix_previous = transformation_matrix

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

                # Calculate the centroid of the aligned points.
                centroid = np.mean(aligned_source_points, axis=0)

                # Shift the points to have the centroid at (0, 0).
                aligned_source_points_shifted = aligned_source_points - centroid

                # Create a 3D point cloud with incremental z-coordinate and convert into mm unit.
                point_cloud_current = [(xi * conversion_factor, yi * conversion_factor, z_coordinate) for xi, yi in
                                    aligned_source_points_shifted]

                if save_file:
                    # Write the coordinates to the text file
                    with open("workflow_processed_data_output/output_point_cloud.txt", 'a') as file:
                        for coord in point_cloud_current:
                            file.write(f"{coord[0]:.2f} {coord[1]:.2f} {coord[2]:.2f}\n")

                point_cloud.append(point_cloud_current)
                previous_contour = aligned_source_points
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
    plt.show()


def find_center_of_letter_X(binary_mask):
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
    center_y = (top_left[1] + bottom_right[1]) // 2

    return binary_mask_retained, (center_x, center_y)



def get_registration_point(input_file, crop, carina_point_frame, display_images, z_offset, save_file, conversion_factor):
    with Image.open(input_file) as im:
        im.seek(carina_point_frame)  # Move to marked page (frame)
        image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array

        open_cv_image = image[crop:, :, ::-1].copy()  # Crop image at xxx pixels from top

        # Apply the color filter
        binary_mask = filter_color(open_cv_image, color1, color2)
        if display_images:
            plt_cv2_images('Color filtered image', binary_mask)
        
        # Find the coordinates of the letter X
        binary_mask_x, position = find_center_of_letter_X(binary_mask, )
        if display_images:
            plt_cv2_images('Color filtered image', binary_mask_x)
    
    # Convert into mm unit
    registration_point = [position[0] * conversion_factor, position[0] * conversion_factor, carina_point_frame*z_offset]
    
    return np.array(registration_point)


if __name__ == "__main__":
    color1 = (0, 255, 0)  # Green circle
    color2 = (192, 220, 192)  # Circle dots color

    # Initialize the z-coordinate for the first image
    z_offset = 0.1  # Increment by 0.2 mm
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
    save_file = True
    display_images = False
    save_images_for_controll = False 

    # Process values
    registration_point = get_registration_point(input_file, crop, carina_point_frame, display_images, z_offset, save_file, conversion_factor)
    process_tif_file(input_file, z_offset, conversion_factor, save_file, color1, color2, smoothing_kernel_size, threshold_value, display_images, registration_point, carina_point_frame, save_images_for_controll)

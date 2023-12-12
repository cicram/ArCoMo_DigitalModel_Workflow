import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Function to filter by RGB color
def filter_color(input_image, color):
    # Create masks for the two colors
    color = np.array([color[2], color[1], color[0]], dtype=np.uint8)


    binary_mask = cv2.inRange(input_image, color, color)
    # Combine the masks to create a binary mask

    return binary_mask

def plt_cv2_images(title, image):
        cv2.imshow(title, image)
        cv2.waitKey(0)  # Wait for a key press to move to the next image
        cv2.destroyAllWindows()  # Close the image window

if __name__ == "__main__":
    color = [255, 255, 255]
    crop = 200
    crop2 = 900
    OCT_start_frame = 4
    OCT_end_frame = 285
    points = []
    scaling = 103
    z_space = 0.2
    input_file = "stent_data/ArCoMo3_pullback_long_stent_marking_stent.tif"
    with Image.open(input_file) as im:
        for page in range(OCT_start_frame, OCT_end_frame, 1):
            print(page)
            im.seek(page)  # Move to the current page (frame)
            image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
            # Plot the grayscale image
            if False:
                plt.imshow(gray_image, cmap='gray')
                plt.title('Grayscale Image')
                plt.show()
            open_cv_image = image[crop:crop2, :, ::-1].copy()  # Crop image at xxx pixels from top

            # Apply the color filter
            binary_mask = filter_color(open_cv_image, color)
            if False:
                plt_cv2_images('Color filtered image', binary_mask)

            
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
                    points.append([centroid_x/scaling, centroid_y/scaling, page*z_space])
            # Plot the points
            if False:
                plt.scatter(centroid_x_list, centroid_y_list, color='red', marker='.')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.title('Centroid Coordinates of Pixel Clusters')
                plt.show()
            
    with open('sten_pointcloud_output.xyz', 'w') as file:
        for point in points:
            x, y, z = point
            file.write(f"{x:.2f} {y:.2f} {z:.2f}\n")
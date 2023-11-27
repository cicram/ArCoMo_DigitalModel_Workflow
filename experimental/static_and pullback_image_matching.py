import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Example using SIFT features and FLANN matching
sift = cv2.SIFT_create()
OCT_start_frame_pullback = 100
OCT_end_frame_pullback = 140
OCT_start_frame_stationary = 5
OCT_end_frame_stationary = 50
input_file_pullback = "C:/Users/JL/ArCoMo_Data/ArCoMo3/OCT Pullback ArCoMo3 5mm.tif"
input_file_stationary = "C:/Users/JL/ArCoMo_Data/ArCoMo3/Stationary/ArCoMo3_oct_stat_mid_lad_pre_pci_blank.tif"

len_pull = OCT_end_frame_pullback - OCT_start_frame_pullback
len_stat = OCT_end_frame_stationary - OCT_start_frame_stationary
# Initialize similarity matrix
similarity_matrix = np.zeros((len_stat, len_pull))
crop = 100

def plt_cv2_images(title, image):
        cv2.imshow(title, image)
        cv2.waitKey(0)  # Wait for a key press to move to the next image
        cv2.destroyAllWindows()  # Close the image window

#Load images
with Image.open(input_file_pullback) as im_pullback:
    for page_pullback in range(OCT_start_frame_pullback, OCT_end_frame_pullback, 1):
        print("Pullback frame number:" + str(page_pullback))
        im_pullback.seek(page_pullback)  # Move to the current page (frame)
        image_pullback = np.array(im_pullback.convert('RGB'))  # Convert PIL image to NumPy array
        image_pullback = image_pullback[:-crop, :, ::-1].copy()  # Crop image at xxx pixels from top
        # Extract features
        kp2, des2 = sift.detectAndCompute(image_pullback, None)
        #plt_cv2_images('Pullback image', image_pullback)

        with Image.open(input_file_stationary) as im_stationary:
            for page_stationary in range(OCT_start_frame_stationary, OCT_end_frame_stationary, 1):
                print("Stationary frame number:" + str(page_stationary))
                im_stationary.seek(page_stationary)  # Move to the current page (frame)
                image_stationary = np.array(im_stationary.convert('RGB'))  # Convert PIL image to NumPy array
                image_stationary = image_stationary[:-crop, :, ::-1].copy()  # Crop image at xxx pixels from top
                #plt_cv2_images('Stationary image', image_stationary)

                # Extract features
                kp1, des1 = sift.detectAndCompute(image_stationary, None)

                # Use FLANN matcher
                flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
                matches = flann.knnMatch(des1, des2, k=2)

                # Apply Lowe's ratio test to filter good matches
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

                # Compute similarity score (you can use the number of good matches or other criteria)
                similarity_matrix[page_stationary-OCT_start_frame_stationary, page_pullback-OCT_start_frame_pullback] = len(good_matches)

# Print or use the similarity matrix as needed
# Find indices of the maximum similarity value
max_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
max_row, max_col = max_indices

# Print the indices where the maximum similarity value is found
print("Maximum similarity value found at (row, col):", max_row, max_col)


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for the indices
x, y = np.meshgrid(range(len_pull), range(len_stat))

# Plot the 3D surface
ax.plot_surface(x, y, similarity_matrix, cmap='viridis')

# Find indices of the maximum similarity value
max_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
max_row, max_col = max_indices

# Mark the maximum point with a red dot
ax.scatter(max_col,max_row, similarity_matrix[max_row, max_col], color='red', s=100, label='Max Similarity')

# Label the axes
ax.set_xlabel('Image Series Pullback')
ax.set_ylabel('Image Series Stationary')
ax.set_zlabel('Similarity')

# Display the plot
plt.show()
import cv2
import numpy as np
from PIL import Image

# Load template
template = cv2.imread('C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/ArCoMo_Data/ArCoMo7/ArCoMo7_oct_catheter_ref.png', cv2.IMREAD_COLOR)

# Assuming TIF images are named as img_1.tif, img_2.tif, etc.
input_file = 'C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/ArCoMo_Data/ArCoMo7/ArCoMo7_oct_blank.tif'

# Display template
cv2.imshow("template", template)
cv2.waitKey(0)  
cv2.destroyAllWindows()

# Function to rotate an image around a specified center
def rotate_image(image, angle, center):
    rot_matrix = cv2.getRotationMatrix2D(center, angle/10, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

    return rotated_image

# Function to find the best translation and rotation
def find_best_translation_rotation(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    if False:
        cv2.imshow("image", image)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

    h, w, _ = template.shape
    center = (w // 2, h // 2)

    cropped_image = image[max_loc[1] :max_loc[1] + h, 
                          max_loc[0] :max_loc[0] + h]

    if False:   
        cv2.imshow("cropped image", cropped_image)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

    # Best translation

    # Search for the best rotation around the translated position
    best_rotation = 0
    best_match_score = -1

    for angle in range(-100, 100):
        rotated_template = rotate_image(template, angle, center)
        
        result = cv2.matchTemplate(cropped_image, rotated_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_match_score:
            best_match_score = max_val
            best_rotation = angle

    return max_loc, best_rotation

rotation_angles = []

with Image.open(input_file) as im:
    for page in range(3, 200, 1):
        im.seek(page)
        image = np.array(im.convert('RGB'))
        image_flipped = np.flipud(image)
        height, width, channels = image_flipped.shape
        open_cv_image = image_flipped[0: height-130, :, ::-1].copy()

        # Find best translation and rotation
        translation, rotation = find_best_translation_rotation(open_cv_image, template)

        print(f"Frame {page}: Translation = {translation}, Rotation = {rotation} degrees")
        rotation_angles.append(rotation)

# Print the rotation angles for each frame
for i, angle in enumerate(rotation_angles, start=1):
    print(f"Frame {i}: Rotation Angle = {angle} degrees")

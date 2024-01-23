import cv2
import numpy as np
from PIL import Image, ImageChops
import numpy as np
import math 
import matplotlib.pyplot as plt

def rmsdiff(x, y):
  """Calculates the root mean square error (RSME) between two images"""
  errors = np.asarray(ImageChops.difference(x, y)) / 255
  return math.sqrt(np.mean(np.square(errors)))

def rotate_image(image, angle, center):
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

    return rotated_image

# Assuming TIF images are named as img_1.tif, img_2.tif, etc.
input_file = 'C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/ArCoMo_Data/ArCoMo14/ArCoMo14_oct_blank.tif'
output_folder = "C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/temp/my_rotated_images/"
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

        #open_cv_image = image_flipped[450: height-450, 450: width-450, ::-1].copy()

        h, w, _ = open_cv_image.shape
        center = (w // 2, h // 2)
        if previous_image is not None:
            im1 = Image.fromarray(open_cv_image)            
            im2 = Image.fromarray(previous_image)
            #im2.show()
            #im2.show()

            mse = []
            for i in range(-30, 30):
                im2_rot = im2.rotate(i/10)
                #im2_rot.show()

                mse.append(rmsdiff(im1,im2_rot))

            print((mse.index(min(mse))-30)/10)
            rotation_total += (mse.index(min(mse))-30)/10
            with open("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/temp/rotations.txt", 'a') as file:
                file.write(f"{page} {rotation_total:.2f} {(mse.index(min(mse))-30)/10:.2f} \n")

        previous_image = open_cv_image
        rotation_angles.append(rotation_total)
        rotated_image = rotate_image(open_cv_image, rotation_total, center)   

        
        output_filename = f"{output_folder}rotated_image_{page}.png"
        #cv2.imwrite(output_filename, rotated_image)
        
plt.plot(rotation_angles)
plt.show()

if False:
    input_file = 'C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/ArCoMo_Data/ArCoMo7/ArCoMo7_oct_blank.tif'

    # Function to rotate an image around a specified center
    def rotate_image(image, angle, center):
        rot_matrix = cv2.getRotationMatrix2D(center, angle/10, 1.0)
        rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

        return rotated_image

    # Function to find the best translation and rotation
    def find_best_rotation(image, image_previous):

        h, w, _ = image.shape
        center = (w // 2, h // 2)

        if False:   
            cv2.imshow("cropped image", cropped_image)
            cv2.waitKey(0)  
            cv2.destroyAllWindows()

        # Best translation

        # Search for the best rotation around the translated position
        best_rotation = 0
        best_match_score = -1

        for angle in range(-30, 30):
            rotated_image = rotate_image(image, angle, center)
            result = cv2.matchTemplate(rotated_image, image_previous, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_match_score:
                max_loc_best = max_loc
                best_match_score = max_val
                best_rotation = angle
        return best_rotation

    output_folder = "C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/temp/my_rotated_images/"
    rotation_angles = []
    rotation_total = 0
    previous_image = None
    with Image.open(input_file) as im:
        for page in range(2, 270, 1):
            im.seek(page)
            image = np.array(im.convert('RGB'))
            image_flipped = image #np.flipud(image)
            height, width, channels = image_flipped.shape
            open_cv_image = image_flipped[300: height-300, 300: width-300, ::-1].copy()

            #open_cv_image = image_flipped[450: height-450, 450: width-450, ::-1].copy()

            h, w, _ = open_cv_image.shape
            center = (w // 2, h // 2)
            if previous_image is not None:
                # Find best translation and rotation
                rotation = find_best_rotation(open_cv_image, previous_image)

                rotation_total += rotation/10
                rotation_angles.append(rotation_total)

                rotated_image = rotate_image(open_cv_image, rotation_total*(10), center)   
                            
                if False:
                    cv2.imshow("rotated image", rotated_image)
                    cv2.imshow("original image", open_cv_image)

                    cv2.waitKey(0)  
                    cv2.destroyAllWindows()
                                # Save rotated image as TIFF
                output_filename = f"{output_folder}rotated_image_{page}.png"
                cv2.imwrite(output_filename, rotated_image)
                if False:
                    cv2.imshow("rotated image", rotated_image)
                    cv2.imshow("original image", open_cv_image)

                    cv2.imshow("template image", previous_image)

                    cv2.waitKey(0)  
                    cv2.destroyAllWindows()
            previous_image = open_cv_image

    import matplotlib.pyplot as plt
    plt.plot(rotation_angles)
    plt.show()

    # Print the rotation angles for each frame
    for i, angle in enumerate(rotation_angles, start=1):
        print(f"Frame {i}: Rotation Angle = {angle} degrees")






if False:
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
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

        return rotated_image

    # Function to find the best translation and rotation
    def find_best_translation_rotation(image, template):
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        if True:
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

        for angle in range(-180, 180):
            rotated_template = rotate_image(template, angle, center)
            cv2.imshow("image", rotated_template)
            
            cv2.waitKey(0)  
            cv2.destroyAllWindows()
            result = cv2.matchTemplate(cropped_image, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_match_score:
                best_match_score = max_val
                best_rotation = angle

        return max_loc, best_rotation

    rotation_angles = []

    with Image.open(input_file) as im:
        for page in range(2, 298, 1):
            im.seek(page)
            image = np.array(im.convert('RGB'))
            image_flipped = np.flipud(image)
            height, width, channels = image_flipped.shape
            open_cv_image = image_flipped[0: height-130, :, ::-1].copy()

            # Find best translation and rotation
            translation, rotation = find_best_translation_rotation(open_cv_image, template)

            print(f"Frame {page}: Translation = {translation}, Rotation = {rotation} degrees")
            rotation_angles.append(rotation)

    import matplotlib.pyplot as plt
    plt.plot(rotation_angles)
    plt.show()

    # Print the rotation angles for each frame
    for i, angle in enumerate(rotation_angles, start=1):
        print(f"Frame {i}: Rotation Angle = {angle} degrees")

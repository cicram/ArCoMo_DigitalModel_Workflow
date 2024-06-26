from PIL import Image
import numpy as np
import cv2 as cv
from scipy.interpolate import splprep, splev
import numpy as np
from scipy.ndimage import rotate
from PIL import Image
import numpy as np
import cv2 as cv
import numpy as np
import math
from dijkstar import find_path

class ContourDrawer:
    def __init__(self):
        self.retPt = []
        self.drawing = False
        self.path_total = []
        self.final = None
        self.flag_first_point = True
        self.x_previous = 0
        self.y_previous = 0

    def click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.retPt = [(x, y)]

        elif event == cv.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.flag_first_point = True
        elif event == cv.EVENT_MOUSEMOVE and self.drawing:
            self.retPt.append((x, y))
            self.path_total.append(self.retPt.copy())
            if not self.flag_first_point:
                cv.line(self.final, (self.x_previous, self.y_previous), (x, y), (0, 0, 255), 1)
            cv.imshow("image", self.final)
            self.retPt = []
            self.x_previous = x
            self.y_previous = y
            self.flag_first_point = False 

    def fit_spline_to_path(self, path, z_value, conversion_factor):
        # Extract x and y coordinates from the path
        concatenated_path = [point for subpath in path for point in subpath]
        path_x, path_y = zip(*concatenated_path)
        # Remove duplicates
        points = list(zip(path_x, path_y))
        points = list(dict.fromkeys(points))
        path_x, path_y = zip(*points)

        try:
            # Fit a spline to the path
            tck, _ = splprep([path_x, path_y], s=500, per=True)

            # Evaluate the spline to get points on the fitted spline
            u = np.linspace(0, 1, 500)  # Adjust the number of points as needed
            x, y = splev(u, tck)
            z_values = [z_value] * len(x)
            # Return the fitted spline as a list of points
            return list(zip(x*conversion_factor, y*conversion_factor, z_values))
        except: 
            print("could not fit spline")
            return None

    def segment_calc(self, input_file, OCT_start_frame, OCT_end_frame, crop_top, crop_bottom, z_distance, conversion_factor, existing_contours):
        saved_splines = []

        with Image.open(input_file) as im:
            for page in range(OCT_start_frame, OCT_end_frame, 1):
                print(f"Current page: {page}")
                next_slide = False
                self.path_total = []
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                image_flipped = np.flipud(image)
                height, width, channels = image_flipped.shape
                open_cv_image = image_flipped[crop_bottom: height-crop_top, :, ::-1].copy()  # Crop image at xxx pixels from top

                self.final = open_cv_image.copy()

                # Opens image and allows drawing with the mouse
                cv.namedWindow("image")
                cv.setMouseCallback("image", self.click)
                cv.imshow("image", self.final)

                if existing_contours:
                    for existing_contour in existing_contours:
                        # Check if the existing contour matches the current page
                        existing_z_value = existing_contour[0][2]
                        if existing_z_value == (page - OCT_start_frame) * z_distance:
                            # Draw the existing contour
                            cv.polylines(self.final, [np.array(existing_contour)[:, :2].astype(int)], isClosed=True, color=(0, 0, 255), thickness=1)
                            # Add x and y values to self.path_total
                            existing_x, existing_y, existing_z = zip(*existing_contour)
                            existing_x = [int(val/conversion_factor) for val in existing_x]
                            existing_y = [int(val/conversion_factor) for val in existing_y]
                            self.path_total.append(list(zip(existing_x, existing_y)))
                            for itr in range(len(self.path_total[0])-1):
                                cv.line(self.final, (self.path_total[0][itr]), (self.path_total[0][itr+1]), (0, 0, 255), 1)
                            cv.imshow("image", self.final)

                while True:
                    cv.imshow("image", self.final)
                    key = cv.waitKey(2) & 0xFF

                    if key == ord("r") or key == ord("R"):
                        # Reset path if the "R" key is pressed
                        self.final = open_cv_image.copy()
                        self.path_total = []

                    elif key == ord("s") or key == ord("S"):
                        # Save the path if the "S" key is pressed
                        if self.path_total != []:
                            # Fit a spline to the path and save the result
                            fitted_spline = self.fit_spline_to_path(self.path_total, (page-OCT_start_frame)*z_distance, conversion_factor)
                            if fitted_spline is not None:
                                saved_splines.append(fitted_spline)
                                break
                            else:
                                print("could not fit a spline, try again!")
                                self.final = open_cv_image.copy()
                                self.path_total = []
                        else:
                            break
                cv.destroyWindow('image')


        return saved_splines

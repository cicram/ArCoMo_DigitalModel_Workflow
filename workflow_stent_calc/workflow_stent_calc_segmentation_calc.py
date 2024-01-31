from PIL import Image
import numpy as np
import cv2 as cv
import math
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
            return None

    def segment_calc(self, input_file, OCT_start_frame, OCT_end_frame, crop_top, z_distance, conversion_factor):
        saved_splines = []

        with Image.open(input_file) as im:
            for page in range(105, 115, 1):
                next_slide = False
                self.path_total = []
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                image_flipped = np.flipud(image)
                height, width, channels = image_flipped.shape
                open_cv_image = image_flipped[0: height-crop_top, :, ::-1].copy()  # Crop image at xxx pixels from top

                self.final = open_cv_image.copy()

                # Opens image and allows drawing with the mouse
                cv.namedWindow("image")
                cv.setMouseCallback("image", self.click)
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

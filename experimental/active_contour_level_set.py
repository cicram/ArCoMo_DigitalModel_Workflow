import numpy as np
import cv2
import matplotlib.pyplot as plt

# Global variables
ix, iy = -1, -1
drawing = False
rect_over = False
rect = (0, 0, 1, 1)

# Mouse callback function for OpenCV window
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect_over, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), -1)
            rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), -1)
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        rect_over = True

# Load the image
img = cv2.imread('experimental/Test_live_wire.JPG', 1)
img_backup = img.copy()

# Create a window and bind the function to window
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or rect_over:
        break

cv2.destroyAllWindows()

# Initialize mask with zeros
mask = np.zeros(img.shape, np.uint8)

# Draw the rectangle on the mask
cv2.rectangle(mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, -1)

# Perform GrabCut segmentation
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(img_backup, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Create mask where sure and likely backgrounds set to 0, otherwise 1
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Multiply image with new mask to subtract background
img = img_backup * mask2[:, :, None]
# Display the image
plt.imshow(img), plt.colorbar(), plt.show()

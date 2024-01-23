from PIL import Image, ImageChops
import numpy as np
import math 
import matplotlib.pyplot as plt

def rmsdiff(x, y):
  """Calculates the root mean square error (RSME) between two images"""
  errors = np.asarray(ImageChops.difference(x, y)) / 255
  return math.sqrt(np.mean(np.square(errors)))


im1 = Image.open('C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/temp/my_rotated_images/rotated_image_3.png')

im2 = Image.open('C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/temp/my_rotated_images/rotated_image_8.png')
print(im1)

mse = []
for i in range(-50, 50):
  im2_rot = im2.rotate(i/10)
  mse.append(rmsdiff(im1,im2_rot))

print((mse.index(min(mse))-50)/10) 
plt.plot(mse)
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# Read the data from the text file
data = np.genfromtxt("workflow_data/centerline_phantom.txt", skip_header=1, dtype=float, usecols=(0, 2))

data = np.insert(data, 0, [0.0, 0.0], axis=0)
# Extract the Px, Py, and Pz values
x = data[:, 0]
y = data[:, 1]

# Calculate cumulative distance
cumulative_distance = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
cumulative_distance = np.insert(cumulative_distance, 0, 0.0)  # Insert zero at the beginning

# Create an interpolation function for x and y based on cumulative distance
fx = interp1d(cumulative_distance, x, kind='linear')
fy = interp1d(cumulative_distance, y, kind='linear')

# Calculate new cumulative distances at 0.1 mm intervals
new_cumulative_distance = np.arange(0, cumulative_distance[-1], 0.1)

# Interpolate x and y values at the new cumulative distances
new_x = fx(new_cumulative_distance)
new_y = fy(new_cumulative_distance)

# Create an array with all zeros
zeros = np.zeros_like(new_x)

# Stack new_x, zeros, and new_y horizontally
output_data = np.column_stack((new_x, zeros, new_y))

# Define the filename for the output text file
output_filename = "workflow_data/centerline_phantom.txt"

# Save the data to a text file
#np.savetxt(output_filename, output_data, fmt='%.4f', delimiter=' ')

# Create a 2D plot of the resampled points
plt.plot(new_x, new_y, "x")

print(len(new_x))
# Set axis labels
plt.xlabel('Px')
plt.ylabel('Py')

# Display the plot
plt.show()
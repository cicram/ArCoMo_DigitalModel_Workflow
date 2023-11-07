if True: 
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline

    # Define your X, Y, and Z coordinates.
    x = np.array([0, 3, -2.5, 1, -0.5])
    y = np.linspace(0, 3, len(x))
    z = np.array([0, 15, 40, 50, 60])

    # Create a cubic spline interpolating function for X, Y, and Z.
    spline_x = CubicSpline(z, x)
    spline_y = CubicSpline(z, y)
    spline_z = CubicSpline(z, z)

    # Initialize variables
    resampled_points = np.array([x[0], y[0], z[0]])  # Start with the first point
    current_position = np.array([x[0], y[0], z[0]])
    current_index = 0
    interval_distance = 0.1  # Distance for resampling

    # Generate more points for the spline
    z_new = np.linspace(0, 60, 100)  # Adjust the number of points as needed
    x_new = spline_x(z_new)
    y_new = spline_y(z_new)
    z_new = spline_z(z_new)

    while current_index < len(z_new) - 1:
        next_position = np.array([x_new[current_index + 1], y_new[current_index + 1], z_new[current_index + 1]])
        direction_vector = next_position - current_position
        direction_length = np.linalg.norm(direction_vector)

        # Check if we need to insert points along the segment
        while direction_length >= interval_distance:
            t = interval_distance / direction_length
            new_point = current_position + t * direction_vector
            resampled_points = np.vstack((resampled_points, new_point))

            current_position = new_point
            direction_vector = next_position - current_position
            direction_length = np.linalg.norm(direction_vector)

        current_index += 1

    # Add the last point of the original centerline
    resampled_points = np.vstack((resampled_points, np.array([x_new[-1], y_new[-1], z_new[-1]])))

    # Add the second straight line from z = 10 to x = 10 and the same y level
    z_intersection = 10
    y_intersection = spline_y(z_intersection)  # Get the corresponding y value
    x_intersection = spline_x(z_intersection)  # Get the corresponding y value
    x_straight_line = np.linspace(x_intersection, x_intersection + 3, 10)  # Straight line from x = 10 to x = 0
    y_straight_line = np.full_like(x_straight_line, y_intersection)
    z_straight_line = np.full_like(x_straight_line, z_intersection)
    second_line = np.column_stack((x_straight_line, y_straight_line, z_straight_line))

    # Combine the resampled spline and the straight line
    combined_curve = np.vstack((resampled_points, second_line))
    regi_point = np.array([(x_intersection + 0.75), y_intersection, (z_intersection +1)])
    np.savetxt('registration_point_CT2.xyz', regi_point, delimiter=' ')

    # Save all points (resampled and the second curve) in the same XYZ file
    np.savetxt('combined_curve.xyz', combined_curve, delimiter=' ')

    # Save the first curve (resampled points) in a separate XYZ file
    np.savetxt('resampled_curve.xyz', resampled_points, delimiter=' ')

    # Save the second curve in a separate XYZ file
    np.savetxt('second_curve.xyz', second_line, delimiter=' ')


    # Plot the combined curve in 3D.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(combined_curve[:,0], combined_curve[:,2], combined_curve[:,1], "x", label='Combined Curve')
    ax.plot(x_intersection + 1, z_intersection + 1, y_intersection ,"o")
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

def rotation_matrix_from_vectors(vec1, vec2):
    # Normalize the input vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    if False:
        # Calculate the cross product to find the rotation axis
        axis = np.cross(vec1, vec2)

        # Calculate the dot product to find the cosine of the angle
        cos_theta = np.dot(vec1, vec2)

        # Calculate the sine of the angle using the magnitude of the cross product
        sin_theta = np.linalg.norm(axis)

        # Normalize the rotation axis
        axis /= sin_theta

        # Construct the rotation matrix
        K = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])

        rotation_matrix = np.identity(3) + K + K @ K * (1 - cos_theta) / (sin_theta ** 2)
    axis = np.cross(vec1, vec2)
    cosA = np.dot(vec1, vec2)
    k = 1.0 / (1.0 + cosA)

    rotation_matrix = np.array([[axis[0] * axis[0] * k + cosA, axis[1] * axis[0] * k - axis[2], axis[2] * axis[0] * k + axis[1]],
                      [axis[0] * axis[1] * k + axis[2], axis[1] * axis[1] * k + cosA, axis[2] * axis[1] * k - axis[0]],
                      [axis[0] * axis[2] * k - axis[1], axis[1] * axis[2] * k + axis[0], axis[2] * axis[2] * k + cosA]])


    return rotation_matrix

# Define your X, Y, and Z coordinates.
x = np.array([0, 3, -2.5, 1, -0.5])
y = np.linspace(0, 3, len(x))
z = np.array([0, 15, 40, 50, 60])

# Create a cubic spline interpolating function for X, Y, and Z.
spline_x = CubicSpline(z, x)
spline_y = CubicSpline(z, y)
spline_z = CubicSpline(z, z)

# Initialize variables
resampled_points = np.array([x[0], y[0], z[0]])  # Start with the first point
current_position = np.array([x[0], y[0], z[0]])
current_index = 0
interval_distance = 0.1  # Distance for resampling

# Generate more points for the spline
z_new = np.linspace(0, 60, 100)  # Adjust the number of points as needed
x_new = spline_x(z_new)
y_new = spline_y(z_new)
z_new = spline_z(z_new)


 # Add the second straight line from z = 10 to x = 10 and the same y level
z_intersection = 10
y_intersection = spline_y(z_intersection)  # Get the corresponding y value
x_intersection = spline_x(z_intersection)  # Get the corresponding y value

while current_index < len(z_new) - 1:
    next_position = np.array([x_new[current_index + 1], y_new[current_index + 1], z_new[current_index + 1]])
    direction_vector = next_position - current_position
    direction_length = np.linalg.norm(direction_vector)

    # Check if we need to insert points along the segment
    while direction_length >= interval_distance:
        t = interval_distance / direction_length
        new_point = current_position + t * direction_vector
        resampled_points = np.vstack((resampled_points, new_point))

        current_position = new_point
        direction_vector = next_position - current_position
        direction_length = np.linalg.norm(direction_vector)

    current_index += 1

# Add the last point of the original centerline
resampled_points = np.vstack((resampled_points, np.array([x_new[-1], y_new[-1], z_new[-1]])))
# Create a figure and a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Loop through the points and plot ellipses
for i in range(len(resampled_points) - 1):
    x_current, y_current, z_current = resampled_points[i][0], resampled_points[i][1], resampled_points[i][2]
    x_next, y_next, z_next = resampled_points[i+1][0], resampled_points[i+1][1], resampled_points[i+1][2]

    # Calculate the direction vector
    direction_vector = np.array([x_next - x_current, y_next - y_current, z_next - z_current])

    # Calculate the length of the direction vector
    direction_length = np.linalg.norm(direction_vector)

    # Normalize the direction vector
    if direction_length > 0:
        direction_vector /= direction_length

    # Create a parametric representation of the ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    a = 2  # Semi-major axis
    b = 1  # Semi-minor axis
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)

    # Rotate the ellipse to match the direction vector
    R = rotation_matrix_from_vectors
    rotated_ellipse = np.dot(R, np.array([ellipse_x, ellipse_y, np.zeros_like(ellipse_x)]))

    # Translate the ellipse to the current (x, y, z) point
    translated_ellipse = rotated_ellipse + np.array([[x_current], [y_current], [z_current]])

    # Plot the ellipse
    ax.plot(translated_ellipse[0], translated_ellipse[1], translated_ellipse[2])

    plt.plot(translated_ellipse[0], translated_ellipse[1], translated_ellipse[2])
    plt.show()
# Set axis labels and display the plot
ax.plot(x_intersection + 1, y_intersection + 1, z_intersection ,"o")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

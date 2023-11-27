import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.interpolate import splprep, splev
import cv2
import numpy as np
from PIL import Image
import vtkmodules.all as vtk

class center_line_registration: 
    def parse_registration_point_OCT(self, file_path):
        data = []
        # Open the text file for reading
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into three values
                parts = line.strip().split()

                # Ensure there are three values on each line
                if len(parts) == 3:
                    # Parse the values as floats and append them to the respective lists
                    px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                    data.append((px, py, pz))
        return np.array(data[0])

    
    def parse_registration_point_CT(self, file_path):
        data = []
        # Open the text file for reading
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into three values
                parts = line.strip().split()

                # Ensure there are three values on each line
                if len(parts) == 5:
                    if parts[0]=="Point":
                    # Parse the values as floats and append them to the respective lists
                        px, py, pz = float(parts[2]), float(parts[3]), float(parts[4])
                        data.append((px, py, pz))

        data = np.array(data)
        center_point = np.mean(data, axis=0)

        return center_point


    def parse_point_cloud_centerline(self, file_path, display_results):
        flag_header = False
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) == 21:  # Ensure the line has at least 3 values
                    if not flag_header:
                        flag_header = True
                    else:
                        px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                        data.append((px, py, pz))
                else:
                    flag_header = False

        data = np.array(data)

        if display_results:
            x_filtered = data[:, 0]
            y_filtered = data[:, 1]
            z_filtered = data[:, 2]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_filtered, y_filtered, z_filtered, c="blue", marker='o')
            ax.set_xlabel('Px')
            ax.set_ylabel('Py')
            ax.set_zlabel('Pz')
            plt.show()

        return data


    def resample_center_line(self, center_line, display_results, z_distance):
        # Desired distance between resampled points
        interval_distance = z_distance

        # Initialize variables
        resampled_points = [center_line[0]]  # Start with the first point
        current_position = center_line[0]
        current_index = 0

        while current_index < len(center_line) - 1:
            next_position = center_line[current_index + 1]
            direction_vector = next_position - current_position
            # get length between points
            direction_length = np.linalg.norm(direction_vector)

            # Check if we need to insert points along the segment, if segment is finished, do not interpolate anymore (distance > xxx)
            if direction_length <= 4.0:
                while direction_length >= interval_distance:
                    # Calculate the next resampled point and add it
                    t = interval_distance / direction_length
                    new_point = current_position + t * direction_vector
                    resampled_points.append(new_point)

                    # Update the current position and remaining length
                    current_position = new_point
                    direction_vector = next_position - current_position
                    direction_length = np.linalg.norm(direction_vector)
            else:
                # Skip interpolation and move to the next point
                resampled_points.append(next_position)
                current_position = next_position

            # Move to the next point
            current_index += 1

        # Add the last point of the original centerline
        resampled_points.append(center_line[-1])

        # Convert the resampled points to a NumPy array
        resampled_points = np.array(resampled_points)

        if display_results:
            x_orig = center_line[:, 0]
            y_orig = center_line[:, 1]
            z_orig = center_line[:, 2]
            x_resampled = resampled_points[:, 0]
            y_resampled = resampled_points[:, 1]
            z_resampled = resampled_points[:, 2]
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_resampled, y_resampled, z_resampled, c="blue", marker='o')
            ax.scatter(x_orig, y_orig, z_orig, c="red", marker='o')

            ax.set_xlabel('Px')
            ax.set_ylabel('Py')
            ax.set_zlabel('Pz')
            plt.show()

        return resampled_points


    def find_centerline_vectors(self, center_line_pp, display_results):
        # Initialize an array to store the vectors
        vectors = []

        # Iterate through the points
        for i in range(len(center_line_pp-1)):
            # Get the neighboring points
            prev_idx = max(i - 1, 0)
            next_idx = min(i + 1, len(center_line_pp) - 1)

            # Calculate vectors from the current point to its neighbors
            #prev_vector = center_line_pp[i] - center_line_pp[prev_idx]
            next_vector = center_line_pp[next_idx] - center_line_pp[i]

            # Calculate the normal vector using the cross product
            # normal_vector = np.cross(prev_vector, next_vector)

            # Normalize the vector
            next_vector = next_vector / np.linalg.norm(next_vector)
            vectors.append(next_vector)

        if display_results:
            # Create a figure and axis for the plot
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            # Plot the centerline as a line connecting the points
            ax.plot(center_line_pp[:, 0], center_line_pp[:, 1], center_line_pp[:, 2], marker='o', linestyle='-', color='b', label='Centerline')

            # Plot the normal vectors as quivers at each point
            for i in range(len(center_line_pp)):
                x, y, z = center_line_pp[i]
                nx, ny, nz = vectors[i]
                ax.quiver(x, y, z, nx, ny, nz, length=0.1, normalize=True, color='r')

            # Set labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

            # Show the plot
            plt.show()

        return vectors

    def parse_OCT_lumen_point_cloud(self, file_path):
        data = []
        # Open the text file for reading
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into three values
                parts = line.strip().split()
                # Ensure there are three values on each line
                if len(parts) == 3:
                    # Parse the values as floats and append them to the respective lists
                    px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                    data.append((px, py, pz))

        # Create a dictionary to store grouped data with z-values as keys
        centered_grouped_data = {}

        # Iterate through the data and group by z-values
        for x, y, z in data:
            if z not in centered_grouped_data:
                centered_grouped_data[z] = []
            centered_grouped_data[z].append([x, y, z])

        # Flip the order of the groups by sorting based on z-values
        sorted_data = sorted(centered_grouped_data.items(), key=lambda item: item[0], reverse=True)

        # Create a new dictionary with updated z-values
        flipped_grouped_data = {}
        max_z = sorted_data[0][0]  # Get the maximum z-value
        for i, (z, group) in enumerate(sorted_data):
            if i == 0:
                flipped_grouped_data[0] = group  # Assign the first group to z=0
            else:
                flipped_grouped_data[max_z - z] = group

        # Round the z values to one decimal place
        rounded_grouped_data = {}
        for z, group in flipped_grouped_data.items():
            rounded_z = round(z, 1)
            rounded_grouped_data[rounded_z] = group

        return rounded_grouped_data


    def rotation_matrix_from_vectors(self, vec1, vec2):
        # Calculate the dot product
        dot_product = np.dot(vec1, vec2)

        # Calculate the magnitudes of the vectors
        magnitude_vec1 = np.linalg.norm(vec1)
        magnitude_vec2 = np.linalg.norm(vec2)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_vec1 * magnitude_vec2)

        # Calculate the angle in radians
        theta_radians = np.arccos(cos_theta)

        # Convert the angle to degrees
        rot_angle = np.degrees(theta_radians)
        with open("workflow_processed_data_output/image_translations/rotation_angel_registration.txt", 'w') as file:
            file.write(f"{rot_angle:.2f}\n")
        
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


    def find_closest_point_index(self, point_cloud, target_point):
        # Convert the point cloud to a NumPy array for efficient calculations
        point_cloud_array = np.array(point_cloud)

        # Calculate the Euclidean distances between the target point and all points in the point cloud
        distances = np.linalg.norm(point_cloud_array - target_point, axis=1)

        # Find the index of the point with the minimum distance
        closest_point_index = np.argmin(distances)

        return closest_point_index


    def register_OCT_frames_onto_centerline(self, grouped_OCT_frames, centerline_registration_start, centerline_vectors,
                                            resampled_pc_centerline, OCT_registration_frame, z_distance, rotated_registration_point_OCT, save_file, display_results):
        saved_registered_splines = []
        rotated_vectors = []
        target_centerline_point_display = []
        orig_frames = []
        z_level_preivous = None

        ####################################################################
        # Create a VTK renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1, 1, 1)  # Set background color to white

        # Create a VTK render window
        render_window = vtk.vtkRenderWindow()
        render_window.SetWindowName("VTK 3D Visualization")
        render_window.SetSize(800, 800)
        render_window.AddRenderer(renderer)

        # Create a VTK render window interactor
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)
        ####################################################################


        # Find start idx to know at what centerline index they have to be aligned 
        z_level_registration = round(OCT_registration_frame * z_distance, 1)
        count = 0
        for z_level, frame_points in grouped_OCT_frames.items():
            if z_level > z_level_registration:
                count += 1
        closest_centerline_point_idx = centerline_registration_start - count

        # Iterate through the splines and align them on the centerline points.
        for z_level, frame_points in grouped_OCT_frames.items():
            # Find the corresponding centerline point and its vector.
            if z_level_preivous is None:
                z_level_preivous = z_level

            closest_centerline_point_idx += round((z_level - z_level_preivous) / z_distance)
            z_level_preivous = z_level

            target_centerline_point = resampled_pc_centerline[closest_centerline_point_idx]
            normal_vector = centerline_vectors[closest_centerline_point_idx]

            # Calculate the transformation matrix to align the spline with the centerline point's vector.
            source_normal_vector = np.array([0, 0, -1])

            rotation_matrix = self.rotation_matrix_from_vectors(source_normal_vector, normal_vector)

            registered_spline = np.array(frame_points)  # Copy the spline points
            z_visual = frame_points[0][2]
            # Apply the rotation to the entire frame (spline points).
            registered_spline = np.dot(rotation_matrix, registered_spline.T).T  # Apply rotation

            # Perform the translation to center the spline on the centerline point.
            translation_vector = target_centerline_point - registered_spline.mean(axis=0)
            registered_spline += translation_vector

            ####################################################################################
            input_file_pullback = "C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_data/OCT.tif"
            print(z_level)
            if z_level == 1 or z_level == 10 or z_level == 20 or z_level == 30 or z_level == 40 or z_level == 50:
                with Image.open(input_file_pullback) as im_pullback:
                    # centerline_point = center_line[idx_start - page_pullback]

                    # Image to be loaded
                    page_pullback = round((57 - z_level)/0.2)
                    im_pullback.seek(page_pullback)  # Move to the current page (frame)
                    image_pullback = np.array(im_pullback.convert('RGB'))  # Convert PIL image to NumPy array
                    image_pullback = image_pullback[:, :, ::-1].copy()  # Crop image at xxx pixels from top

                    gray_image = cv2.cvtColor(image_pullback, cv2.COLOR_BGR2GRAY)

                    # Get image dimensions
                    height, width = gray_image.shape
                    scaling = 98

                    # Create a VTK structured grid
                    structured_grid = vtk.vtkStructuredGrid()
                    structured_grid.SetDimensions(gray_image.shape[1], gray_image.shape[0], 1)

                    # Create VTK points and assign the image data
                    points = vtk.vtkPoints()
                    colors = vtk.vtkUnsignedCharArray()
                    colors.SetNumberOfComponents(3)  # RGB has three components
                    colors.SetName("Colors")
                    mypoints = []
                    for i in range(gray_image.shape[0]):
                        for j in range(gray_image.shape[1]):
                            mypoints.append(np.array([(j - width / 2) / scaling, (i - height / 2) / scaling, z_visual]))
                    
                    mypoints = np.array(mypoints)
                    rotated_points = np.dot(rotation_matrix, mypoints.T).T  # Apply rotation
                    rotated_points += translation_vector

                    for i in range(height):
                        for j in range(width):
                            points.InsertNextPoint(rotated_points[i * width + j])
                            rgb_values = image_pullback[i, j]
                            colors.InsertNextTuple3(rgb_values[2], rgb_values[1], rgb_values[0])

                    # Set the points and scalars for the structured grid
                    structured_grid.SetPoints(points)
                    structured_grid.GetPointData().SetScalars(colors)

                    # Create a VTK mapper for the structured grid
                    mapper = vtk.vtkDataSetMapper()
                    mapper.SetInputData(structured_grid)
                    mapper.ScalarVisibilityOn()
                    mapper.SetScalarModeToUsePointData()
                    mapper.SetScalarRange(0, 255)  # Set the range of grayscale values

                    # Create a VTK actor for the structured grid
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)

                    # Add the actor to the renderer
                    renderer.AddActor(actor)

            
            ####################################################################################

            # Append the registered spline to the list
            saved_registered_splines.append(registered_spline)
            rot_vec = np.dot(rotation_matrix, source_normal_vector)
            
            # Save registration point
            if z_level == z_level_registration:
            # Write the coordinates to the text file
                rotated_registration_point_OCT = np.dot(rotation_matrix, rotated_registration_point_OCT.T).T  # Apply rotation
                if save_file:
                    with open("workflow_processed_data_output/rotated_OCT_registration_point.txt", 'w') as file:
                        file.write(f"{rotated_registration_point_OCT[0]:.2f} {rotated_registration_point_OCT[1]:.2f} {rotated_registration_point_OCT[2]:.2f}\n")

            # Save rotation and translation
            with open("workflow_processed_data_output/image_translations/centerline_registration_rotation.txt", 'a') as file:
                file.write(f"{rotation_matrix[0][0]:.2f} {rotation_matrix[0][1]:.2f} {rotation_matrix[0][2]:.2f} \
                           {rotation_matrix[1][0]:.2f} {rotation_matrix[1][1]:.2f} {rotation_matrix[1][2]:.2f} \
                           {rotation_matrix[2][0]:.2f} {rotation_matrix[2][1]:.2f} {rotation_matrix[1][2]:.2f}\n")
            center_point = registered_spline.mean(axis=0)
            with open("workflow_processed_data_output/image_translations/centerline_registration_translation_center.txt", 'a') as file:
                file.write(f"{center_point[0]:.2f} {center_point[1]:.2f} {center_point[2]:.2f}\n")

            with open("workflow_processed_data_output/image_translations/centerline_registration_translation.txt", 'a') as file:
                file.write(f"{translation_vector[0]:.2f} {translation_vector[1]:.2f} {translation_vector[2]:.2f}\n")
            frame_points = np.array(frame_points)
            translation_vector = target_centerline_point - frame_points.mean(axis=0)
            orig_frames.append(frame_points + translation_vector)
            rotated_vectors.append(rot_vec)
            target_centerline_point_display.append(target_centerline_point)

        #############################################################################
        points__ = vtk.vtkPoints()

        for point_frame in registered_spline:
            points__.InsertNextPoint(point_frame)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points__)

        # Create a sphere for the points
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(0.3)  # Set the radius of the sphere
        sphere_source.SetThetaResolution(10)
        sphere_source.SetPhiResolution(10)

        # Create a glyph filter to associate the sphere with each point
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere_source.GetOutputPort())
        glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleFactor(1.0)  # Set the scale factor for the spheres

        # Create a mapper for the polydata
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        # Create a VTK actor for the polydata
        actor_centerline = vtk.vtkActor()
        actor_centerline.SetMapper(mapper)
        actor_centerline.GetProperty().SetColor(1.0, 0.0, 0.0)  # Set color to red

        # Add the actor to the renderer
        #renderer.AddActor(actor_centerline)

        # CALC
        # Create a VTK actor for the structured grid
        actor_calc = vtk.vtkActor()
        actor_calc.SetMapper(mapper)

        # Add the actor to the renderer
        renderer.AddActor(actor_calc)

        # Load and display the .obj file
        obj_file_path = 'C:/Users/JL/NX_parts/ArCoMo3_3d_calc.obj'
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_file_path)

        # Create a VTK mapper for the .obj file
        obj_mapper = vtk.vtkPolyDataMapper()
        obj_mapper.SetInputConnection(reader.GetOutputPort())

        # Create a VTK actor for the .obj file
        obj_actor_calc = vtk.vtkActor()
        obj_actor_calc.SetMapper(obj_mapper)
        obj_actor_calc.GetProperty().SetOpacity(0.5)  # Set opacity to 0.5 for semi-transparency
        obj_actor_calc.GetProperty().SetColor([1.0, 0, 0])  # Set color to light blue

        # Add the actor for the .obj file to the renderer
        renderer.AddActor(obj_actor_calc)

        #Outershell
        # Create a VTK actor for the structured grid
        actor_outershell = vtk.vtkActor()
        actor_outershell.SetMapper(mapper)

        # Add the actor to the renderer
        renderer.AddActor(actor_outershell)

        # Load and display the .obj file
        obj_file_path = 'C:/Users/JL/NX_parts/ArCoMo3_3d_outershell.obj'
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_file_path)

        # Create a VTK mapper for the .obj file
        obj_mapper = vtk.vtkPolyDataMapper()
        obj_mapper.SetInputConnection(reader.GetOutputPort())

        # Create a VTK actor for the .obj file
        obj_actor_outershell = vtk.vtkActor()
        obj_actor_outershell.SetMapper(obj_mapper)
        obj_actor_outershell.GetProperty().SetOpacity(0.2)  # Set opacity to 0.5 for semi-transparency
        obj_actor_outershell.GetProperty().SetColor([0, 0, 1.0])  # Set color to light blue

        # Add the actor for the .obj file to the renderer
        renderer.AddActor(obj_actor_outershell)

        # Innershell
        # Create a VTK actor for the structured grid
        actor_innershell = vtk.vtkActor()
        actor_innershell.SetMapper(mapper)

        # Add the actor to the renderer
        renderer.AddActor(actor_innershell)

        # Load and display the .obj file
        obj_file_path = 'C:/Users/JL/NX_parts/ArCoMo3_3d_innershell.obj'
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_file_path)

        # Create a VTK mapper for the .obj file
        obj_mapper = vtk.vtkPolyDataMapper()
        obj_mapper.SetInputConnection(reader.GetOutputPort())

        # Create a VTK actor for the .obj file
        obj_actor_innershell = vtk.vtkActor()
        obj_actor_innershell.SetMapper(obj_mapper)
        obj_actor_innershell.GetProperty().SetOpacity(0.5)  # Set opacity to 0.5 for semi-transparency
        obj_actor_innershell.GetProperty().SetColor([0.0, 1.0, 0])  # Set color to light blue

        # Add the actor for the .obj file to the renderer
        renderer.AddActor(obj_actor_innershell)
        # Create a cube axes actor
        cube_axes_actor = vtk.vtkCubeAxesActor()
        renderer.AddActor(cube_axes_actor)

        # Set cube axes actor properties
        cube_axes_actor.SetUseTextActor3D(1)
        cube_axes_actor.SetBounds(obj_actor_innershell.GetBounds())
        cube_axes_actor.SetCamera(renderer.GetActiveCamera())
        cube_axes_actor.GetTitleTextProperty(0).SetFontSize(1)

        # Customize cube axes actor appearance
        tickColor = [1.0, 0.0, 0.0]  # Set your desired tick color
        cube_axes_actor.GetTitleTextProperty(0).SetColor(tickColor)
        cube_axes_actor.GetLabelTextProperty(0).SetColor(tickColor)
        cube_axes_actor.GetTitleTextProperty(1).SetColor(tickColor)
        cube_axes_actor.GetLabelTextProperty(1).SetColor(tickColor)
        cube_axes_actor.GetTitleTextProperty(2).SetColor(tickColor)
        cube_axes_actor.GetLabelTextProperty(2).SetColor(tickColor)
        cube_axes_actor.GetXAxesLinesProperty().SetColor(tickColor)
        cube_axes_actor.GetYAxesLinesProperty().SetColor(tickColor)
        cube_axes_actor.GetZAxesLinesProperty().SetColor(tickColor)
        cube_axes_actor.GetXAxesGridlinesProperty().SetColor(tickColor)
        cube_axes_actor.GetYAxesGridlinesProperty().SetColor(tickColor)
        cube_axes_actor.GetZAxesGridlinesProperty().SetColor(tickColor)
        cube_axes_actor.XAxisMinorTickVisibilityOff()
        cube_axes_actor.YAxisMinorTickVisibilityOff()
        cube_axes_actor.ZAxisMinorTickVisibilityOff()

        cube_axes_actor.SetXTitle("X")
        cube_axes_actor.SetYTitle("Y")
        cube_axes_actor.SetZTitle("Z")
        cube_axes_actor.SetFlyModeToStaticEdges()

        # Apply transformation to the cube axes actor
        transform = vtk.vtkTransform()
        transform.Translate(0.0, 0.0, 0.0)
        transform.RotateZ(-90)
        transform.RotateY(-90)
        cube_axes_actor.SetUserTransform(transform)

        # Add actors to the renderer
        renderer.AddActor(cube_axes_actor)

        # Reset camera and render
        renderer.ResetCamera()
        render_window.Render()

        # Start the VTK render window interactor
        render_window_interactor.Start()
        ########################################################################


        if save_file or True:

            # Write the data to a text file
            with open("workflow_processed_data_output/saved_registered_splines.txt", "w") as file:
                for spline in saved_registered_splines:
                    spline = spline[::2]
                    for point in spline:
                        file.write(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}\n")

        # Display all the saved registered splines in a single plot
        if display_results:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for spline in saved_registered_splines:
                spline = spline[::30]
                ax.scatter(spline[:, 0], spline[:, 1], spline[:, 2], marker='o')
            # Set labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('Registered Splines')
            plt.show()
        if display_results and False:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            i = 0
            for frame in orig_frames:
                frame = frame[::10]
                i += 1
                if  199 < i < 201:
                    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], color="blue", marker='o')
            i = 0
            for spline in saved_registered_splines:
                spline = spline[::10]
                i += 1
                if  199 < i < 201:
                    ax.scatter(spline[:, 0], spline[:, 1], spline[:, 2], color="red", marker='o')
            i = 0
            for vector in rotated_vectors:
                x, y, z = target_centerline_point_display[i]
                nx, ny, nz = vector
                i += 1
                ax.quiver(x, y, z, nx, ny, nz, length=1.5, normalize=True, color='b')

            for i in range(len(resampled_pc_centerline)):
                x, y, z = resampled_pc_centerline[i]
                nx, ny, nz = centerline_vectors[i]
                ax.quiver(x, y, z, nx, ny, nz, length=1, normalize=True, color='r')
            # Set labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('Registered Splines for the First 3 Z-Levels')
            plt.show()


    def get_oct_lumen_rotation_matrix(self, resampled_pc_centerline, centerline_registration_start, grouped_OCT_frames, registration_point_OCT, registration_point_CT, OCT_registration_frame, z_distance, display_results):
        target_centerline_point = resampled_pc_centerline[centerline_registration_start]
        # find correct frame
        registration_frame = grouped_OCT_frames[round(grouped_OCT_frames[0][0][2] - (OCT_registration_frame * z_distance), 1)]
        registered_frame = np.array(registration_frame)  # Copy the frame points

        # Perform the translation to center the spline and registration point on the centerline point.
        translation_vector = target_centerline_point - registered_frame.mean(axis=0)
        registered_frame += translation_vector
        registration_point_OCT += translation_vector
        if display_results:
            plt.plot(registered_frame[:, 0], registered_frame[:, 1], "x")
            plt.plot(registration_point_OCT[0], registration_point_OCT[1], "o")
            plt.show()

        # Create vectors
        vector_oct_registration_point_ = registration_point_OCT - target_centerline_point
        vector_ct_registration_point_ = registration_point_CT - target_centerline_point
        vector_oct_registration_point = [vector_oct_registration_point_[0], vector_oct_registration_point_[1], 0]
        vector_ct_registration_point = [vector_ct_registration_point_[0], vector_ct_registration_point_[1], 0]

        rotation_matrix = self.rotation_matrix_from_vectors(vector_oct_registration_point, vector_ct_registration_point)
        rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Convert the angle from radians to degrees if needed
        print(np.degrees(rotation_angle))
        ########## Verified angle, is correct, now check in 3d plot howmuch the frames are rotated *****************************
        rotated_registration_point = np.dot(rotation_matrix, registration_point_OCT.T).T 

        rotated_frame = np.dot(registered_frame, rotation_matrix.T)
        translation_vector = target_centerline_point - rotated_frame.mean(axis=0)
        rotated_frame += translation_vector
        rotated_registration_point += translation_vector

        if display_results or True:
            plt.plot(registered_frame[:, 0], registered_frame[:, 1], "x", color="red")
            plt.plot(registration_point_OCT[0], registration_point_OCT[1], "o", color="red")
            plt.plot(rotated_registration_point[0], rotated_registration_point[1], "o", color="blue")
            plt.plot(rotated_frame[:, 0], rotated_frame[:, 1], "x", color="blue")
            plt.plot(registration_point_CT[0], registration_point_CT[1], "x", color="black")
            plt.plot(target_centerline_point[0], target_centerline_point[1], "x", color="green")


            plt.show()

        plt.plot(target_centerline_point[0], target_centerline_point[1], "o", color="black")
        plt.plot(registration_point_OCT[0], registration_point_OCT[1], "x", color="blue")
        plt.plot(registration_point_CT[0], registration_point_CT[1], "x", color="red")
        plt.plot(rotated_registration_point[0], rotated_registration_point[1], "o", color="blue")
        plt.plot([target_centerline_point[0], registration_point_OCT[0]],[target_centerline_point[1], registration_point_OCT[1]], color="blue")
        plt.plot([target_centerline_point[0], registration_point_CT[0]],[target_centerline_point[1], registration_point_CT[1]], color="red")
        plt.plot([target_centerline_point[0], rotated_registration_point[0]],[target_centerline_point[1], rotated_registration_point[1]], color="blue")
        plt.show()

        if display_results:
            print(rotated_registration_point)
            print(registration_point_OCT)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            x, y, z = target_centerline_point
            nx, ny, c = vector_oct_registration_point
            ax.quiver(x, y, z, nx, ny, 0, length=0.1, normalize=True, color='r')
            nx, ny, c = vector_ct_registration_point
            #ax.quiver(x, y, z, nx, ny, 0, length=0.1, normalize=True, color='b')
            nx, ny, nz = vector_oct_registration_point_
            ax.quiver(x, y, z, nx, ny, nz, length=0.1, normalize=True, color='g')

            #ax.scatter(registered_frame[:, 0], registered_frame[:, 1], registered_frame[:, 2], marker=, color="red")
            ax.scatter(registration_point_OCT[0], registration_point_OCT[1], registration_point_OCT[2], marker='o', color="blue")
            ax.scatter(rotated_registration_point[0], rotated_registration_point[1], rotated_registration_point[2], marker='o', color="red")


            x1, y1, z1 = target_centerline_point
            x2, y2, z2 = rotated_registration_point
            # Plot the line
            ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o', linestyle='-', color='b')



            ax.scatter(registration_point_CT[0], registration_point_CT[1], registration_point_CT[2], marker='x')
            ax.scatter(target_centerline_point[0], target_centerline_point[1], target_centerline_point[2], marker='x')
            #ax.scatter(resampled_pc_centerline[:, 0], resampled_pc_centerline[:, 1], resampled_pc_centerline[:, 2], marker='o')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('Registered Splines for the First 3 Z-Levels')
            plt.show()

        return rotation_matrix, rotated_registration_point


    def rotation_matrix_from_vectors_x_y(self, vec1, vec2):
        # Normalize the input vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        v = np.cross(vec1, vec2)
        # s = np.linalg.norm(v)
        c = np.dot(vec1, vec2)

        v1, v2, v3 = v
        h = 1 / (1 + c)

        Vmat = np.array([[0, -v3, v2],
                    [v3, 0, -v1],
                    [-v2, v1, 0]])

        R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
        #return R

        if True:
        # Calculate the angle between the vectors
            theta = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
            # Create the rotation matrix
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0], 
                        [0, 0, 1]])

            return R


    def rotate_frames(self, grouped_OCT_frames, oct_lumen_rotation_matrix):
        # Create an empty dictionary to store the rotated data
        rotated_grouped_data = {}

        # Iterate through the rounded_grouped_data
        for z, group in grouped_OCT_frames.items():
            rotated_group = np.dot(group, oct_lumen_rotation_matrix.T)
            # Store the rotated group in the rotated_grouped_data dictionary
            rotated_grouped_data[z] = rotated_group

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Show the plot
        x_filtered = []
        y_filtered = []
        z_filtered = []
        for z_level, frame_points in grouped_OCT_frames.items():
            for i, data in enumerate(frame_points):
                if z_level < 0.1:
                    x_filtered.append(frame_points[i][0])
                    y_filtered.append(frame_points[i][1])
                    z_filtered.append(frame_points[i][2])
        ax.scatter(x_filtered[::30], y_filtered[::30], z_filtered[::30], c="blue", marker='o')
        x_filtered = []
        y_filtered = []
        z_filtered = []
        for z_level, frame_points in rotated_grouped_data.items():
            for i, data in enumerate(frame_points): 
                if z_level < 0.1:
                    x_filtered.append(frame_points[i][0])
                    y_filtered.append(frame_points[i][1])
                    z_filtered.append(frame_points[i][2])
        ax.scatter(x_filtered[::30], y_filtered[::30], z_filtered[::30], c="red", marker='o')
        ax.set_xlabel('Px')
        ax.set_ylabel('Py')
        ax.set_zlabel('Pz')
        plt.show()

        return rotated_grouped_data


    def parse_marked_point(self, file_path):
        data = []
        # Open the text file for reading
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into three values
                parts = line.strip().split()

                # Ensure there are three values on each line
                if len(parts) == 3:
                    # Parse the values as floats and append them to the respective lists
                    px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                    data.append((px, py, pz))

        data = np.array(data)
        return data


    def smooth_centerline(self, pc_centerline, display_results):
        point_start = self.parse_marked_point("workflow_data/location_2.txt")
        point_end = self.parse_marked_point("workflow_data/location_1.txt")

        indices_start = np.where(np.all(pc_centerline == point_start, axis=1))[0][0]
        indices_end = np.where(np.all(pc_centerline == point_end, axis=1))[0][0]

        fitting_points = []
        number_points = 10

        for i in range(number_points):
            fitting_points.append(pc_centerline[indices_start-number_points-1+i])
    
        for i in range(number_points):
            fitting_points.append(pc_centerline[indices_end+i])

        indices_start_crop =  indices_start - number_points - 1
        indices_stop_crop = indices_end + number_points

        points = fitting_points
        x, y, z = zip(*points)
        # Create a cubic spline using splprep
        tck, u = splprep([x, y, z], s=0, per=0)

        # Evaluate the spline at a higher number of points for smoother appearance
        u_new = np.linspace(0, 1, 50)
        spline_points = splev(u_new, tck)

        pc_centerline = np.delete(pc_centerline, slice(indices_start_crop, indices_stop_crop), axis=0)
        
        # have to flip the spline
        length = len(spline_points[0])-1
        for i, p in enumerate(spline_points[0]):
            point = [spline_points[0][length-i], spline_points[1][length-i], spline_points[2][length-i]]
            pc_centerline = np.insert(pc_centerline, indices_start_crop, point, axis=0)

        if display_results: 
            # Create a 3D scatter plot for the original points
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c='b', label='Original Points', marker='o')

            # Create a 3D line plot for the cubic spline
            ax.plot(spline_points[0], spline_points[1], spline_points[2], "x-", c='r', label='Cubic Spline')

            # Set labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.legend()

            # Show the plot
            x_filtered = pc_centerline[:, 0]
            y_filtered = pc_centerline[:, 1]
            z_filtered = pc_centerline[:, 2]
            ax.scatter(x_filtered, y_filtered, z_filtered, c="blue", marker='o')

            ax.set_xlabel('Px')
            ax.set_ylabel('Py')
            ax.set_zlabel('Pz')
            plt.show()
        
        return pc_centerline

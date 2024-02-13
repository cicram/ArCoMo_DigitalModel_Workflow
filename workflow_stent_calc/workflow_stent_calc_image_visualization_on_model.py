import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.interpolate import splprep, splev
import cv2
import numpy as np
from PIL import Image
import vtkmodules.all as vtk
import math 

class OctImageVisualizier:
    
    def visualize_images(self, input_file, centerline_vectors, rotated_grouped_OCT_lumen, centerline_registration_start, center_line, oct_lumen_rotation_matrix, oct_rotation_matrix, OCT_start_frame, OCT_end_frame, OCT_registration_frame, crop_bottom, crop_top, conversion_factor, z_offset):
        all_image_points = []
        with Image.open(input_file) as im:
            for page in range(OCT_start_frame, OCT_end_frame, 1):
                im.seek(page)  # Move to the current page (frame)
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                image_flipped = np.flipud(image)
                height, width, channels = image_flipped.shape
                open_cv_image = image_flipped[crop_bottom: height-crop_top, :, ::-1].copy()  # Crop image at xxx pixels from top
                gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

                image_points = []
                for i in range(gray_image.shape[0]):
                    for j in range(gray_image.shape[1]):
                        single_point = np.array([j*conversion_factor, i*conversion_factor, round((page - OCT_start_frame)*z_offset,1)])
                        image_points.append(single_point)
                all_image_points.append(image_points)

        ####################################################################
      
        # Align points
        rotated_points = []
        all_image_points = np.array(all_image_points)
        # Calculate the center of rotation
        center_y = (width / 2) * conversion_factor
        center_x = (height / 2) * conversion_factor
        for current_points in all_image_points: # for stents
            current_points = np.array(current_points)
            idx = int(current_points[0][2]/z_offset)
            rot_angle = oct_rotation_matrix[idx]  # Convert the list to a numpy array
            # Translate to the center of rotation
            current_points[:, 0] -= center_x
            current_points[:, 1] -= center_y

            c, s = math.cos(np.radians(rot_angle)), math.sin(np.radians(rot_angle))
        
            # 2D rotation matrix for x-y plane
            rot_xy = np.array([[c, -s],
                            [s, c]])
            
            # Apply rotation only to x-y coordinates
            current_points_xy_rotated = np.dot(current_points[:, :2], rot_xy.T)
            
            # Translate back to the original position
            current_points_xy_rotated[:, 0] += center_x
            current_points_xy_rotated[:, 1] += center_y

            # Combine rotated x-y coordinates with original z-values
            current_points_rotated = np.column_stack((current_points_xy_rotated, current_points[:, 2]))
            
            rotated_points.append(current_points_rotated)
    

        ####################################################################

        # Registration rotation
        rotated_points = np.array(rotated_points)
        rotated_points = np.dot(rotated_points, oct_lumen_rotation_matrix.T)
        
        ####################################################################
        saved_registered_splines = []
        registered_points = []

        z_level_preivous = None
        # Register frames onto centerline
        z_level_registration = round((OCT_registration_frame - OCT_start_frame) * z_offset, 1)
        count = 0

        for z_level, frame_points in rotated_grouped_OCT_lumen.items():
            if z_level > z_level_registration:
                count += 1
        closest_centerline_point_idx = centerline_registration_start - count

        for z_level, frame_points in rotated_grouped_OCT_lumen.items():
            # Find the corresponding centerline point and its vector.
            if z_level_preivous is None:
                z_level_preivous = z_level

            closest_centerline_point_idx += round((z_level - z_level_preivous) / z_offset)
            z_level_preivous = z_level


            target_centerline_point = center_line[closest_centerline_point_idx]
            normal_vector = centerline_vectors[closest_centerline_point_idx]

            # Calculate the transformation matrix to align the spline with the centerline point's vector.
            source_normal_vector = np.array([0, 0, -1])

            rotation_matrix = self.rotation_matrix_from_vectors(source_normal_vector, normal_vector)
            
            registered_spline = np.array(frame_points)  # Copy the spline points

            # Apply the rotation to the entire frame (spline points).
            registered_spline = np.dot(rotation_matrix, registered_spline.T).T  # Apply rotation

            # Perform the translation to center the spline on the centerline point.
            translation_vector = target_centerline_point - registered_spline.mean(axis=0)
            registered_spline += translation_vector

            # Append the registered spline to the list
            saved_registered_splines.append(registered_spline)
            
            # Apply registration to image
            z_level_frames = frame_points[0][2]
            index = round(z_level_frames/z_offset) # reason: order is in reverse (top to down, image interated dwon to top) (due to restructering of grouped frames)
            points_to_register = rotated_points[index]
            points_to_register = np.dot(rotation_matrix, points_to_register.T).T  # Apply rotation
            points_to_register += translation_vector
            registered_points.append(points_to_register)

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

    
        with Image.open(input_file) as im:
            image_diff = OCT_end_frame - OCT_start_frame
            print(image_diff)
            iter = 1
            for page in range(OCT_start_frame, OCT_end_frame, 1):
                ##################################################
                inverse_page = OCT_start_frame + image_diff - iter
                print(inverse_page)
                im.seek(inverse_page)  # Get coorect page here
                iter += 1
                ####################################################
                image = np.array(im.convert('RGB'))  # Convert PIL image to NumPy array
                image_flipped = np.flipud(image)
                height, width, channels = image_flipped.shape
                open_cv_image = image_flipped[crop_bottom: height-crop_top, :, ::-1].copy()  # Crop image at xxx pixels from top
                gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                height, width = gray_image.shape
                
                # Set here, what pages should be shown
                page_to_show = [OCT_registration_frame]

                if inverse_page in page_to_show:
                    index_image = page - OCT_start_frame # reason: order is in reverse (top to down, image interated dwon to top) (due to restructering of grouped frames)
                    registered_points_plot = registered_points[index_image]
                    spline_plot = saved_registered_splines[index_image]

                    # Create a VTK structured grid
                    structured_grid = vtk.vtkStructuredGrid()
                    structured_grid.SetDimensions(gray_image.shape[1], gray_image.shape[0], 1)

                    points = vtk.vtkPoints()
                    colors = vtk.vtkUnsignedCharArray()
                    colors.SetNumberOfComponents(3)  # RGB has three components
                    colors.SetName("Colors")

                    for i in range(height):
                        for j in range(width):
                            points.InsertNextPoint(registered_points_plot[i * width + j])
                            #points.InsertNextPoint(rotated_points[(height - i - 1) * width + (width - j - 1)])
                            rgb_values = open_cv_image[i, j]
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

                    if False:
                        points__ = vtk.vtkPoints()
                        for point_frame in spline_plot:
                            points__.InsertNextPoint(point_frame)

                        polydata = vtk.vtkPolyData()
                        polydata.SetPoints(points__)

                        # Create a sphere for the points
                        sphere_source = vtk.vtkSphereSource()
                        sphere_source.SetRadius(0.05)  # Set the radius of the sphere
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
                        renderer.AddActor(actor_centerline)

                        points__ = vtk.vtkPoints()
                        for point_frame in center_line:
                            points__.InsertNextPoint(point_frame)

                        polydata = vtk.vtkPolyData()
                        polydata.SetPoints(points__)

                        # Create a sphere for the points
                        sphere_source = vtk.vtkSphereSource()
                        sphere_source.SetRadius(0.05)  # Set the radius of the sphere
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
                        actor_centerline.GetProperty().SetColor(1.0, 1.0, 0.0)  # Set color to red

                        # Add the actor to the renderer
                        renderer.AddActor(actor_centerline)

        ################### to be removed plots all splines
            for spline_plot in saved_registered_splines:    
                points__ = vtk.vtkPoints()
                for point_frame in spline_plot:
                    points__.InsertNextPoint(point_frame)

                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points__)

                # Create a sphere for the points
                sphere_source = vtk.vtkSphereSource()
                sphere_source.SetRadius(0.05)  # Set the radius of the sphere
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
                renderer.AddActor(actor_centerline)

                points__ = vtk.vtkPoints()
                for point_frame in center_line:
                    points__.InsertNextPoint(point_frame)

                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points__)

                # Create a sphere for the points
                sphere_source = vtk.vtkSphereSource()
                sphere_source.SetRadius(0.05)  # Set the radius of the sphere
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
                actor_centerline.GetProperty().SetColor(1.0, 1.0, 0.0)  # Set color to red

                # Add the actor to the renderer
                renderer.AddActor(actor_centerline)
    ################### to be removed plots all splines

        
        if False:            
            # Registration point CT
            points_ct_reg = vtk.vtkPoints()

            points_ct_reg.InsertNextPoint(np.array([59.3001, -207.2873, 1733.9572]))

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points_ct_reg)

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
            renderer.AddActor(actor_centerline)
            # CALC
            # Create a VTK actor for the structured grid
            actor_calc = vtk.vtkActor()
            actor_calc.SetMapper(mapper)

            # Add the actor to the renderer
            renderer.AddActor(actor_calc)

            # Load and display the .obj file
            obj_file_path = 'workflow_processed_3d_models/ArCoMo3/Calc_ArCoMo3.obj'
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
            obj_file_path = 'workflow_processed_3d_models/ArCoMo3/Outershell_ArCoMo3.obj'
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
            #renderer.AddActor(obj_actor_outershell)

            # Innershell
            # Create a VTK actor for the structured grid
            actor_innershell = vtk.vtkActor()
            actor_innershell.SetMapper(mapper)

            # Add the actor to the renderer
            renderer.AddActor(actor_innershell)

            # Load and display the .obj file
            obj_file_path = 'workflow_processed_3d_models/ArCoMo3/Innershell_ArCoMo3.obj'
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

            if False:
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
        
        axis = np.cross(vec1, vec2)
        cosA = np.dot(vec1, vec2)
        k = 1.0 / (1.0 + cosA)

        rotation_matrix = np.array([[axis[0] * axis[0] * k + cosA, axis[1] * axis[0] * k - axis[2], axis[2] * axis[0] * k + axis[1]],
                        [axis[0] * axis[1] * k + axis[2], axis[1] * axis[1] * k + cosA, axis[2] * axis[1] * k - axis[0]],
                        [axis[0] * axis[2] * k - axis[1], axis[1] * axis[2] * k + axis[0], axis[2] * axis[2] * k + cosA]])


        return rotation_matrix
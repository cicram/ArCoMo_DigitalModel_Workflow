import cv2
import numpy as np
from PIL import Image
import vtkmodules.all as vtk

OCT_start_frame_pullback = 4
OCT_end_frame_pullback = 5

input_file_pullback = "C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_data/OCT.tif"

def parse_alignement(file_path):
    data = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into three values
            parts = line.strip().split()

            # Ensure there are three values on each line
            if len(parts) == 4:
                # Parse the values as floats and append them to the respective lists
                page, trans_x, trans_y, rotation = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                data.append((page, trans_x, trans_y, rotation))

    data = np.array(data)
    return data

def parse_center_shift(file_path):
    data = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into three values
            parts = line.strip().split()

            # Ensure there are three values on each line
            if len(parts) == 2:
                # Parse the values as floats and append them to the respective lists
                trans_x, trans_y = float(parts[0]), float(parts[1])
                data.append((trans_x, trans_y))

    data = np.array(data)
    return data

def parse_trans_matrix_centerline_reg(file_path):
    data = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into three values
            parts = line.strip().split()

            # Ensure there are three values on each line
            if len(parts) == 3:
                # Parse the values as floats and append them to the respective lists
                trans_x, trans_y, trans_z = float(parts[0]), float(parts[1]), float(parts[2])
                data.append((trans_x, trans_y, trans_z))

    data = np.array(data)
    return data

def parse_rot_matrix_centerline_reg(file_path):
    data = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into three values
            parts = line.strip().split()

            # Ensure there are three values on each line
            if len(parts) == 9:
                # Parse the values as floats and append them to the respective lists
                a, b, c, d, e, f, g, h, i = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                data.append((a, b, c, d, e, f, g, h, i))

    data = np.array(data)
    return data

def parse_rot_angle_co_reg(file_path):
    data = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into three values
            parts = line.strip().split()

            # Ensure there are three values on each line
            if len(parts) == 1:
                # Parse the values as floats and append them to the respective lists
                rotation = float(parts[0])
                data.append((rotation))

    data = np.array(data)
    return data

def parse_centerline(file_path):
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

def parse_OCT_lumen_point_cloud(file_path):
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

#center_line =  parse_centerline("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_data_output/centerline_resmapled.txt")

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
z_dist = 4 * 0.2
z_space = 0.2
data_alignement = parse_alignement("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_data_output/image_translations/alignement_translations.txt")
itr = 0
rot_angle_co_reg = parse_rot_angle_co_reg("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_data_output/image_translations/rotation_angel_registration.txt")
rotation_matrix_co_regist = np.array([[np.cos(rot_angle_co_reg[0]), -np.sin(rot_angle_co_reg[0]), 0.0],
                            [np.sin(rot_angle_co_reg[0]), np.cos(rot_angle_co_reg[0]), 0.0],
                            [0.0, 0.0, 1.0]])
rot_matrix_centerline_reg = parse_rot_matrix_centerline_reg("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_data_output/image_translations/centerline_registration_rotation.txt")
trans_matrix_centerline_reg = parse_trans_matrix_centerline_reg("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_data_output/image_translations/centerline_registration_translation.txt")
trans_matrix_center = parse_trans_matrix_centerline_reg("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_data_output/image_translations/centerline_registration_translation_center.txt")
trans_center_shift = parse_center_shift("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_data_output/image_translations/center_point_shift.txt")
oct_frames = parse_OCT_lumen_point_cloud("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_data_output/output_point_cloud.txt")
point_cloud = parse_trans_matrix_centerline_reg("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_data_output/saved_registered_splines.txt")
# Load images
with Image.open(input_file_pullback) as im_pullback:
    for page_pullback in range(OCT_start_frame_pullback, OCT_end_frame_pullback, 1):
        # centerline_point = center_line[idx_start - page_pullback]
        # Load an image using OpenCV
        im_pullback.seek(page_pullback)  # Move to the current page (frame)
        image_pullback = np.array(im_pullback.convert('RGB'))  # Convert PIL image to NumPy array
        image_pullback = image_pullback[:, :, ::-1].copy()  # Crop image at xxx pixels from top

        gray_image = cv2.cvtColor(image_pullback, cv2.COLOR_BGR2GRAY)

        # Get image dimensions
        height, width = gray_image.shape
        scaling = 98
        # Create a meshgrid for 3D plotting
        x = np.arange(-width / 2, width / 2, 1)
        y = np.arange(-height / 2, height / 2, 1)
        x, y = np.meshgrid(x, y)
        skipping = 1
        x = x[::skipping, ::skipping] / 98
        y = y[::skipping, ::skipping] / 98

        # Create a VTK structured grid
        structured_grid = vtk.vtkStructuredGrid()
        structured_grid.SetDimensions(gray_image.shape[1], gray_image.shape[0], 1)

        # Create VTK points and assign the image data
        points = vtk.vtkPoints()
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetName("GrayScale")

        # Get alignement rotation and translation
        rotation_matrix_alignement = np.array([[np.cos(data_alignement[itr][2]), -np.sin(data_alignement[itr][2]), 0.0],
                                    [np.sin(data_alignement[itr][2]), np.cos(data_alignement[itr][2]), 0.0],
                                    [0.0, 0.0, 1.0]])
        rotation_matrix_reg = np.array([[rot_matrix_centerline_reg[itr][0], rot_matrix_centerline_reg[itr][1], rot_matrix_centerline_reg[itr][2]],
                                    [rot_matrix_centerline_reg[itr][3], rot_matrix_centerline_reg[itr][4], rot_matrix_centerline_reg[itr][5]],
                                    [rot_matrix_centerline_reg[itr][6], rot_matrix_centerline_reg[itr][7], rot_matrix_centerline_reg[itr][8]]])
        translation_vector_alignment = np.array([data_alignement[itr][0]/scaling, data_alignement[itr][1]/scaling, 0.0])
        translation_vector_center_shift = np.array([trans_center_shift[itr][0]/scaling, trans_center_shift[itr][1]/scaling, 0.0])
        translation_vector_reg = np.array([trans_matrix_centerline_reg[itr][0], trans_matrix_centerline_reg[itr][1], trans_matrix_centerline_reg[itr][2]])
        trans_vector_center = np.array([trans_matrix_center[itr][0], trans_matrix_center[itr][1], trans_matrix_center[itr][2]])
        translation_vector = translation_vector_alignment + translation_vector_reg
        rotation_matrix_alig  = np.dot(rotation_matrix_alignement, rotation_matrix_co_regist)
        
        if True:
            for i in range(gray_image.shape[0]):
                for j in range(gray_image.shape[1]):
                    point = np.array([(j - width / 2) / scaling, (i - height / 2) / scaling, z_dist])
                    rotated_point__ = np.dot(rotation_matrix_alignement, point)
                    trans_point__ = rotated_point__ + translation_vector_alignment + translation_vector_center_shift
                    rotated_point_ = np.dot(rotation_matrix_co_regist, trans_point__)
                    rotated_point = np.dot(rotation_matrix_reg, rotated_point_.T).T
                    rotated_translated_point_ = rotated_point + translation_vector_reg
                    rotated_translated_point = np.array([rotated_translated_point_[2], rotated_translated_point_[1], rotated_translated_point_[0]])
                    points.InsertNextPoint(rotated_translated_point_)
                    scalars.InsertNextValue(gray_image[i, j])
        
        z_dist += z_space
        itr += 1
        # Set the points and scalars for the structured grid
        structured_grid.SetPoints(points)
        structured_grid.GetPointData().SetScalars(scalars)

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

        

    points__ = vtk.vtkPoints()

    frame_points = np.array(oct_frames[0.8])
    for point_frame in frame_points:
        rotated_point = np.dot(rotation_matrix_reg, point_frame.T).T
        rotated_translated_point_ = rotated_point + translation_vector_reg
        points__.InsertNextPoint(rotated_translated_point_)
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
    renderer.AddActor(actor_centerline)

# Add centerline
if True:
    #center_line = parse_centerline("C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_data/centerline.txt")
    # Create a VTK points object
    points_centerline = vtk.vtkPoints()
    
    # Insert points from the center line
    for point in point_cloud:
        points_centerline.InsertNextPoint(point[0], point[1], point[2])

    # Create a VTK polydata object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points_centerline)

    # Create a sphere for the points
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(0.1)  # Set the radius of the sphere
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

#STL file
if False:
    # Load and display the .stl file
    stl_file_path = 'C:/Users/JL/Code/ArCoMo_DigitalModel_Workflow/workflow_processed_3d_models/ArCoMo3_shell.stl'  # Set the path to your STL file
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_file_path)

    # Create a VTK mapper for the .stl file
    stl_mapper = vtk.vtkPolyDataMapper()
    stl_mapper.SetInputConnection(stl_reader.GetOutputPort())

    # Create a VTK actor for the .stl file
    stl_actor = vtk.vtkActor()
    stl_actor.SetMapper(stl_mapper)
    stl_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Set color to green

    # Add the actor for the .stl file to the renderer
    renderer.AddActor(stl_actor)


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

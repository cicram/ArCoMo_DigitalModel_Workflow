import vtkmodules.all as vtk
import numpy as np

class point_cloud_visual_editing:
    def __init__(self):
        self.point_cloud1 = None
        self.point_cloud2 = None
        self.actor1 = None
        self.actor2 = None
        self.actor3 = None
        self.actor4 = None
        self.renderer = None
        self.render_window = None
        self.render_window_interactor = None
        self.point_picker = None
        self.selected_points_red = []
        self.selected_points_blue = []
        self.switch_state = 0
        self.switch_text_actor = None
        self.radius_slider = None
        self.radius_text_actor = None
        self.fused_point_cloud = None
        self.previous_points_state = 0
        self.ct_points_state = 1
        self.oct_points_state = 1
        self.pointSize = 3
        
    # Data parsing functions

    def parse_lumen_point_cloud(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                    data.append((px, py, pz))
                else:
                    print(f"Skipping invalid line: {line.strip()}")
        return np.array(data)

    def parse_point_cloud_CT_lumen(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                    data.append((px, py, pz))
                else:
                    print(f"Skipping invalid line: {line.strip()}")
        formatted_data = [' '.join(map(str, inner)) for inner in data]
        result = [list(map(float, inner.split())) for inner in formatted_data]
        return np.array(result)

    # Button creation helper functions

    def CreateButtonOff(self, image):
        white = [155, 155, 155]
        self.CreateImage(image, white, white)

    def CreateButtonBlue(self, image):
        white = [155, 155, 155]
        blue = [0, 0, 255]
        self.CreateImage(image, white, blue)

    def CreateButtonRed(self, image):
        white = [155, 155, 155]
        red = [255, 0, 0]
        self.CreateImage(image, white, red)

    def CreateImage(self, image, color1, color2):
        size = 12
        dims = [size, size, 1]
        lim = size / 3.0
        image.SetDimensions(dims[0], dims[1], dims[2])
        arr = vtk.vtkUnsignedCharArray()
        arr.SetNumberOfComponents(3)
        arr.SetNumberOfTuples(dims[0] * dims[1])
        arr.SetName('scalars')

        for y in range(dims[1]):
            for x in range(dims[0]):
                if x >= lim and x < 2 * lim and y >= lim and y < 2 * lim:
                    arr.SetTuple3(y*size + x, color2[0], color2[1], color2[2])
                else:
                    arr.SetTuple3(y*size + x, color1[0], color1[1], color1[2])

        image.GetPointData().AddArray(arr)
        image.GetPointData().SetActiveScalars('scalars')

    # Renderer setup and point cloud actor creation

    def create_point_cloud_actor(self, points, color):
        vtk_points = vtk.vtkPoints()
        vtk_cells = vtk.vtkCellArray()

        for point in points:
            vtk_id = vtk_points.InsertNextPoint(point)
            vtk_cells.InsertNextCell(1)
            vtk_cells.InsertCellPoint(vtk_id)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetVerts(vtk_cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)

        return actor

    # Helper function to highlight selected points

    def highlight_selected_points(self):
        if self.switch_state == 0:
            mapper = self.actor2.GetMapper()
            selected_points = self.selected_points_blue
        else:
            mapper = self.actor1.GetMapper()
            selected_points = self.selected_points_red

        actor_points = mapper.GetInput()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        for i in range(actor_points.GetNumberOfPoints()):
            if i in selected_points:
                colors.InsertNextTuple([0, 255, 0])
            else:
                if self.switch_state == 0:
                    colors.InsertNextTuple([0, 0, 255])
                else:
                    colors.InsertNextTuple([255, 0, 0])

        actor_points.GetPointData().SetScalars(colors)

    # Update actors based on the switch state

    def update_actors(self):
        if self.switch_state == 0:
            self.renderer.RemoveActor(self.actor2)
            self.actor2 = self.create_point_cloud_actor(self.point_cloud2, [0, 0, 1])
            self.actor2.GetProperty().SetPointSize(self.pointSize)
            self.renderer.AddActor(self.actor2)
        else:
            self.renderer.RemoveActor(self.actor1)
            self.actor1 = self.create_point_cloud_actor(self.point_cloud1, [1, 0, 0])
            self.actor1.GetProperty().SetPointSize(self.pointSize)
            self.renderer.AddActor(self.actor1)
        self.render_window.Render()

    # Select neighbors within a radius of a point

    def select_neighbors_within_radius(self, point_id, radius):
        if self.switch_state == 0:
            point_coords = self.point_cloud2[point_id]
            vtk_points = self.actor2.GetMapper().GetInput().GetPoints()
        else:
            point_coords = self.point_cloud1[point_id]
            vtk_points = self.actor1.GetMapper().GetInput().GetPoints()

        for i in range(vtk_points.GetNumberOfPoints()):
            if i == point_id:
                continue

            neighbor_coords = vtk_points.GetPoint(i)
            distance = np.linalg.norm(np.array(point_coords) - np.array(neighbor_coords))

            if distance <= radius:
                if self.switch_state == 0:
                    self.selected_points_blue.append(i)
                else:
                    self.selected_points_red.append(i)

        self.highlight_selected_points()
        self.render_window.Render()

    # Right button press callback function

    def point_pick_callback(self, obj, event):
        click_pos = self.render_window_interactor.GetEventPosition()
        self.point_picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        selected_actor = self.point_picker.GetActor()

        if selected_actor:
            selected_point_id = self.point_picker.GetPointId()
            if self.switch_state == 0:
                if selected_point_id >= 0:
                    if selected_point_id not in self.selected_points_blue:
                        self.selected_points_blue.append(selected_point_id)
                    
                    self.select_neighbors_within_radius(selected_point_id, self.radius_slider.GetValue())
                    print(f"Selected points: {self.selected_points_blue}")
            else:
                if selected_point_id >= 0:
                    if selected_point_id not in self.selected_points_red:
                        self.selected_points_red.append(selected_point_id)
                    
                    self.select_neighbors_within_radius(selected_point_id, self.radius_slider.GetValue())
                    print(f"Selected points: {self.selected_points_red}")

# ----------------------------------------------------------------------- #
# Blue red switch button #
# ----------------------------------------------------------------------- #

    def update_switch_state(self):
        self.switch_state = (self.switch_state + 1) % 2
        if self.switch_state == 0:
            #self.switch_text_actor.SetInput("Remove from: Blue")
            pass
        else:
            pass
            #self.switch_text_actor.SetInput("Remove from: Red")
        #self.switch_text_actor.Modified()
        print(f"Switched to: {'Blue' if self.switch_state == 0 else 'Red'}")

    def switch_callback(self, obj, event):
        self.update_switch_state()
        # Callback function for the Clear Selection button

    def create_point_cloud_actors(self):
        self.actor1 = self.create_point_cloud_actor(self.point_cloud1, [1, 0, 0])
        self.actor2 = self.create_point_cloud_actor(self.point_cloud2, [0, 0, 1])
        self.actor3 = self.create_point_cloud_actor(self.point_cloud3, [1, 1, 0])
        self.actor4 = self.create_point_cloud_actor(self.point_cloud4, [0, 1, 1])
        
        self.actor1.GetProperty().SetPointSize(3)
        self.actor2.GetProperty().SetPointSize(3)
        self.actor3.GetProperty().SetPointSize(3)
        self.actor4.GetProperty().SetPointSize(3)

    def hide_ct_points(self, obj, event):
        if self.ct_points_state == 1:
            self.renderer.RemoveActor(self.actor2)
            self.renderer.RemoveActor(self.actor4)
        else:
            self.renderer.AddActor(self.actor2)
            self.renderer.AddActor(self.actor4)
        self.ct_points_state = 1 - self.ct_points_state
        self.render_window.Render()

    def hide_oct_points(self, obj, event):
        if self.oct_points_state == 1:
            self.renderer.RemoveActor(self.actor1)
            self.renderer.RemoveActor(self.actor3)
        else:
            self.renderer.AddActor(self.actor1)
            self.renderer.AddActor(self.actor3)
        self.oct_points_state = 1 - self.oct_points_state
        self.render_window.Render()

    def previous_points_callback(self, obj, event):
        if self.previous_points_state == 1:
            self.renderer.RemoveActor(self.actor3)
            self.renderer.RemoveActor(self.actor4)
        else:
            self.renderer.AddActor(self.actor3)
            self.renderer.AddActor(self.actor4)
        self.previous_points_state = 1 - self.previous_points_state
        self.render_window.Render()

    def slider_callback(self, obj, event):
        radius = self.radius_slider.GetValue()
        self.radius_text_actor.SetTextScaleModeToNone()
        self.radius_text_actor.SetText(0, f"Selection radius: {radius}")
        self.radius_text_actor.Modified()

    def remove_points_callback(self, obj, event):
        self.remove_selected_points()
        
        print(f"Removed selected points. Remaining points: {len(self.point_cloud2) if self.switch_state == 0 else len(self.point_cloud1)}")

    def remove_selected_points(self):
        if self.switch_state == 0:
            print(len(self.point_cloud2))
            self.point_cloud2 = np.delete(self.point_cloud2, self.selected_points_blue, axis=0)
            print(len(self.point_cloud2))
            self.update_actors()
        else:
            self.point_cloud1 = np.delete(self.point_cloud1, self.selected_points_red, axis=0)
            self.update_actors()

        self.selected_points_blue = []
        self.selected_points_red = []

    def clear_selection_callback(self, obj, event):
        self.selected_points_blue = []
        self.selected_points_red = []
        self.highlight_selected_points()

    # Function to fuse two point clouds
    def fuse_point_clouds(self):
        # Combine the two point clouds (assumes the point clouds have the same format)
        self.fused_point_cloud = np.concatenate((self.point_cloud1, self.point_cloud2), axis=0)

    # Callback function for the new button to save fused point clouds
    def save_button_callback(self, obj, event):
        # Fuse the two point clouds
        self.fuse_point_clouds()

        self.render_window.Finalize()  # Finalize the render window
        self.render_window_interactor.TerminateApp()
        del self.render_window_interactor
        del self.render_window       

    def run_editor(self, ct_points, oct_lumen, oct_points_previous=None, ct_points_previous=None, oct_calc=None, oct_stent=None):
        self.point_cloud1 = oct_lumen
        self.point_cloud2 = ct_points
        self.actor1 = self.create_point_cloud_actor(self.point_cloud1, [1, 0, 0])
        self.actor2 = self.create_point_cloud_actor(self.point_cloud2, [0, 0, 1])

        self.actor1.GetProperty().SetPointSize(self.pointSize)
        self.actor2.GetProperty().SetPointSize(self.pointSize)

        # previous run points
        self.point_cloud3 = oct_points_previous
        self.point_cloud4 = ct_points_previous
        
        self.create_point_cloud_actors()

        self.renderer = vtk.vtkRenderer()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetFullScreen(True)

        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetInteractorStyle(style)
       
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetWindowName("Point Cloud Editor")

        self.render_window_interactor.SetRenderWindow(self.render_window)
        style.AddObserver("RightButtonPressEvent", lambda obj, event: self.point_pick_callback(obj, event))
        self.render_window_interactor.SetInteractorStyle(style)

        self.renderer.AddActor(self.actor1)
        self.renderer.AddActor(self.actor2)
        self.renderer.SetBackground(1, 1, 1)

        self.point_picker = vtk.vtkPointPicker()
        self.render_window_interactor.SetPicker(self.point_picker)

        self.render_window.Render()

        # Show previous points button
        # Position the new button in the low-left corner
        bds_new = [0] * 6
        sz_new = 50.0
        bds_new[0] = 20  # Adjust the X-coordinate 
        bds_new[1] = bds_new[0] + sz_new
        bds_new[2] = 470  # Adjust the Y-coordinate 
        bds_new[3] = bds_new[2] + sz_new
        bds_new[4] = bds_new[5] = 0.0

        # Create a VTK button representation for the new button
        previous_points_button_rep = vtk.vtkTexturedButtonRepresentation2D()
        button_texture_blue = vtk.vtkImageData()
        button_texture_red = vtk.vtkImageData()
        self.CreateButtonBlue(button_texture_blue)
        self.CreateButtonRed(button_texture_red)

        previous_points_button_rep.SetNumberOfStates(2)
        previous_points_button_rep.SetButtonTexture(0, button_texture_blue)
        previous_points_button_rep.SetButtonTexture(1, button_texture_red)
        # Place the new button widget

        previous_points_button_text_actor = vtk.vtkTextActor()
        previous_points_button_text_actor.GetTextProperty().SetFontSize(20)
        previous_points_button_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        previous_points_button_text_actor.GetTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        previous_points_button_text_actor.SetPosition(bds_new[0], bds_new[3])
        button_label = "Show ground truth points"  # Replace with your desired button label
        previous_points_button_text_actor.SetTextScaleModeToNone()  # Disable text scaling
        previous_points_button_text_actor.SetInput(button_label) 
        self.renderer.AddActor2D(previous_points_button_text_actor)

        previous_points_button_rep.SetPlaceFactor(1)
        previous_points_button_rep.PlaceWidget(bds_new)

        previous_points_button_widget = vtk.vtkButtonWidget()
        previous_points_button_widget.SetInteractor(self.render_window_interactor)
        previous_points_button_widget.SetRepresentation(previous_points_button_rep)

        # Connect the callback function to the new button
        previous_points_button_widget.AddObserver("StateChangedEvent", self.previous_points_callback)
        previous_points_button_widget.On()

        # Show oct points button
        # Position the new button in the low-left corner
        bds_new = [0] * 6
        sz_new = 50.0
        bds_new[0] = 20  # Adjust the X-coordinate 
        bds_new[1] = bds_new[0] + sz_new
        bds_new[2] = 670  # Adjust the Y-coordinate 
        bds_new[3] = bds_new[2] + sz_new
        bds_new[4] = bds_new[5] = 0.0

        # Create a VTK button representation for the new button
        oct_points_button_rep = vtk.vtkTexturedButtonRepresentation2D()
        button_texture_blue = vtk.vtkImageData()
        button_texture_red = vtk.vtkImageData()
        self.CreateButtonBlue(button_texture_blue)
        self.CreateButtonRed(button_texture_red)

        oct_points_button_rep.SetNumberOfStates(2)
        oct_points_button_rep.SetButtonTexture(0, button_texture_blue)
        oct_points_button_rep.SetButtonTexture(1, button_texture_red)
        # Place the new button widget

        oct_points_button_text_actor = vtk.vtkTextActor()
        oct_points_button_text_actor.GetTextProperty().SetFontSize(20)
        oct_points_button_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        oct_points_button_text_actor.GetTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        oct_points_button_text_actor.SetPosition(bds_new[0], bds_new[3])
        button_label = "Hide OCT points"  # Replace with your desired button label
        oct_points_button_text_actor.SetTextScaleModeToNone()  # Disable text scaling
        oct_points_button_text_actor.SetInput(button_label) 
        self.renderer.AddActor2D(oct_points_button_text_actor)

        oct_points_button_rep.SetPlaceFactor(1)
        oct_points_button_rep.PlaceWidget(bds_new)

        oct_points_button_widget = vtk.vtkButtonWidget()
        oct_points_button_widget.SetInteractor(self.render_window_interactor)
        oct_points_button_widget.SetRepresentation(oct_points_button_rep)

        # Connect the callback function to the new button
        oct_points_button_widget.AddObserver("StateChangedEvent", self.hide_oct_points)
        oct_points_button_widget.On()

        # Show ct points button
        # Position the new button in the low-left corner
        bds_new = [0] * 6
        sz_new = 50.0
        bds_new[0] = 20  # Adjust the X-coordinate 
        bds_new[1] = bds_new[0] + sz_new
        bds_new[2] = 570  # Adjust the Y-coordinate 
        bds_new[3] = bds_new[2] + sz_new
        bds_new[4] = bds_new[5] = 0.0

        # Create a VTK button representation for the new button
        ct_points_button_rep = vtk.vtkTexturedButtonRepresentation2D()
        button_texture_blue = vtk.vtkImageData()
        button_texture_red = vtk.vtkImageData()
        self.CreateButtonBlue(button_texture_blue)
        self.CreateButtonRed(button_texture_red)

        ct_points_button_rep.SetNumberOfStates(2)
        ct_points_button_rep.SetButtonTexture(0, button_texture_blue)
        ct_points_button_rep.SetButtonTexture(1, button_texture_red)
        # Place the new button widget

        ct_points_button_text_actor = vtk.vtkTextActor()
        ct_points_button_text_actor.GetTextProperty().SetFontSize(20)
        ct_points_button_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        ct_points_button_text_actor.GetTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        ct_points_button_text_actor.SetPosition(bds_new[0], bds_new[3])
        button_label = "Hide CT points"  # Replace with your desired button label
        ct_points_button_text_actor.SetTextScaleModeToNone()  # Disable text scaling
        ct_points_button_text_actor.SetInput(button_label) 
        self.renderer.AddActor2D(ct_points_button_text_actor)

        ct_points_button_rep.SetPlaceFactor(1)
        ct_points_button_rep.PlaceWidget(bds_new)

        ct_points_button_widget = vtk.vtkButtonWidget()
        ct_points_button_widget.SetInteractor(self.render_window_interactor)
        ct_points_button_widget.SetRepresentation(ct_points_button_rep)

        # Connect the callback function to the new button
        ct_points_button_widget.AddObserver("StateChangedEvent", self.hide_ct_points)
        ct_points_button_widget.On()

        # Create a switch button
        # Position the new button in the low-left corner
        bds_new = [0] * 6
        sz_new = 50.0
        bds_new[0] = 20  # Adjust the X-coordinate 
        bds_new[1] = bds_new[0] + sz_new
        bds_new[2] = 70  # Adjust the Y-coordinate 
        bds_new[3] = bds_new[2] + sz_new
        bds_new[4] = bds_new[5] = 0.0

        # Create a VTK button representation for the new button
        switch_button_rep = vtk.vtkTexturedButtonRepresentation2D()
        button_texture_blue = vtk.vtkImageData()
        button_texture_red = vtk.vtkImageData()
        self.CreateButtonBlue(button_texture_blue)
        self.CreateButtonRed(button_texture_red)

        switch_button_rep.SetNumberOfStates(2)
        switch_button_rep.SetButtonTexture(0, button_texture_blue)
        switch_button_rep.SetButtonTexture(1, button_texture_red)
        # Place the new button widget

        switch_button_text_actor = vtk.vtkTextActor()
        switch_button_text_actor.GetTextProperty().SetFontSize(20)
        switch_button_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        switch_button_text_actor.GetTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        switch_button_text_actor.SetPosition(bds_new[0], bds_new[3])
        button_label = "Switch Button"  # Replace with your desired button label
        switch_button_text_actor.SetTextScaleModeToNone()  # Disable text scaling
        switch_button_text_actor.SetInput(button_label) 
        self.renderer.AddActor2D(switch_button_text_actor)

        switch_button_rep.SetPlaceFactor(1)
        switch_button_rep.PlaceWidget(bds_new)

        switch_button_widget = vtk.vtkButtonWidget()
        switch_button_widget.SetInteractor(self.render_window_interactor)
        switch_button_widget.SetRepresentation(switch_button_rep)

        # Connect the callback function to the new button
        switch_button_widget.AddObserver("StateChangedEvent", self.switch_callback)
        switch_button_widget.On()

        # Radius slider

        self.radius_slider = vtk.vtkSliderRepresentation2D()
        self.radius_slider.SetMinimumValue(0)
        self.radius_slider.SetMaximumValue(10)
        self.radius_slider.SetValue(5)
        self.radius_slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        self.radius_slider.GetPoint1Coordinate().SetValue(0.1, 0.1)
        self.radius_slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        self.radius_slider.GetPoint2Coordinate().SetValue(0.4, 0.1)
        self.radius_slider.SetSliderLength(0.02)
        self.radius_slider.SetSliderWidth(0.03)
        self.radius_slider.SetEndCapLength(0.01)
        self.radius_slider.SetEndCapWidth(0.03)
        self.radius_slider.SetTubeWidth(0.01)
        self.radius_slider.SetLabelFormat("%0.1f")
        self.radius_slider.SetTitleText("Selection radius")
        self.radius_slider.GetSliderProperty().SetColor(0.0, 1.0, 0.0)
        self.radius_slider.GetTitleProperty().SetColor(0, 0, 0)
        self.radius_slider.GetTitleProperty().SetColor(0, 0, 0)
        self.radius_slider.GetSelectedProperty().SetColor(0, 0, 0)
        self.radius_slider.GetTubeProperty().SetColor(0, 0, 0)
        self.radius_slider.GetSelectedProperty().SetColor(0, 0, 0)
        self.radius_slider.GetCapProperty().SetColor(0, 0, 0)
        
        slider_widget = vtk.vtkSliderWidget()
        slider_widget.SetInteractor(self.render_window_interactor)
        slider_widget.SetRepresentation(self.radius_slider)
        slider_widget.KeyPressActivationOff()
        slider_widget.On()

        # Point removal button #

        remove_button = vtk.vtkTexturedButtonRepresentation2D()
        remove_button.SetNumberOfStates(1)
        button_texture_1 = vtk.vtkImageData()

        self.CreateButtonOff(button_texture_1)

        remove_button.SetButtonTexture(0, button_texture_1)

        remove_button_widget = vtk.vtkButtonWidget()
        remove_button_widget.SetInteractor(self.render_window_interactor)
        remove_button_widget.SetRepresentation(remove_button)
        remove_button_widget.AddObserver("StateChangedEvent", self.remove_points_callback)

        self.render_window.Render()


        upperRight = vtk.vtkCoordinate()
        upperRight.SetCoordinateSystemToNormalizedDisplay()
        upperRight.SetValue(1.0, 1.0)

        bds = [0] * 6
        sz = 50.0
        bds[0] = upperRight.GetComputedDisplayValue(self.renderer)[0] - sz - 20
        bds[1] = bds[0] + sz
        bds[2] = upperRight.GetComputedDisplayValue(self.renderer)[1] - sz - 20 
        bds[3] = bds[2] + sz
        bds[4] = bds[5] = 0.0

        remove_button_text_actor = vtk.vtkTextActor()
        remove_button_text_actor.GetTextProperty().SetFontSize(20)
        remove_button_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        remove_button_text_actor.GetTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        remove_button_text_actor.SetPosition(bds[0]-100, bds[3]-75)
        button_label = "Remove Button"  # Replace with your desired button label
        remove_button_text_actor.SetTextScaleModeToNone()  # Disable text scaling
        remove_button_text_actor.SetInput(button_label) 
        self.renderer.AddActor2D(remove_button_text_actor)

        remove_button.SetPlaceFactor(1)
        remove_button.PlaceWidget(bds)
        remove_button_widget.On()

        
        # Clear selection button
        # Create a VTK button widget for clearing the selection
        clear_button_rep = vtk.vtkTexturedButtonRepresentation2D()
        button_texture_off_clear = vtk.vtkImageData()
        self.CreateButtonOff(button_texture_off_clear)

        clear_button_rep.SetNumberOfStates(1)
        clear_button_rep.SetButtonTexture(0, button_texture_off_clear)

        # Position the clear button in the bottom-right corner
        bottomRight = vtk.vtkCoordinate()
        bottomRight.SetCoordinateSystemToNormalizedDisplay()
        bottomRight.SetValue(1.0, 0.0)


        bds_clear = [0] * 6
        bds_clear[0] = bottomRight.GetComputedDisplayValue(self.renderer)[0] - sz - 20  # Adjust the X-coordinate
        bds_clear[1] = bds_clear[0] + sz
        bds_clear[2] = bottomRight.GetComputedDisplayValue(self.renderer)[1] + 70  # Adjust the Y-coordinate
        bds_clear[3] = bds_clear[2] + sz
        bds_clear[4] = bds_clear[5] = 0.0


        # Create the clear button widget
        clear_button_widget = vtk.vtkButtonWidget()
        clear_button_widget.SetInteractor(self.render_window_interactor)
        clear_button_widget.SetRepresentation(clear_button_rep)

        clear_button_text_actor = vtk.vtkTextActor()
        clear_button_text_actor.GetTextProperty().SetFontSize(20)
        clear_button_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        clear_button_text_actor.GetTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        clear_button_text_actor.SetPosition(bds_clear[0]-170, bds_clear[3])
        button_label = "Clear Selection Button"  # Replace with your desired button label
        clear_button_text_actor.SetTextScaleModeToNone()  # Disable text scaling
        clear_button_text_actor.SetInput(button_label) 
        self.renderer.AddActor2D(clear_button_text_actor)

        # Connect the callback function to the clear button
        clear_button_widget.AddObserver("StateChangedEvent", self.clear_selection_callback)
        clear_button_rep.SetPlaceFactor(1)
        clear_button_rep.PlaceWidget(bds_clear)
        clear_button_widget.On()

        # ----------------------------------------------------------------------- #
        # Save point clouds button #
        # ----------------------------------------------------------------------- #

        # Position the new button in the top-left corner
        bds_new = [0] * 6
        sz_new = 50.0
        bds_new[0] = 20  # Adjust the X-coordinate for the top-left corner
        bds_new[1] = bds_new[0] + sz_new
        bds_new[2] = self.render_window.GetSize()[1] - 70  # Adjust the Y-coordinate for the top-left corner
        bds_new[3] = bds_new[2] + sz_new
        bds_new[4] = bds_new[5] = 0.0

        # Create a VTK button representation for the new button
        new_button_rep = vtk.vtkTexturedButtonRepresentation2D()
        button_texture_off_new = vtk.vtkImageData()
        self.CreateButtonOff(button_texture_off_new)

        new_button_rep.SetNumberOfStates(1)
        new_button_rep.SetButtonTexture(0, button_texture_off_new)
        # Place the new button widget
        new_button_rep.SetPlaceFactor(1)
        new_button_rep.PlaceWidget(bds_new)

        save_button_text_actor = vtk.vtkTextActor()
        save_button_text_actor.GetTextProperty().SetFontSize(20)
        save_button_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        save_button_text_actor.GetTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        save_button_text_actor.SetPosition(bds_new[0], bds_new[3]-75)
        button_label = "Save and Close Button"  # Replace with your desired button label
        save_button_text_actor.SetTextScaleModeToNone()  # Disable text scaling
        save_button_text_actor.SetInput(button_label) 
        self.renderer.AddActor2D(save_button_text_actor)


        new_button_widget = vtk.vtkButtonWidget()
        new_button_widget.SetInteractor(self.render_window_interactor)
        new_button_widget.SetRepresentation(new_button_rep)
        # Connect the callback function to the new button
        new_button_widget.AddObserver("StateChangedEvent", self.save_button_callback)
        new_button_widget.On()

        # ----------------------------------------------------------------------- #
        # instruction text #
        # ----------------------------------------------------------------------- #

        instruction_text_actor = vtk.vtkTextActor()
        instruction_text_actor.GetTextProperty().SetFontSize(20)
        instruction_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        instruction_text_actor.GetTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        instruction_text_actor.SetPosition(bds_new[0] + 500, bds_new[3]-30)
        label = "Use Left mouse button to move point cloud, use right mouse button to select points"  # Replace with your desired button label
        instruction_text_actor.SetTextScaleModeToNone()  # Disable text scaling
        instruction_text_actor.SetInput(label) 
        self.renderer.AddActor2D(instruction_text_actor)

        # Create and set the render window for the interactor
        self.render_window.Render()

        # Initialize the interactor
        self.render_window_interactor.Initialize()

        # Start the interaction
        self.render_window.Render()
        self.render_window_interactor.Start()

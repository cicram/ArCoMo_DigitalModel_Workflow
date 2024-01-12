import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from matplotlib.lines import Line2D

class PointCloudSmoothingVisualizer:
    def __init__(self, file_path, registration_point_CT):
        self.pc_centerline = self.parse_point_cloud_centerline(file_path)
        self.colors = ['black'] * len(self.pc_centerline)
        self.current_green_index = 0
        self.registration_point_CT = registration_point_CT
        self.current_blue_index = 1
        self.view_limits = None
        self.move_blue_point = True
        self.fig = plt.figure(figsize=(12, 8))
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.legend_elements = [Line2D([0], [0], marker='o', color='yellow', label='Fitting start point',
                                      markerfacecolor='yellow', markersize=15),
                               Line2D([0], [0], marker='o', color='red', label='Fitting end point',
                                      markerfacecolor='red', markersize=15)]
        self.ax3d.legend(handles=self.legend_elements, loc="upper right")
        self.create_buttons()

    def on_pick(self, event):
        ind = event.ind[0]
        self.colors[self.current_blue_index] = 'black'
        self.colors[self.current_green_index] = 'black'
        self.colors[ind] = 'yellow'
        self.colors[ind + 5] = 'red'
        self.current_green_index = ind
        self.current_blue_index = ind + 5
        self.update_plot()

    def parse_point_cloud_centerline(self, file_path):
        flag_header = False
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    if parts[0]=="[Main" and flag_header:
                        flag_header = False
                    if flag_header:
                        px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                        data.append((px, py, pz))
                    if parts[0]=="Px" and parts[1]=="Py" and parts[2]=="Pz":
                        flag_header = True
                    

        data = np.array(data)
        return data

    def update_plot(self):
        self.ax3d.clear()
        px = self.pc_centerline[:, 0]
        py = self.pc_centerline[:, 1]
        pz = self.pc_centerline[:, 2]
        scatter = self.ax3d.scatter(px, py, pz, c=self.colors, marker='o', picker=True)
        px = self.registration_point_CT[0]
        py = self.registration_point_CT[1]
        pz = self.registration_point_CT[2]
        scatter = self.ax3d.scatter(px, py, pz, marker='o', color="blue")
        self.ax3d.set_xlabel('Px')
        self.ax3d.set_ylabel('Py')
        self.ax3d.set_zlabel('Pz')
        if self.view_limits:
            self.ax3d.set_xlim(self.view_limits[0])
            self.ax3d.set_ylim(self.view_limits[1])
            self.ax3d.set_zlim(self.view_limits[2])
        scatter.set_picker(True)
        self.fig.canvas.draw()

    def move_next_startpoint(self, event):
        if self.current_blue_index < len(self.pc_centerline) - 1:
            self.current_green_index += 1
            self.colors[self.current_green_index] = 'yellow'
            self.colors[self.current_green_index - 1] = 'black'
            if self.current_blue_index <= self.current_green_index:
                self.colors[self.current_blue_index] = 'black'
                self.current_blue_index = self.current_green_index + 1
                self.colors[self.current_blue_index] = 'red'
            elif self.current_blue_index == self.current_green_index:
                self.colors[self.current_blue_index] = 'yellow'
                self.current_blue_index = self.current_green_index + 1
                self.colors[self.current_blue_index] = 'red'
            self.view_limits = (self.ax3d.get_xlim(), self.ax3d.get_ylim(), self.ax3d.get_zlim())
            self.update_plot()

    def move_previous_startpoint(self, event):
        if self.current_blue_index > 0:
            self.current_green_index -= 1
            self.colors[self.current_green_index] = 'yellow'
            self.colors[self.current_green_index + 1] = 'black'
            if self.current_blue_index <= self.current_green_index:
                self.colors[self.current_blue_index] = 'black'
                self.current_blue_index = self.current_green_index + 1
                self.colors[self.current_blue_index] = 'red'
            elif self.current_blue_index == self.current_green_index:
                self.colors[self.current_blue_index] = 'yellow'
                self.current_blue_index = self.current_green_index + 1
                self.colors[self.current_blue_index] = 'red'
            self.view_limits = (self.ax3d.get_xlim(), self.ax3d.get_ylim(), self.ax3d.get_zlim())
            self.update_plot()

    def move_next_endpoint(self, event):
        if self.current_blue_index < len(self.pc_centerline) - 1:
            self.current_blue_index += 1
            self.colors[self.current_blue_index] = 'red'
            self.colors[self.current_blue_index - 1] = 'black'
            if self.current_blue_index <= self.current_green_index:
                self.colors[self.current_blue_index] = 'black'
                self.current_blue_index = self.current_green_index + 1
                self.colors[self.current_blue_index] = 'red'
            elif self.current_blue_index == self.current_green_index:
                self.colors[self.current_blue_index] = 'yellow'
                self.current_blue_index = self.current_green_index + 1
                self.colors[self.current_blue_index] = 'red'
            self.view_limits = (self.ax3d.get_xlim(), self.ax3d.get_ylim(), self.ax3d.get_zlim())
            self.update_plot()

    def move_previous_endpoint(self, event):
        if self.current_blue_index > 0:
            self.current_blue_index -= 1
            self.colors[self.current_blue_index] = 'red'
            self.colors[self.current_blue_index + 1] = 'black'
            if self.current_blue_index <= self.current_green_index:
                self.colors[self.current_blue_index] = 'black'
                self.current_blue_index = self.current_green_index + 1
                self.colors[self.current_blue_index] = 'red'
            elif self.current_blue_index == self.current_green_index:
                self.colors[self.current_blue_index] = 'yellow'
                self.current_blue_index = self.current_green_index + 1
                self.colors[self.current_blue_index] = 'red'
            self.view_limits = (self.ax3d.get_xlim(), self.ax3d.get_ylim(), self.ax3d.get_zlim())
            self.update_plot()

    def fit_spline(self, event):
        point_start = self.pc_centerline[self.current_green_index]
        point_end = self.pc_centerline[self.current_blue_index]
        indices_start = np.where(np.all(self.pc_centerline == point_start, axis=1))[0][0]
        indices_end = np.where(np.all(self.pc_centerline == point_end, axis=1))[0][0]
        fitting_points = []
        number_points = 10
        for i in range(number_points):
            fitting_points.append(self.pc_centerline[indices_start - number_points - 1 + i])
        for i in range(number_points):
            fitting_points.append(self.pc_centerline[indices_end + i])
        indices_start_crop = indices_start - number_points - 1
        indices_stop_crop = indices_end + number_points
        points = fitting_points
        x, y, z = zip(*points)
        tck, u = splprep([x, y, z], s=0, per=0)
        u_new = np.linspace(0, 1, 50)
        spline_points = splev(u_new, tck)
        self.pc_centerline = np.delete(self.pc_centerline, slice(indices_start_crop, indices_stop_crop), axis=0)
        self.colors = np.delete(self.colors, slice(indices_start_crop, indices_stop_crop), axis=0)
        length = len(spline_points[0]) - 1
        for i, p in enumerate(spline_points[0]):
            point = [spline_points[0][length - i], spline_points[1][length - i], spline_points[2][length - i]]
            self.pc_centerline = np.insert(self.pc_centerline, indices_start_crop, point, axis=0)
            self.colors = np.insert(self.colors, indices_start_crop, "black", axis=0)
        self.update_plot()

    def accept_centerline(self, event):
        plt.close(self.fig)

    def create_buttons(self):
        button_next_startpoint = Button(self.fig.add_axes([0.1, 0.02, 0.1, 0.03]), 'Next (start point)')
        button_previous_startpoint = Button(self.fig.add_axes([0.25, 0.02, 0.1, 0.03]), 'Previous (start point)')
        button_next_endpoint = Button(self.fig.add_axes([0.4, 0.02, 0.1, 0.03]), 'Next (end point)')
        button_previous_endpoint = Button(self.fig.add_axes([0.55, 0.02, 0.1, 0.03]), 'Previous (end point)')
        button_fit = Button(self.fig.add_axes([0.7, 0.02, 0.1, 0.03]), 'Fit spline')
        button_accept = Button(self.fig.add_axes([0.85, 0.02, 0.1, 0.03]), 'Accept')

        button_next_startpoint.on_clicked(self.move_next_startpoint)
        button_previous_startpoint.on_clicked(self.move_previous_startpoint)
        button_next_endpoint.on_clicked(self.move_next_endpoint)
        button_previous_endpoint.on_clicked(self.move_previous_endpoint)
        button_fit.on_clicked(self.fit_spline)
        button_accept.on_clicked(self.accept_centerline)

        self.ax3d.mouse_init(rotate_btn=3)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.update_plot()
        plt.show()

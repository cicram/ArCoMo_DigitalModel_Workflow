import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from matplotlib.lines import Line2D

class PointCloudRegistrationPointSelectionVisualizer:
    def __init__(self, resampled_pc_centerline, registration_point_CT):
        self.ct_reg_point = registration_point_CT
        self.pc_centerline = resampled_pc_centerline
        self.colors = ['black'] * len(self.pc_centerline)
        self.selected_point_index_red = 1
        self.selected_point_index_blue = 2
        self.selected_registration_point_CT = (0, 0, 0)
        self.view_limits = None
        self.move_blue_point = False
        self.fig = plt.figure(figsize=(12, 8))
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.legend_elements = [
            Line2D([0], [0], marker='o', color='red', label='Fitting end point', markerfacecolor='red', markersize=15),
            Line2D([0], [0], marker='o', color='blue', label='Second point', markerfacecolor='blue', markersize=15)
        ]
        self.ax3d.legend(handles=self.legend_elements, loc="upper right")
        self.create_buttons()

    def on_pick(self, event):
        ind = event.ind[0]
        if self.move_blue_point:
            self.colors[self.selected_point_index_blue] = 'black'
            self.colors[ind] = 'blue'
            self.selected_point_index_blue = ind
        else:
            self.colors[self.selected_point_index_red] = 'black'
            self.colors[ind] = 'red'
            self.selected_point_index_red = ind
        self.update_plot()

    def update_plot(self):
        self.ax3d.clear()

        # plot registration point from CT
        scatter_ct = self.ax3d.scatter(self.ct_reg_point[0], self.ct_reg_point[1], self.ct_reg_point[2],
                                       c="blue", marker='o', picker=True)

        px = self.pc_centerline[:, 0]
        py = self.pc_centerline[:, 1]
        pz = self.pc_centerline[:, 2]
        scatter = self.ax3d.scatter(px, py, pz, c=self.colors, marker='o', picker=True)

        self.ax3d.set_xlabel('Px')
        self.ax3d.set_ylabel('Py')
        self.ax3d.set_zlabel('Pz')
        if self.view_limits:
            self.ax3d.set_xlim(self.view_limits[0])
            self.ax3d.set_ylim(self.view_limits[1])
            self.ax3d.set_zlim(self.view_limits[2])

        scatter.set_picker(True)
        scatter_ct.set_picker(True)
        self.fig.canvas.draw()

    def move_next_selected(self, event):
        if self.move_blue_point:
            index = self.selected_point_index_blue
            max_index = len(self.pc_centerline) - 1
            if index is None or index < max_index:
                self.selected_point_index_blue = index + 1 if index is not None else 0
                self.colors[self.selected_point_index_blue] = 'blue'
                if index is not None:
                    self.colors[index] = 'black'
                self.view_limits = (self.ax3d.get_xlim(), self.ax3d.get_ylim(), self.ax3d.get_zlim())
                self.update_plot()
        else:
            index = self.selected_point_index_red
            max_index = len(self.pc_centerline) - 1
            if index < max_index:
                self.selected_point_index_red += 1
                self.colors[self.selected_point_index_red] = 'red'
                self.colors[self.selected_point_index_red - 1] = 'black'
                self.view_limits = (self.ax3d.get_xlim(), self.ax3d.get_ylim(), self.ax3d.get_zlim())
                self.update_plot()

    def move_previous_selected(self, event):
        if self.move_blue_point:
            index = self.selected_point_index_blue
            if index is not None and index > 0:
                self.selected_point_index_blue -= 1
                self.colors[self.selected_point_index_blue] = 'blue'
                self.colors[self.selected_point_index_blue + 1] = 'black'
                self.view_limits = (self.ax3d.get_xlim(), self.ax3d.get_ylim(), self.ax3d.get_zlim())
                self.update_plot()
        else:
            index = self.selected_point_index_red
            if index > 0:
                self.selected_point_index_red -= 1
                self.colors[self.selected_point_index_red] = 'red'
                self.colors[self.selected_point_index_red + 1] = 'black'
                self.view_limits = (self.ax3d.get_xlim(), self.ax3d.get_ylim(), self.ax3d.get_zlim())
                self.update_plot()

    def midpoint(self, point1, point2):
        self.selected_registration_point_CT = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, (point1[2] + point2[2]) / 2)

    def accept_centerline(self, event):
        point1 = self.pc_centerline[self.selected_point_index_red]
        point2 = self.pc_centerline[self.selected_point_index_blue]
        self.midpoint(point1, point2)
        plt.close(self.fig)

    def toggle_selection(self, event):
        if self.move_blue_point:
            self.move_blue_point = False
        else:
            self.move_blue_point = True

    def create_buttons(self):
        button_next_startpoint = Button(self.fig.add_axes([0.1, 0.02, 0.1, 0.03]), 'Next point')
        button_previous_startpoint = Button(self.fig.add_axes([0.25, 0.02, 0.1, 0.03]), 'Previous point')
        button_accept = Button(self.fig.add_axes([0.7, 0.02, 0.1, 0.03]), 'Accept')
        button_toggle = Button(self.fig.add_axes([0.85, 0.02, 0.1, 0.03]), 'Toggle Selection')

        button_next_startpoint.on_clicked(self.move_next_selected)
        button_previous_startpoint.on_clicked(self.move_previous_selected)
        button_accept.on_clicked(self.accept_centerline)
        button_toggle.on_clicked(self.toggle_selection)

        self.ax3d.mouse_init(rotate_btn=3)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.update_plot()
        plt.show()

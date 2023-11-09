import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ThreeDPlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Plot with Points")

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.points = []  # Store the points (x, y, z, color)

        self.start_button = ttk.Button(self.root, text="Start Point", command=self.start_point)
        self.start_button.pack()

        self.end_button = ttk.Button(self.root, text="End Point", command=self.end_point)
        self.end_button.pack()


        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()
        self.fig.canvas.callbacks.connect('button_press_event', self.on_click)
        self.add_example_points()  # Add example points


    def start_point(self):
        self.current_color = 'yellow'

    def end_point(self):
        self.current_color = 'blue'

    def on_click(self, event):
        if event.inaxes == self.ax:
            x, y, z = event.xdata, event.ydata, event.zdata
            self.plot_point(x, y, z, self.current_color)

    def plot_point(self, x, y, z, color):
        self.ax.scatter(x, y, z, color=color, s=30)
        self.points.append((x, y, z, color))
        self.canvas.draw()

    def add_example_points(self):
        # Add some example points
        example_points = [
            (1, 2, 3, 'black'),
            (4, 5, 6, 'black'),
            (7, 8, 9, 'black'),
            (7, 8, 12, 'black'),
            (7, 8, 14, 'black'),

        ]
        for x, y, z, color in example_points:
            self.plot_point(x, y, z, color)

if __name__ == '__main__':
    root = tk.Tk()
    app = ThreeDPlotter(root)
    root.mainloop()

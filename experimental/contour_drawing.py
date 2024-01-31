import tkinter as tk
from tkinter import Canvas

class ContourDrawer:
    def __init__(self, root, image_path):
        self.root = root
        self.canvas = Canvas(root, width=500, height=500)
        self.canvas.pack()
        self.image_path = image_path
        self.image = tk.PhotoImage(file=image_path)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        self.drawing = False
        self.contour_points = []

        self.canvas.bind('<ButtonPress-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_contour)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)

    def start_drawing(self, event):
        self.drawing = True
        self.contour_points = [(event.x, event.y)]

    def draw_contour(self, event):
        if self.drawing:
            self.contour_points.append((event.x, event.y))
            self.canvas.create_line(self.contour_points[-2], self.contour_points[-1], fill='red', width=2)

    def stop_drawing(self, event):
        if self.drawing:
            self.drawing = False
            # Your logic for the drawn contour (e.g., save the points or process the region)
            print('Contour points:', self.contour_points)

if __name__ == "__main__":
    root = tk.Tk()
    image_path = 'experimental/Test_live_wire.JPG'  # Replace with the path to your image
    drawer = ContourDrawer(root, image_path)
    root.mainloop()
import open3d as o3d
import numpy as np
from enum import Enum

class PointCloudEditor:
    class Mode(Enum):
        SELECT = 0
        REMOVE = 1
        CROP = 2

    def __init__(self, first_cloud_path, second_cloud_path):
        # Load the first point cloud and color it red
        self.pcd1 = o3d.io.read_point_cloud(first_cloud_path)
        self.pcd1.paint_uniform_color([1, 0, 0])  # Red

        # Load the second point cloud and color it blue
        self.pcd2 = o3d.io.read_point_cloud(second_cloud_path)
        self.pcd2.paint_uniform_color([0, 0, 1])  # Blue

        # Create a visualization window
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        # Add both point clouds to the visualization
        self.vis.add_geometry(self.pcd1)
        self.vis.add_geometry(self.pcd2)

        # Initialize selected indices for the blue point cloud
        self.selected_indices = []

        # Initialize cropping box
        self.cropping_box = None

        # Initialize editing mode
        self.mode = self.Mode.SELECT

        # Register keyboard callback
        self.vis.register_key_callback(ord('B'), self.toggle_select_mode)
        self.vis.register_key_callback(ord('R'), self.toggle_remove_mode)
        self.vis.register_key_callback(ord('C'), self.toggle_crop_mode)
        self.vis.register_key_callback(ord('S'), self.save_point_clouds)

    def toggle_select_mode(self, action):
            self.mode = self.Mode.SELECT
            print("Select mode: ON")

    def toggle_remove_mode(self, action):
            self.mode = self.Mode.REMOVE
            print("Remove mode: ON")

    def toggle_crop_mode(self, action):
            self.mode = self.Mode.CROP
            print("Crop mode: ON")

    def key_callback(self, vis, key, scancode, action, mods):
        if self.mode == self.Mode.SELECT:
            # Handle select mode
            self.vis.draw_geometries_with_editing(self.pcd1)

            if key == ord("Q"):
                # Button to finish selecting points from the blue point cloud
                print("Selected points indices:", self.selected_indices)
                vis.update_geometry(self.pcd2)
                vis.poll_events()

        elif self.mode == self.Mode.REMOVE:
            # Handle remove mode
            if key == ord("Q"):
                # Button to finish selecting points from the blue point cloud
                print("Removing selected points from the blue point cloud...")
                if self.selected_indices:
                    self.pcd2.points = o3d.utility.Vector3dVector(
                        [p for i, p in enumerate(np.asarray(self.pcd2.points)) if i not in self.selected_indices]
                    )
                    self.selected_indices = []
                    vis.update_geometry(self.pcd2)
                    vis.poll_events()

        elif self.mode == self.Mode.CROP:
            # Handle crop mode
            if key == ord("Q"):
                # Button to finish cropping
                if self.cropping_box is not None:
                    print("Cropping selected region from both point clouds...")
                    cropped_pcd1 = self.pcd1.crop(self.cropping_box)
                    cropped_pcd2 = self.pcd2.crop(self.cropping_box)
                    self.pcd1 = cropped_pcd1
                    self.pcd2 = cropped_pcd2
                    vis.update_geometry(self.pcd1)
                    vis.update_geometry(self.pcd2)
                    self.cropping_box = None
                    vis.poll_events()

    def save_point_clouds(self, action):
        if action == o3d.visualization.ActionEvent.Press:
            # Save both point clouds
            o3d.io.write_point_cloud("first_cloud_saved.pcd", self.pcd1)
            o3d.io.write_point_cloud("second_cloud_saved.pcd", self.pcd2)
            print("Point clouds saved.")

    def run(self):
        print("Press 'Q' to quit.")
        print("Press 'B' to begin selecting points from the blue point cloud.")
        print("Press 'R' to remove selected points from the blue point cloud.")
        print("Press 'C' to crop the selected region from both point clouds.")
        print("Press 'S' to save both point clouds.")
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    editor = PointCloudEditor("original_cropped.pcd", "original_cropped2.pcd")
    editor.run()

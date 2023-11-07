import open3d as o3d
import numpy as np


def pick_points(cloud):
    print("Press [shift + left click] to pick a point")
    print("Press [shift + right click] to undo point picking")
    print("Press 'Q' to close the window after picking points")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(cloud)
    vis.run()  # User picks points
    vis.destroy_window()
    return vis.get_picked_points()  # A list of indices of picked points


if __name__ == "__main__":
    if False:
        print("Demo for manual geometry cropping")
        print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
        print("2) Press 'K' to lock screen and to switch to selection mode")
        print("3) Drag for rectangle selection,")
        print("   or use ctrl + left click for polygon selection")
        print("4) Press 'C' to get a selected geometry and to save it")
        print("5) Press 'F' to switch to freeview mode")
        pcd = o3d.io.read_point_cloud("C:/Users/JL/Code/workflow/original_cropped_old.pcd")
        o3d.visualization.draw_geometries_with_editing([pcd])


    if True:
        # Load the point cloud
        cloud = o3d.io.read_triangle_mesh("C:/Users/JL/Code/workflow/mesh_CT_model.stl")
        cloud = cloud.sample_points_poisson_disk(number_of_points=10000)
        search_radius = 0.5
        # Pick a point in interactive visualization
        point_indices = pick_points(cloud)
        print(point_indices)
        # Create a mask to keep the points that were not picked
        mask = np.ones(len(cloud.points), dtype=bool)
        mask[point_indices] = False

        # Create a new point cloud with the cropped points removed
        cropped_cloud = o3d.geometry.PointCloud()
        cropped_cloud.points = o3d.utility.Vector3dVector(np.asarray(cloud.points)[mask])

        # Save the cropped point cloud
        save_to_file = "original_cropped2.pcd"
        o3d.io.write_point_cloud(save_to_file, cropped_cloud)
        print("Saved the cropped point cloud to " + save_to_file)

        # Visualize the cropped point cloud
        o3d.visualization.draw_geometries([cropped_cloud])

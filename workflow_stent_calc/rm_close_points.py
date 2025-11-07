import numpy as np
from scipy.spatial import KDTree


def rm_close_points(point_cloud1, point_cloud2, r=0.35):
    """Remove points from point_cloud1 (CT) that are close to point_cloud2 (OCT). To reduce manual remove using
    workflow_stent_calc_visual_pointcloud_editing_VTC_point.py

    Args
        point_cloud1 (np.array): CT based point cloud
        point_cloud2 (np.array): OCT based point cloud"""

    tree2 = KDTree(point_cloud2)
    close_indices_in_cloud2 = tree2.query_ball_point(point_cloud1, r=r)

    # Get indices in point_cloud1 that have at least one neighbor in point_cloud2
    close_indices_in_cloud1 = np.where([len(indices) > 0 for indices in close_indices_in_cloud2])[0]

    # Create mask that keeps only the far points
    mask = np.ones(len(point_cloud1), dtype=bool)
    mask[close_indices_in_cloud1] = False

    reduced_point_cloud = point_cloud1[mask]

    return reduced_point_cloud



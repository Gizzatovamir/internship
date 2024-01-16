import open3d as o3d
import numpy as np


def numpy_cloud_to_open3d(xyz: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def visualize_cloud(xyz: np.ndarray, colors=None):
    pcd = numpy_cloud_to_open3d(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


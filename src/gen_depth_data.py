import os
import pathlib
import sqlite3
from read_log import read_one_msg
from utils.read_data import get_parser
from query import query_get_one_msg
import utils.constants as constants
from typing import Dict, List
import pandas as pd


import numpy as np

velodatatype = np.dtype(
    {"x": ("<u2", 0), "y": ("<u2", 2), "z": ("<u2", 4), "i": ("u1", 6), "l": ("u1", 7)}
)
velodatasize = 8


def range_projection(
    current_vertex,
    fov_up=10.67,
    fov_down=-30.67,
    proj_H=32,
    proj_W=900,
    max_range=80,
    cut_z=False,
    low=0.1,
    high=6,
):
    """Project a pointcloud into a spherical projection, range image.
    Args:
    current_vertex: raw point clouds
    Returns:
    proj_range: projected range image with depth, each pixel contains the corresponding depth
    proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
    proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)

    if cut_z:
        z = current_vertex[:, 2]
        current_vertex = current_vertex[
            (depth > 0) & (depth < max_range) & (z < high) & (z > low)
        ]  # get rid of [0, 0, 0] points
        depth = depth[(depth > 0) & (depth < max_range) & (z < high) & (z > low)]
    else:
        current_vertex = current_vertex[
            (depth > 0) & (depth < max_range)
        ]  # get rid of [0, 0, 0] points
        depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full(
        (proj_H, proj_W), -1, dtype=np.float32
    )  # [H,W] range (-1 is no data)
    proj_vertex = np.full(
        (proj_H, proj_W, 4), -1, dtype=np.float32
    )  # [H,W] index (-1 is no data)
    proj_idx = np.full(
        (proj_H, proj_W), -1, dtype=np.int32
    )  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array(
        [scan_x, scan_y, scan_z, np.ones(len(scan_x))]
    ).T
    proj_idx[proj_y, proj_x] = indices

    return proj_range, proj_vertex, proj_idx


def data2xyzi(data, flip=True):
    xyzil = data.view(velodatatype)
    xyz = np.hstack([xyzil[axis].reshape([-1, 1]) for axis in ["x", "y", "z"]])
    xyz = xyz * 0.005 - 100.0

    if flip:
        R = np.eye(3)
        R[2, 2] = -1
        # xyz = xyz @ R
        xyz = np.matmul(xyz, R)

    return xyz, xyzil["i"]


def get_velo(velofile):
    return data2xyzi(np.fromfile(velofile))


def gen_depth_data(
    _db: sqlite3.Cursor, dst_folder: pathlib.Path, query_request: str, normalize=False
):
    """Generate projected range data in the shape of (64, 900, 1).
    The input raw data are in the shape of (Num_points, 3).
    """
    if dst_folder.exists():
        print("generating depth data in: ", dst_folder)
    else:
        print("creating new depth folder: ", dst_folder)
        os.mkdir(dst_folder)

    # load LiDAR scan files
    for timestamp, current_vertex in _db.execute(query_request).fetchall():
        print(type(current_vertex))
        print(type(timestamp))
        lidar_data = read_one_msg(current_vertex, timestamp)
        fov_up = 30.67
        fov_down = -10.67
        proj_H = 32
        proj_W = 900
        lowest = 0.1
        highest = 6
        proj_range, proj_vertex, _ = range_projection(
            lidar_data.point_cloud,
            fov_up=fov_up,
            fov_down=fov_down,
            proj_H=proj_H,
            proj_W=proj_W,
            max_range=80,
            cut_z=False,
            low=lowest,
            high=highest,
        )

        # normalize the image
        if normalize:
            proj_range = proj_range / np.max(proj_range)

        # generate the destination path
        dst_path = os.path.join(dst_folder, str(timestamp).zfill(6))

        np.save(dst_path, proj_range)
        # depths.append(proj_range)
        print("finished generating depth data at: ", dst_path)


if __name__ == "__main__":
    info_dict: List[Dict[str, str]] = [
        {"help": "path to db", "type": str},
        {"help": "path to result", "type": str},
    ]
    parser = get_parser(["--path", "--dst_path"], info_dict)
    args = parser.parse_args()
    dir_path = pathlib.Path(str(args.path))
    db = sqlite3.connect(dir_path.as_posix())
    cursor = db.cursor()
    gen_depth_data(
        db.cursor(),
        pathlib.Path(args.dst_path),
        query_get_one_msg(constants.LIDAR_POINTS_TOPIC_ID),
    )

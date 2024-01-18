import src.query
import sqlite3
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import base64
import matplotlib.pyplot as plt
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from doedu.navedu.localization_and_mapping.lam.components.data import Lidar
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from typing import List, Tuple, Dict
from time import time
from src.utils import visualize_cloud

LIDAR_TOPIC = "/lidar_points"
GNSS_LIDAR_TOPIC = "/gnss_lidar/wgs2utm/utm_coords"
GNSS_PAYLOAD_TOPIC = "/gnss_payload/wgs2utm/utm_coords"
NUMPY_DTYPE = np.float64

_POINT_FIELDS_AND_NUMPY_DTYPES_PAIRS = [
    (PointField.INT8, np.dtype("int8")),
    (PointField.UINT8, np.dtype("uint8")),
    (PointField.INT16, np.dtype("int16")),
    (PointField.UINT16, np.dtype("uint16")),
    (PointField.INT32, np.dtype("int32")),
    (PointField.UINT32, np.dtype("uint32")),
    (PointField.FLOAT32, np.dtype("float32")),
    (PointField.FLOAT64, np.dtype("float64")),
]

_POINT_FIELD_TO_NUMPY_DTYPE = dict(_POINT_FIELDS_AND_NUMPY_DTYPES_PAIRS)

_NUMPY_DTYPE_TO_POINT_FIELD = {
    numpy_dtype: point_field_type
    for point_field_type, numpy_dtype in _POINT_FIELDS_AND_NUMPY_DTYPES_PAIRS
}

# sizes (in bytes) of PointField types
_POINT_FIELDS_SIZES = {
    PointField.INT8: 1,
    PointField.UINT8: 1,
    PointField.INT16: 2,
    PointField.UINT16: 2,
    PointField.INT32: 4,
    PointField.UINT32: 4,
    PointField.FLOAT32: 4,
    PointField.FLOAT64: 8,
}

_DUMMY_FIELD_PREFIX = "__"


def _point_fields_to_numpy_dtypes(fields, point_step):
    """
    Convert a list of PointFields to a numpy record datatype.
    """
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(("%s%d" % (_DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = _POINT_FIELD_TO_NUMPY_DTYPE[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += _POINT_FIELDS_SIZES[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(("%s%d" % (_DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list


def ros_point_cloud_to_numpy_point_cloud(
    point_cloud: PointCloud2, field_names: List[str] = None
) -> np.array:
    if field_names is None:
        field_names = ["x", "y", "z", "intensity"]

    dtype_list = _point_fields_to_numpy_dtypes(
        point_cloud.fields, point_cloud.point_step
    )
    point_cloud = np.frombuffer(point_cloud.data, dtype_list)
    point_cloud = structured_to_unstructured(point_cloud[field_names]).copy()
    point_cloud = point_cloud.astype(np.float32)

    return point_cloud


def ros_point_cloud_to_lidar(
    point_cloud: PointCloud2, timestamp: float = None
) -> Lidar:
    numpy_point_cloud = ros_point_cloud_to_numpy_point_cloud(point_cloud).astype(
        np.float32
    )[:, :3]
    if timestamp is None:
        timestamp = time()
    return Lidar.build(point_cloud=numpy_point_cloud, timestamp=timestamp)


def read_one_msg(binary_data, timestamp) -> Lidar:
    msg = deserialize_cdr(binary_data, "sensor_msgs/msg/PointCloud2")
    lidar_data = ros_point_cloud_to_lidar(msg, timestamp=timestamp)
    # visualize_cloud(lidar_data.point_cloud)
    return lidar_data

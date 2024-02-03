from __future__ import annotations
import numpy as np
from time import time
from pyquaternion import Quaternion
from utils.pose import Pose


class Lidar:
    def __init__(self, point_cloud: np.ndarray, timestamp: float) -> None:
        self._point_cloud = point_cloud
        self._timestamp = timestamp
        self._feature_cloud = None

    @property
    def point_cloud(self) -> np.ndarray:
        return self._point_cloud

    @point_cloud.setter
    def point_cloud(self, point_cloud: np.ndarray):
        self._point_cloud = point_cloud

    @property
    def feature_cloud(self) -> np.ndarray:
        if self._feature_cloud is None:
            return self._point_cloud
        return self._feature_cloud

    @feature_cloud.setter
    def feature_cloud(self, feature_cloud: np.ndarray):
        self._feature_cloud = feature_cloud

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def copy(self) -> Lidar:
        lidar_copy = Lidar(
            point_cloud=self._point_cloud.copy(),
            timestamp=self._timestamp,
        )
        if self._feature_cloud is not None:
            lidar_copy.feature_cloud = self._feature_cloud.copy()
        return lidar_copy

    @classmethod
    def build(
            cls,
            point_cloud: np.ndarray = None,
            timestamp: float = None
    ) -> Lidar:
        point_cloud = np.zeros((0, 3)) if point_cloud is None else point_cloud
        timestamp = time() if timestamp is None else timestamp
        return cls(point_cloud, timestamp=timestamp)


class GNSS:
    def __init__(
        self, pose: Pose, timestamp: float, has_orientation: bool
    ):
        self._pose = pose
        self._timestamp = timestamp
        self._has_orientation = has_orientation

    @property
    def pose(self) -> Pose:
        return self._pose

    @pose.setter
    def pose(self, pose: Pose) -> None:
        self._pose = pose

    @property
    def has_orientation(self) -> bool:
        return self._has_orientation

    @has_orientation.setter
    def has_orientation(self, flag: bool) -> None:
        self._has_orientation = flag

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def copy(self) -> GNSS:
        gnss_copy = GNSS(
            pose=self._pose.copy(),
            timestamp=self._timestamp,
            has_orientation=self._has_orientation
        )
        return gnss_copy

    @classmethod
    def build(
            cls,
            pose: Pose = None,
            timestamp: float = None,
            has_orientation: bool = False
    ) -> GNSS:
        pose = Pose() if pose is None else pose
        timestamp = time() if timestamp is None else timestamp
        return cls(pose, timestamp=timestamp, has_orientation=has_orientation)


class WheelOdometry:
    pass


class Imu:
    def __init__(
            self,
            orientation: Quaternion,
            angular_velocity: np.ndarray,
            linear_acceleration: np.ndarray,
            orientation_covariance: np.ndarray,
            angular_velocity_covariance: np.ndarray,
            linear_acceleration_covariance: np.ndarray,
    ):
        self._orientation = orientation
        self._angular_velocity = angular_velocity
        self._linear_acceleration = linear_acceleration
        self._orientation_covariance = orientation_covariance
        self._angular_velocity_covariance = angular_velocity_covariance
        self._linear_acceleration_covariance = linear_acceleration_covariance

    @property
    def linear_acceleration(self) -> np.ndarray:
        return self._linear_acceleration

    @property
    def angular_velocity(self) -> np.ndarray:
        return self._angular_velocity

    def rotate(self, rotation_matrix: np.ndarray):
        self._orientation = self._orientation * Quaternion(
            matrix=rotation_matrix)
        self._angular_velocity = np.matmul(
            rotation_matrix, self._angular_velocity)
        self._linear_acceleration = np.matmul(
            rotation_matrix, self._linear_acceleration)

    def copy(self):
        return Imu(
            orientation=Quaternion(
                x=self._orientation.x, y=self._orientation.y,
                z=self._orientation.z, w=self._orientation.w),
            angular_velocity=self.angular_velocity.copy(),
            linear_acceleration=self.linear_acceleration.copy(),
            orientation_covariance=self._orientation_covariance.copy(),
            angular_velocity_covariance=self._angular_velocity_covariance.copy(),
            linear_acceleration_covariance=self._linear_acceleration_covariance.copy())

    @classmethod
    def build(
            cls,
            orientation: Quaternion,
            angular_velocity: np.ndarray,
            linear_acceleration: np.ndarray,
            orientation_covariance: np.ndarray = np.eye(3),
            angular_velocity_covariance: np.ndarray = np.eye(3),
            linear_acceleration_covariance: np.ndarray = np.eye(3),
    ) -> Imu:
        return cls(
            orientation=orientation,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            orientation_covariance=orientation_covariance,
            angular_velocity_covariance=angular_velocity_covariance,
            linear_acceleration_covariance=linear_acceleration_covariance)


class StereoImage:
    pass

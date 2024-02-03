from __future__ import annotations
import torch
import numpy as np
# import g2o
import gtsam
from pyquaternion import Quaternion


DTYPE = torch.float64


class Pose:
    def __init__(
        self,
        rotation_matrix: torch.Tensor = torch.eye(3),
        translation: torch.Tensor = torch.zeros((3,)),
        covariance: torch.Tensor = torch.eye(6)
    ):
        self._rotation_matrix: torch.Tensor = rotation_matrix.to(dtype=DTYPE)
        self._translation: torch.Tensor = translation.to(dtype=DTYPE)
        self._covariance: torch.Tensor = covariance.to(dtype=DTYPE)

    @property
    def translation(self) -> torch.Tensor:
        return self._translation.to(dtype=DTYPE)

    @translation.setter
    def translation(self, translation: torch.Tensor):
        assert translation.shape == (3, 1) or translation.shape == (3,), \
            f'translation.shape == {translation.shape}'
        self._translation = translation

    @property
    def rotation_matrix(self) -> torch.Tensor:
        return self._rotation_matrix.to(dtype=DTYPE)

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix: torch.Tensor):
        assert rotation_matrix.shape == (3, 3), \
            f'rotation_matrix.shape == {rotation_matrix.shape}'
        self._rotation_matrix = rotation_matrix

    @property
    def orientation(self) -> Quaternion:
        return Quaternion(
            matrix=self._rotation_matrix.numpy(), rtol=1e-02, atol=1e-02
        )

    @orientation.setter
    def orientation(self, quaternion: Quaternion):
        self._rotation_matrix = torch.from_numpy(quaternion.rotation_matrix)

    @property
    def transformation_matrix(self) -> torch.Tensor:
        """
        Column-major transformation matrix
        """
        transformation = torch.eye(4, dtype=DTYPE)
        transformation[:3, :3] = self._rotation_matrix
        transformation[:3, 3] = self._translation
        return transformation

    @transformation_matrix.setter
    def transformation_matrix(self, transformation: torch.Tensor):
        assert transformation.shape == (4, 4), \
            f'rotation_matrix.shape == {transformation.shape}'
        self._rotation_matrix = transformation[:3, :3]
        self._translation = transformation[:3, 3]

    @property
    def covariance(self) -> torch.Tensor:
        return self._covariance

    @covariance.setter
    def covariance(self, covariance: torch.Tensor):
        assert covariance.shape == (6, 6), \
            f'rotation_matrix.shape == {covariance.shape}'
        self._covariance = covariance

    # @property
    # def g2o(self) -> g2o.Isometry3d:
    #     return g2o.Isometry3d(self.transformation_matrix.numpy())

    @property
    def gtsam(self) -> gtsam.Pose3:
        translation = self._translation.numpy().tolist()
        quaternion = self.orientation
        return gtsam.Pose3(
            r=gtsam.Rot3.Quaternion(
                w=quaternion.w,
                x=quaternion.x,
                y=quaternion.y,
                z=quaternion.z
            ),
            t=gtsam.Point3(*translation)
        )

    def copy(self) -> Pose:
        return Pose(
            rotation_matrix=self._rotation_matrix.clone(),
            translation=self._translation.clone(),
            covariance=self._covariance.clone()
        )

    def inverse(self) -> Pose:
        transformation = self.transformation_matrix
        transformation = torch.linalg.inv(transformation)
        return Pose(transformation[:3, :3], transformation[:3, 3])

    def __mul__(self, other: Pose) -> Pose:
        current_transformation_matrix = self.transformation_matrix
        other_transformation_matrix = other.transformation_matrix

        # T_current * T_other
        new_transformation_matrix = torch.matmul(
            current_transformation_matrix, other_transformation_matrix)

        current_covariance = self.covariance
        other_covariance = other.covariance
        new_covariance = torch.matmul(
            current_covariance, other_covariance)

        return Pose(
            rotation_matrix=new_transformation_matrix[:3, :3],
            translation=new_transformation_matrix[:3, 3],
            covariance=new_covariance
        )


def get_translation_distance(pose_from: Pose, pose_to: Pose) -> float:
    translation_from = pose_from.translation
    translation_to = pose_to.translation
    return torch.sqrt(
        torch.sum(torch.pow(translation_from - translation_to, 2))
    ).numpy()


def get_yaw_pitch_roll_distance(pose_from: Pose, pose_to: Pose):
    ypr_from = np.array(pose_from.orientation.yaw_pitch_roll)
    ypr_to = np.array(pose_to.orientation.yaw_pitch_roll)
    return np.abs(ypr_from - ypr_to)


def gtsam_pose3_to_pose(gtsam_pose: gtsam.Pose3) -> Pose:
    pose = Pose()
    pose.transformation_matrix = torch.from_numpy(
        gtsam_pose.matrix().astype(np.float64))
    return pose


def get_between_pose(from_pose: Pose, to_pose: Pose) -> Pose:
    # between_pose_gtsam = to_pose.gtsam.between(from_pose.gtsam)
    # between_pose = gtsam_pose3_to_pose(between_pose_gtsam)
    # return between_pose
    return to_pose.inverse() * from_pose


def transform(pose: Pose, xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz: [N, 3]
    """
    assert len(xyz.shape) == 2, f'{len(xyz.shape)} != 2'
    assert xyz.shape[1] == 3, f'{xyz.shape[1]} != 3'

    if xyz.dtype != DTYPE:
        xyz = xyz.to(DTYPE)

    xyz_homogen = torch.ones((xyz.shape[0], 4), dtype=xyz.dtype)
    xyz_homogen[:, :3] = xyz
    xyz_transformed = torch.matmul(
        pose.transformation_matrix, xyz_homogen.t()
    ).t()[:, :3]

    return xyz_transformed


def find_gnss_relative_orientation(
    base_pose: Pose, auxiliary_pose: Pose
) -> Quaternion:
    base_translation = base_pose.translation
    auxiliary_translation = auxiliary_pose.translation
    vector = base_translation - auxiliary_translation
    vector /= np.linalg.norm(vector)
    yaw = np.arctan2(vector[1], vector[0])
    quat = Quaternion(axis=[0, 0, 1], angle=yaw)
    return quat


import open3d as o3d
import numpy as np
import torch


def numpy_cloud_to_open3d(xyz: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def visualize_cloud(xyz: np.ndarray, colors=None):
    pcd = numpy_cloud_to_open3d(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


def read_one_need_descriptor_from_seq_ft(file_num, descriptors, seq_len, poses=None):
    read_complete_flag = True
    descriptors_seq = torch.zeros((1, seq_len, 256)).type(torch.FloatTensor).cuda()
    for i in np.arange(
        int(file_num) - (seq_len // 2), int(file_num) - (seq_len // 2) + seq_len
    ):  # length can be changed !!!!!!!
        if i < 0 or i >= descriptors.shape[0]:
            read_complete_flag = False
            for m in np.arange(
                int(file_num) - (seq_len // 2), int(file_num) - (seq_len // 2) + seq_len
            ):
                descriptors_seq[0, int(m - int(file_num) + (seq_len // 2)), :] = (
                    torch.from_numpy(descriptors[int(file_num), :])
                    .type(torch.FloatTensor)
                    .cuda()
                )
            return descriptors_seq, read_complete_flag
        descriptor_tensor = (
            torch.from_numpy(descriptors[i, :]).type(torch.FloatTensor).cuda()
        )
        descriptors_seq[
            0, int(i - int(file_num) + (seq_len // 2)), :
        ] = descriptor_tensor

    return descriptors_seq, read_complete_flag


def read_one_need_descriptor_from_seq_ft(file_num, descriptors, seq_len, poses=None):
    read_complete_flag = True
    descriptors_seq = torch.zeros((1, seq_len, 256)).type(torch.FloatTensor).cuda()
    for i in np.arange(
        int(file_num) - (seq_len // 2), int(file_num) - (seq_len // 2) + seq_len
    ):  # length can be changed !!!!!!!
        if i < 0 or i >= descriptors.shape[0]:
            read_complete_flag = False
            for m in np.arange(
                int(file_num) - (seq_len // 2), int(file_num) - (seq_len // 2) + seq_len
            ):
                descriptors_seq[0, int(m - int(file_num) + (seq_len // 2)), :] = (
                    torch.from_numpy(descriptors[int(file_num), :])
                    .type(torch.FloatTensor)
                    .cuda()
                )
            return descriptors_seq, read_complete_flag
        descriptor_tensor = (
            torch.from_numpy(descriptors[i, :]).type(torch.FloatTensor).cuda()
        )
        descriptors_seq[
            0, int(i - int(file_num) + (seq_len // 2)), :
        ] = descriptor_tensor

    return descriptors_seq, read_complete_flag


def read_descriptors(index_in_db_dir, descriptors, seq_len) -> torch.Tensor:
    half_seq: int = int(seq_len // 2)
    start_index: int = index_in_db_dir - half_seq
    end_index: int = index_in_db_dir - half_seq + seq_len - 1
    descriptors_seq:torch.FloatTensor = torch.zeros((1, seq_len, 256)).type(torch.FloatTensor).cuda()
    for index in range(start_index, end_index):
        if index < len(descriptors):
            descriptor_tensor = (
                torch.from_numpy(descriptors[index, :]).type(torch.FloatTensor).cuda()
            )
            descriptors_seq[0, int(index - int(index_in_db_dir)+ half_seq), :] = descriptor_tensor
        else:
            descriptor_tensor = (
                torch.from_numpy(descriptors[len(descriptors) - index, :]).type(torch.FloatTensor).cuda()
            )
            descriptors_seq[0, int(index - int(index_in_db_dir) + half_seq), :] = descriptor_tensor
    return descriptors_seq

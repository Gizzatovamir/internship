import torch
from models import featureExtracter
import numpy as np
from tqdm import tqdm
import os
import yaml


def read_one_need_from_seq(file_num:str, file_name:str, seq_len, poses=None, range_image_root=None):
    read_complete_flag = True
    depth_data_seq = torch.zeros((1, seq_len, 32, 900)).type(torch.FloatTensor).cuda()
    for i in np.arange(
        int(file_num) - (seq_len // 2), int(file_num) - (seq_len // 2) + seq_len
    ):
        if not os.path.exists(range_image_root + file_name + ".npy"):
            read_complete_flag = False
            depth_data_tmp = np.load(range_image_root + file_name + ".npy")
            depth_data_tensor_tmp = (
                torch.from_numpy(depth_data_tmp).type(torch.FloatTensor).cuda()
            )
            depth_data_tensor_tmp = torch.unsqueeze(depth_data_tensor_tmp, dim=0)
            depth_data_tensor_tmp = torch.unsqueeze(depth_data_tensor_tmp, dim=0)
            for m in np.arange(
                int(file_num) - (seq_len // 2), int(file_num) - (seq_len // 2) + seq_len
            ):
                depth_data_seq[
                    :, int(m - int(file_num) + (seq_len // 2)), :, :
                ] = depth_data_tensor_tmp
            return depth_data_seq, read_complete_flag
        depth_data = np.load(range_image_root + file_name + ".npy")
        depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
        depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
        depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
        depth_data_seq[
            :, int(i - int(file_num) + (seq_len // 2)), :, :
        ] = depth_data_tensor

    return depth_data_seq, read_complete_flag


def load_files(folder):
    """Load all files in a folder and sort."""
    file_paths = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(folder))
        for f in fn
    ]
    file_paths.sort()
    return file_paths


class Gen:
    def __init__(
        self,
        seqlen=3,
        pretrained_weights=None,
        range_image_database_root=None,
        range_image_query_root=None,
    ):
        self.seq_len = seqlen
        self.amodel = featureExtracter(seqL=self.seq_len)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        print(self.amodel)
        self.parameters = self.amodel.parameters()
        self.weights = pretrained_weights
        self.range_image_database_root = range_image_database_root
        self.range_image_query_root = range_image_query_root

    def eval(self):
        resume_filename = self.weights
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        self.amodel.load_state_dict(checkpoint["state_dict"])  # 加载状态字典

        interval = 1

        scan_paths_database = load_files(self.range_image_database_root)
        print("the number of reference scans ", len(scan_paths_database), scan_paths_database[0])
        des_list = np.zeros((int(len(scan_paths_database) // interval) + 1, 256))
        for index, timestamp in enumerate([el.split(".")[-2].split('/')[-1] for el in scan_paths_database]):
            current_batch, read_complete_flag = read_one_need_from_seq(
                str(index),
                str(timestamp),
                self.seq_len,
                range_image_root=self.range_image_database_root,
            )
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)
            des_list[int(index // interval), :] = (
                current_batch_des[0, :].cpu().detach().numpy()
            )
            del current_batch
        des_list = des_list.astype("float32")
        np.save("des_list_database", des_list)

        scan_paths_query = load_files(self.range_image_query_root)
        print("the number of query scans ", len(scan_paths_query))
        des_list_query = np.zeros((int(len(scan_paths_query) // interval) + 1, 256))
        for index, timestamp in enumerate([el.split(".")[-2].split('/')[-1] for el in scan_paths_query]):
            print(timestamp)
            current_batch, read_complete_flag = read_one_need_from_seq(
                str(index),
                str(timestamp),
                self.seq_len,
                range_image_root=self.range_image_query_root,
            )
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)
            des_list_query[int(index // interval), :] = (
                current_batch_des[0, :].cpu().detach().numpy()
            )
            del current_batch
        des_list_query = des_list_query.astype("float32")
        np.save("des_list_query", des_list_query)


if __name__ == "__main__":
    # abs path
    config_filename = "/home/amir/Desktop/pet_projects/internship/config/config.yml"
    config = yaml.safe_load(open(config_filename))
    seqlen = config["gen_sub_descriptors"]["seqlen"]
    pretrained_weights = config["gen_sub_descriptors"]["weights"]
    range_image_database_root = config["data_root"]["range_image_database_root"]
    range_image_query_root = config["data_root"]["range_image_query_root"]
    gen_descs = Gen(
        seqlen=seqlen,
        pretrained_weights=pretrained_weights,
        range_image_database_root=range_image_database_root,
        range_image_query_root=range_image_query_root,
    )
    gen_descs.eval()

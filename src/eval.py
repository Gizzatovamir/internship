import pathlib
from models.gem import GeM
from models.seqTransformerCat import featureExtracter
import numpy as np
import torch
from utils import read_data, vis_utils
import yaml
import matplotlib.pyplot as plt
from typing import List, Dict
import faiss


class Predict:
    def __init__(
        self,
        seqlen: int = 20,
        db_path: pathlib.Path = pathlib.Path(""),
        gem_pretrained_weights=None,
        feature_extracter_weights=None,
        descriptors_path: pathlib.Path = None,
    ):
        self.seqlen: int = seqlen
        self.gem_weights: pathlib.Path = gem_pretrained_weights
        self.feature_ex_weights: pathlib.Path = feature_extracter_weights
        self.descriptors_db_path: pathlib.Path = db_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: GeM = GeM().to(self.device)
        self.extracter_model: featureExtracter = featureExtracter().to(self.device)
        self.descriptors: np.ndarray = np.load(descriptors_path.as_posix())

    def load_gem_model(self) -> None:
        checkpoint = torch.load(self.gem_weights.as_posix())
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def load_feature_extracter(self):
        checkpoint = torch.load(self.feature_ex_weights.as_posix())
        self.extracter_model.load_state_dict(checkpoint["state_dict"])
        self.extracter_model.eval()

    def gen_depth_data_seq(self, test_path: pathlib.Path) -> torch.FloatTensor:
        range_image_list = read_data.get_np_array_from_path(test_path)
        depth_data_seq: torch.FloatTensor = (
            torch.zeros((1, len(range_image_list), 32, 900))
            .type(torch.FloatTensor)
            .cuda()
        )
        for index, depth_data in enumerate(range_image_list):
            depth_data_tensor = (
                torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
            )
            depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
            depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
            depth_data_seq[:, index, :, :] = depth_data_tensor
        return depth_data_seq

    def create_value_seq(self, test_path: pathlib.Path) -> np.ndarray:
        self.load_feature_extracter()
        depth_data_seq = self.gen_depth_data_seq(test_path)
        des_list: torch.FloatTensor = (
            torch.zeros((1, len(depth_data_seq), 256)).type(torch.FloatTensor).cuda()
        )
        for index, depath_data in enumerate(depth_data_seq):
            current_batch_des = self.extracter_model(depth_data_seq)
            current_batch_des = current_batch_des.squeeze(1)
            des_list[index] = current_batch_des
        del self.feature_ex_weights
        # self.feature_ex_weights = featureExtracter().to(self.device)
        return self.model(des_list).squeeze(1)[0, :].cpu().detach().numpy()

    def get_descriptor_list(self) -> np.ndarray:
        descriptors_list = np.zeros((int(self.descriptors.shape[0]), 256))
        for index, timestamp in enumerate(self.descriptors_db_path.glob("*.npy")):
            current_batch = read_data.read_descriptors(
                index, descriptors=self.descriptors, seq_len=self.seqlen
            )
            self.model.eval()
            current_batch_des = self.model(current_batch)
            current_batch_des = current_batch_des.squeeze(1)
            descriptors_list[int(index), :] = (
                current_batch_des[0, :].cpu().detach().numpy()
            )
        return descriptors_list.astype("float32")

    def eval(self, eval_dir_path: pathlib.Path):
        # creating whole np ndarray that consists out of compressed descriptors
        # (each descriptor is a subdescriptor from sequence of len == 20)
        # the sequence is a loop
        self.load_gem_model()
        des_list = self.get_descriptor_list()
        nlist = 1
        k = 22
        d = 256
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not index.is_trained
        index.train(des_list)
        assert index.is_trained
        index.add(des_list)
        eval_seq_descriptors = self.create_value_seq(eval_dir_path)
        D, I = index.search(eval_seq_descriptors.reshape(1, -1), k)
        np.save(eval_dir_path.as_posix() + "D", D)
        np.save(eval_dir_path.as_posix() + "I", I)
        # plt.imshow(D)
        # plt.show()


if __name__ == "__main__":
    info_dict_list: List[Dict[str, str]] = [{"help": "path to cfg", "type": str}]
    parser = read_data.get_parser(["--cfg"], info_dict_list)
    args = parser.parse_args()
    config: dict = yaml.safe_load(open(args.cfg))
    seqlen: int = int(config["gen_sub_descriptors"]["seqlen"])
    pretrained_weights: pathlib.Path = pathlib.Path(
        config["test_gem_prepare"]["weights"]
    )
    feature_ex_pretrained_weights: pathlib.Path = pathlib.Path(
        config["gen_sub_descriptors"]["weights"]
    )
    range_image_database_root: pathlib.Path = pathlib.Path(
        config["data_root"]["range_image_database_root"]
    )
    descriptors_path: pathlib.Path = pathlib.Path(
        config["test_gem_prepare"]["sub_descriptors_database_file"]
    )
    eval_claass: Predict = Predict(
        db_path=range_image_database_root,
        gem_pretrained_weights=pretrained_weights,
        feature_extracter_weights=feature_ex_pretrained_weights,
        descriptors_path=descriptors_path,
    )
    eval_dir: pathlib.Path = pathlib.Path(
        "/home/amir/Desktop/pet_projects/internship/to_test"
    )
    eval_claass.eval(eval_dir)

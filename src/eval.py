import pathlib
from models.gem import GeM
import numpy as np
import torch
import src.utils as utils
import yaml
import matplotlib.pyplot as plt
from typing import List
import faiss


class Predict:
    def __init__(
        self,
        seqlen: int = 20,
        db_path: pathlib.Path = pathlib.Path(""),
        pretrained_weights=None,
        descriptors_path: pathlib.Path = None,
    ):
        self.seqlen: int = seqlen
        self.weights: pathlib.Path = pretrained_weights
        self.descriptors_db_path: pathlib.Path = db_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: GeM = GeM().to(self.device)
        self.descriptors: np.ndarray = np.load(descriptors_path.as_posix())

    def create_value_seq(self, test_path: pathlib.Path) -> np.ndarray:
        pass

    def get_descriptor_list(self) -> np.ndarray:
        checkpoint = torch.load(self.weights.as_posix())
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        descriptors_list = np.zeros((int(self.descriptors.shape[0]), 256))
        for index, timestamp in enumerate(self.descriptors_db_path.glob("*.npy")):
            current_batch = utils.read_descriptors(
                index, descriptors=self.descriptors, seq_len=self.seqlen
            )
            self.model.eval()
            current_batch_des = self.model(current_batch)
            current_batch_des = current_batch_des.squeeze(1)
            descriptors_list[int(index), :] = (
                current_batch_des[0, :].cpu().detach().numpy()
            )
        return descriptors_list.astype("float32")

    def eval(self, value_seq: np.ndarray):
        # creating whole np ndarray that consists out of compressed descriptors
        # (each descriptor is a subdescriptor from sequence of len == 20)
        # the sequence is a loop
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


if __name__ == "__main__":
    config_filename: str = (
        "/home/amir/Desktop/pet_projects/internship/config/config.yml"
    )
    config: dict = yaml.safe_load(open(config_filename))
    seqlen: int = int(config["gen_sub_descriptors"]["seqlen"])
    pretrained_weights: pathlib.Path = pathlib.Path(
        config["test_gem_prepare"]["weights"]
    )
    range_image_database_root: pathlib.Path = pathlib.Path(
        config["data_root"]["range_image_database_root"]
    )
    descriptors_path: pathlib.Path = pathlib.Path(
        config["test_gem_prepare"]["sub_descriptors_database_file"]
    )
    eval_claass: Predict = Predict(
        db_path=range_image_database_root,
        pretrained_weights=pretrained_weights,
        descriptors_path=descriptors_path,
    )
    eval_claass.eval(np.zeros(100))

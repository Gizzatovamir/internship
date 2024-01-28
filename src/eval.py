import pathlib
from models.gem import GeM
import numpy as np
import torch

class Predict:
    def __init__(self, seqlen=20, pretrained_weights=None, descs_database=None):
        self.seqlen: int = seqlen
        self.weights: pathlib.Path = pretrained_weights
        self.descriptors = descs_database
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: GeM = GeM().to(self.device)

    def eval(self, value: np.ndarray):
        checkpoint = torch.load(self.weights.as_posix())
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        current_batch, _ = read_one_need_descriptor_from_seq_ft(f1_index, self.descs_database, seq_len=self.seqlen)
        current_batch_des = self.amodel(current_batch)
        current_batch_des = current_batch_des.squeeze(1)
#!/usr/bin/python3
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np

class ChessValueDataset(Dataset):
    def __init__(self):
        data = np.load("processed/dataset_2M.npz")
        self.X = data['arr_0']
        self.Y = data['arr_1']
        print("loaded", self.X.shape, self.Y.shape)


    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


chess_dataset = ChessValueDataset()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

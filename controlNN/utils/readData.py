import os.path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
class myData(Dataset):
    def __init__(self, rootDir, inputDir, labelDir):
        self.rootDir = rootDir
        self.inputDir = os.path.join(rootDir, inputDir)
        self.labelDir = os.path.join(rootDir, labelDir)
        self.inputList = os.listdir(self.inputDir)

    def __getitem__(self, idx):
        inputName = self.inputList[idx]
        inputAbsName = os.path.join(self.inputDir, inputName)
        inTensor = torch.load(inputAbsName)
        labelAbsName = os.path.join(self.labelDir, inputName)
        labelTensor = torch.load(labelAbsName)
        return inTensor, labelTensor

    def __len__(self):
        return len(self.inputList)


class myData4Test(Dataset):
    def __init__(self, rootDir, inputDir, labelDir):
        self.rootDir = rootDir
        self.inputDir = os.path.join(rootDir, inputDir)
        self.inputList = os.listdir(self.inputDir)

    def __getitem__(self, idx):
        inputName = self.inputList[idx]
        inputAbsName = os.path.join(self.inputDir, inputName)
        inTensor = torch.load(inputAbsName)
        return inTensor

    def __len__(self):
        return len(self.inputList)

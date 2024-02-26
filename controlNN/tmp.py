import torch
import torch.nn
from tool.Global import *
x = torch.load(nn_data_path+"/0.pt")
y = torch.load(nn_label_path+"/0.pt")
logSfM = -y.log_softmax(dim = -1)
loss = torch.sum(logSfM * y)
print(loss * 512)
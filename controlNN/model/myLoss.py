import torch
import torch.nn as nn
class CELoss(nn.Module): # 注意继承 nn.Module
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, x, y):
        logSfM = -x.log_softmax(dim = -1)
        loss = torch.sum(logSfM * y)
        return loss

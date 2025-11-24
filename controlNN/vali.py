import sys
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch
import torchvision
# from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from torch.utils.tensorboard import SummaryWriter
from utils.readData import myData
from model.gateModel import gateNet
from model.myLoss import CELoss
from tool.Global import *

# rootDir = "data4Gate"
rootDir = nn_path
w_path = weight_path
inputDir = "input"
labelDir = "label"
dataSet = myData(rootDir, inputDir, labelDir)
trainSet, valiSet = torch.utils.data.random_split(dataSet, [int(0.8 * len(dataSet)), len(dataSet) - int(0.8 * len(dataSet))])
trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=512, shuffle=True)
valiLoader = torch.utils.data.DataLoader(dataset=valiSet, batch_size=512, shuffle=False)
testSize = len(valiSet)
# 从环境变量获取设备，如果没有设置则使用默认值
import os
from dotenv import load_dotenv
load_dotenv()
device_str = os.getenv("DEVICE")
if device_str is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_str)
print("running on device: ", device)


learningRate = 1e-3
nExperts = 8
epochNum = 200
bestLoss = np.inf
bestF1 = 0.0
bestWeight = None
bestEpoch = 0
patience = 20
batchNum = len(trainLoader)
withoutDev = 0


model = gateNet(num_class=nExperts)
model.load_state_dict(torch.load(w_path+'/best.pth'))
model = model.to(device)
lossF = CELoss()
lossF = lossF.to(device)
optim = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)

# writer = SummaryWriter("./logs_train")


model.eval()
valiLoss = 0
prob_all = []
label_all = []
with torch.no_grad():
    for data in valiLoader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        for i in range(outputs.shape[0]):
            outNP = np.array(nn.functional.softmax(outputs[i]).cpu())
            outIndex = outNP.argsort()
            print(outIndex[4:])
            targetNP = np.array(targets[i].cpu())
            targetIndex = targetNP.argsort()
            print(targetIndex[4:])
        valiLoss += lossF(outputs, targets)
valiLoss /= len(valiLoader)
# print("epoch: {}  batch: {}".format(epoch + 1, batchIndex))
print("    vali loss: {}".format(valiLoss))





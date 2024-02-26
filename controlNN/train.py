# *_*coding:utf-8 *_*
import sys
sys.path.append("E:/py/Nlp/NLP")
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch
import torchvision
from tool.Global import *
# from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from torch.utils.tensorboard import SummaryWriter
from utils.readData import myData
from model.gateModel import gateNet
from model.myLoss import CELoss

# rootDir = "data4Gate"
rootDir = nn_path
inputDir = "input"
labelDir = "label"
w_path = weight_path



dataSet = myData(rootDir, inputDir, labelDir)
trainSet, valiSet = torch.utils.data.random_split(dataSet, [int(0.8 * len(dataSet)), len(dataSet) - int(0.8 * len(dataSet))])
trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=2048, shuffle=True)
valiLoader = torch.utils.data.DataLoader(dataset=valiSet, batch_size=2048, shuffle=False)
testSize = len(valiSet)
device = torch.device("cuda")


learningRate = 1e-3
nExperts = 8
epochNum = 1
bestLoss = np.inf
bestF1 = 0.0
bestWeight = None
bestEpoch = 0
patience = 20
batchNum = len(trainLoader)
withoutDev = 0


model = gateNet(num_class=nExperts)
model = model.to(device)
lossF = CELoss()
lossF = lossF.to(device)
optim = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)

# writer = SummaryWriter("./logs_train")

for epoch in range(epochNum):
    print("-----epoch {} start-----".format(epoch + 1))
    trainLoss = 0
    batchIndex = 0
    for data in trainLoader:
        model.train()  # 训练阶段
        imgs, targets = data
        print(imgs.shape)
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        result_loss = lossF(outputs, targets)
        trainLoss += result_loss
        optim.zero_grad()  # 首先对优化器的梯度清零
        result_loss.backward()  # 反向传播
        optim.step()  # 优化器更新模型参数
        # if batchIndex % 100 == 0:
        #     print("{} / {}".format(batchIndex, len(trainLoader)))
        batchIndex += 1
        # sched.step()
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
            valiLoss += lossF(outputs, targets)
    valiLoss /= len(valiLoader)
    print("epoch: {}  batch: {}".format(epoch + 1, batchIndex))
    print("    vali loss: {}".format(valiLoss))

    # writer.add_scalar("test_loss", valiLoss, epoch * batchNum + batchIndex)
    # writer.add_scalar("test_f1", valiF1, epoch * batchNum + batchIndex)

    if bestLoss > valiLoss:
        withoutDev = 0
        bestLoss = valiLoss
        bestEpoch = epoch + 1
        bestBatch = batchIndex
        bestWeight = model.state_dict()
        torch.save(bestWeight, w_path+"/best.pth")
    else:
        withoutDev += 1
    print("    best loss: {}".format(bestLoss))
    print("    patience: {}".format(patience - withoutDev))
    if withoutDev == patience:
        print("overFitting, exit.")
        print("best epoch: {}".format(bestEpoch))
        print("best batch: {}".format(bestBatch))
        # writer.close()
        sys.exit()
    trainLoss /= len(trainLoader)
    print("train loss: {}".format(trainLoss))
    epochWeight = w_path+'/epoch' + str(epoch) + '.pth'
    torch.save(model.state_dict(), epochWeight)
    # writer.add_scalar("train_loss", trainLoss, epoch + 1)

# tensorboard --logdir=logs_train



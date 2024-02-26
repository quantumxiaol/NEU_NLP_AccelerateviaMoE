import torch
import torch.nn as nn

class gateNet(nn.Module):
    def __init__(self, num_class):
        super(gateNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_class)
        )

    def forward(self, x):
        output = self.fc(x)
        return output
# ex1 = torch.ones([1, 1, 256])
# ex2 = torch.ones([1, 3, 256])
# model = gateNet(num_class=8)
# model.load_state_dict(torch.load("../weights/best.pth"))
# model.eval()
# # print(model(ex1))
# a:torch.Tensor = nn.functional.softmax(model(ex2), dim=-1)
# # a = model(ex2)
#
# a = torch.sum(a, dim=-2)
# print(a)
# sorted, indices = torch.sort(a, descending=True)
# print(sorted, indices)


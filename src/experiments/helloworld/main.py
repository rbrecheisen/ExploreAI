import torch

from torch import nn
from torchsummary import summary

from utilities import select_idle_gpu_device


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(4, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layer3 = nn.Linear(5, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    

model = MyModel().to(select_idle_gpu_device())

summary(model, input_size=(4, ))
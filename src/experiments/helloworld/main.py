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
    

class MyMultiLayerPerceptronForImaging(nn.Module):
    def __init__(self):
        super(MyMultiLayerPerceptronForImaging, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(28*28, 512)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x
    

class MyConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(MyConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        return x
    

device = select_idle_gpu_device()
device = 'cuda:0'
print(device)
print(torch.cuda.device_count())
# model = MyModel().to(device)
# model = MyMultiLayerPerceptronForImaging().to(device)
model = MyConvolutionalNetwork().to(device)
# summary(model, input_size=(4, ))
summary(model, input_size=(1, 28, 28))
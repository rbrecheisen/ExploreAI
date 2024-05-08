import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt

DATADIR = 'C:\\Users\\r.brecheisen\\Development\\Data'


def show_image(image):
    image = image / 2 + 0.5
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, padding='same')
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, padding='same')
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initialize the model
model = SimpleCNN()

# Print the model summary (You might want to use additional tools like torchsummary for detailed summary similar to Keras)
print(model)


if __name__ == '__main__':

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        )
    ])

    # Specify training and test sets based on CIFAR-10 dataset
    training_set = torchvision.datasets.CIFAR10(
        root=DATADIR, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root=DATADIR, train=False, download=True, transform=transform)
    
    # Split training set into training and validation sets
    validation_size = 5000
    training_size = len(training_set) - validation_size
    training_set, validation_set = torch.utils.data.random_split(training_set, [training_size, validation_size])

    # Specify loaders for training, validation and test sets
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=32, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=32, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        training_set, batch_size=32, shuffle=False, num_workers=2)

    # images, labels = next(iter(training_loader))
    # figure = plt.figure(figsize=(20, 5))
    # for idx in range(32):
    #     axis = figure.add_subplot(3, 12, idx + 1, xticks=[], yticks=[])
    #     show_image(images[idx])
    # plt.show()

    for images, labels in training_loader:                  # Iterates over batches of images, not single images
        labels_onehot = F.one_hot(labels, num_classes=10)

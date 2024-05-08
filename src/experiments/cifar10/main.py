import os
import torch
import torch.utils
import torch.utils.data
import torchvision

import numpy as np
import matplotlib.pyplot as plt

DATADIR = 'C:\\Users\\r.brecheisen\\Development\\Data'

def show_image(image):
    image = image / 2 + 0.5
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))

if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        )
    ])

    training_set = torchvision.datasets.CIFAR10(
        root=DATADIR, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root=DATADIR, train=False, download=True, transform=transform)

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=36, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        training_set, batch_size=36, shuffle=False, num_workers=2)

    images, labels = next(iter(training_loader))

    figure = plt.figure(figsize=(20, 5))
    for idx in range(36):
        axis = figure.add_subplot(3, 12, idx + 1, xticks=[], yticks=[])
        show_image(images[idx])

    plt.show()
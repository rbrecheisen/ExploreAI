import os
import torch
import torch.optim
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import torchvision
import time
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
    

def save_checkpoint(model, optimizer, epoch, path):
    print(f'saving checkpoint for epoch {epoch} to {path}...')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def validate(model, val_loader, criterion, device='cpu'):
    print(f'calculating validation loss and accuracy (device: {device})')
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels_onehot = F.one_hot(labels, num_classes=10).float()
            outputs = model(images)
            loss = criterion(outputs, labels_onehot)
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


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

    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    num_epochs = 10
    best_val_acc = 0.0
    epoch_elapsed_time = 0
    epoch_fixed_elapsed_time = 0
    batch_elapsed_time = 0
    batch_fixed_elapsed_time = 0
    final_epoch = None

    try:
        for epoch in range(num_epochs):
            final_epoch = epoch
            print(f'starting epoch {epoch}...')
            epoch_start_time = int(round(time.time() * 1000))

            model.train() # Set model to training state
            i = 0
            for images, labels in training_loader: # Iterate over batches of images and labels in training data
                batch_start_time = int(round(time.time() * 1000))

                labels_onehot = F.one_hot(labels, num_classes=10).float() # One-hot encode the labels
                optimizer.zero_grad() # Reset gradient to zero for each batch
                outputs = model(images) # Do forward pass of images through model
                loss = F.binary_cross_entropy_with_logits(outputs, labels_onehot) # Calculate loss
                loss.backward() # Do back-propagation to update weights
                optimizer.step() # Take next step along calculated gradients
                batch_stop_time = int(round(time.time() * 1000))
                batch_elapsed_time = batch_stop_time - batch_start_time
                if batch_fixed_elapsed_time == 0:
                    batch_fixed_elapsed_time = batch_elapsed_time
                i += 1

            print(f'{epoch}: calculating validation loss...')
            avg_loss, accuracy = validate(model, validation_loader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu') # Calculate validation accuracy
            print(f'{epoch}: validation loss: {avg_loss}, validation accuracy: {accuracy}')

            if epoch % 5 == 0 or accuracy > best_val_acc: # Every 5 epochs, save model checkpoint and save best accuracy so far
                print(f'{epoch}: validation loss has improved, updating...')
                if accuracy > best_val_acc:
                    best_val_acc = accuracy
                print(f'{epoch}: saving checkpoint...')
                save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pth')

            epoch_end_time = int(round(time.time() * 1000))
            epoch_elapsed_time = epoch_end_time - epoch_start_time
            if epoch_fixed_elapsed_time == 0:
                epoch_fixed_elapsed_time = epoch_elapsed_time
    except KeyboardInterrupt:
        # save_checkpoint(model, optimizer, final_epoch, f'checkpoint_epoch_{final_epoch}.pth')
        pass
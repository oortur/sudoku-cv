import random
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets


SEED = 0xBadCafe
MNIST_CELL_SIZE = 28
DIGIT_CLASSIFIER_PATH = './mnist.model'


class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, 
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, 
                               kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=640, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(save_model_path=DIGIT_CLASSIFIER_PATH, print_accuracy=False):
    # FIXING RANDOM SEED:
    fix_seed(SEED)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    # get the data
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    digit_model = MnistClassifier().to(device)

    n_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    start = time.time()
    for epoch in range(n_epochs):
        # running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = digit_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # # print for mini batches
            # running_loss += loss.item()
            # if i % 5000 == 4999:  # every 5000 mini batches
            #     print('[Epoch %d, %5d Mini Batches] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss/5000))
            #     running_loss = 0.0
    end = time.time()
    print('Done Training')
    print('%0.2f minutes' %((end - start) / 60))

    if print_accuracy:
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                outputs = digit_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on test images: %0.3f %%' % (100 * correct / total))

    # save model
    torch.save(digit_model.state_dict(), save_model_path)

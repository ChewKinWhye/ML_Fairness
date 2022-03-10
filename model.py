import torch
import torchvision.models as models
import torch.nn as nn

# TODO: Add more complex models

class Simple_CNN(nn.Module):
    def __init__(self, num_classes, width):
        super(Simple_CNN, self).__init__()
        self.width = width
        # CL1:   28 x 28  -->    50 x 28 x 28
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, padding=1)
        # MP1: 50 x 28 x 28 -->    50 x 14 x 14
        self.pool1 = nn.MaxPool2d(2, 2)
        # CL2:   50 x 14 x 14  -->    100 x 14 x 14
        self.conv2 = nn.Conv2d(width, width*2, kernel_size=3, padding=1)
        # MP2: 100 x 14 x 14 -->    100 x 7 x 7
        self.pool2 = nn.MaxPool2d(2, 2)
        # LL1:   100 x 7 x 7 = 4900 -->  100
        self.linear1 = nn.Linear(width*2*7*7, 128)
        # LL2:   100  -->  2
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.view(-1, self.width*2*7*7)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)

        return x


class Small_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Small_CNN, self).__init__()

        # CL1:   28 x 28  -->    16 x 28 x 28
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # MP1: 16 x 28 x 28 -->    16 x 14 x 14
        self.pool1 = nn.MaxPool2d(2, 2)
        # CL2:   16 x 14 x 14  -->    32 x 14 x 14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # MP2: 32 x 14 x 14 -->    32 x 7 x 7
        self.pool2 = nn.MaxPool2d(2, 2)
        # LL1:   32 x 7 x 7 = 1568 -->  32
        self.linear1 = nn.Linear(1568, 32)
        # LL2:   32  -->  2
        self.linear2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 1568)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)

        return x


def get_model(model_name, num_classes=2):
    if model_name == 'SimpleCNN':
        net = Simple_CNN(num_classes)
    elif model_name == 'SmallCNN':
        net = Small_CNN(num_classes)
    else:
        raise Exception("Invalid model")

    # elif model_name == 'resnet18':
    #     net = models.resnet18()
    #     # Replace last layer
    #     print(net)
    #     net.fc = nn.Linear(net.fc.in_features, num_classes)
    # elif model_name == 'vgg16':
    #     net = models.vgg16()
    #     # Replace last layer
    #     net.classifier[6] = nn.Linear(net.classifier[6].in_features, num_classes)
    return net

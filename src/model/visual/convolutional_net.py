import torch.nn as nn
import torch.nn.functional as F
from torch import (flatten)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input image dimension
        self.__input_size = (3, 256, 256)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=4)
        self.fc1 = nn.Linear(16 * 15 * 15, 1080)

    def forward(self, x):
        x = self.max_pool1(F.relu(self.conv1(x)))
        x = self.max_pool1(F.relu(self.conv2(x)))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

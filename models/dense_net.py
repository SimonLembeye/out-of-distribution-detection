import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

class DenseLayer(nn.Module):
    def __init__(self, num_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_channels, 4 * growth_rate, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding = 1, bias = False)
    
    def forward(self, x):
        output = self.relu1(self.bn1(x))
        output = self.conv1(output)
        output = self.relu2(self.bn2(output))
        output = self.conv2(output)
        return torch.cat([x, output], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_channels, growth_rate = 12, depth = 100):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(depth):
            layers.append(DenseLayer(num_channels + i * growth_rate, growth_rate))
        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_block(x)
        
class Transition(nn.Module):
    def __init__(self, num_channels, compression = .5):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.conv = nn.Conv2d(num_channels, floor(compression * num_channels), 1, bias = False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(self.bn(x)))

class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate = 12, depth = 100, compression = .5):
        super(DenseNet, self).__init__()
        self.conv = nn.Conv2d(3, 2*growth_rate, 3, bias = False)
        num_channels = 2 * growth_rate

        self.block1 = DenseBlock(num_channels, growth_rate, depth)
        num_channels = num_channels + depth * growth_rate
        self.transition1 = Transition(num_channels, compression)
        num_channels = floor(compression * num_channels)

        self.block2 = DenseBlock(num_channels, growth_rate, depth)
        num_channels = num_channels + depth * growth_rate
        self.transition2 = Transition(num_channels, compression)
        num_channels = floor(compression * num_channels)

        self.block3 = DenseBlock(num_channels, growth_rate, depth)
        num_channels = num_channels + depth * growth_rate

        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.transition1(self.block1(x))
        x = self.transition2(self.block2(x))
        x = self.block3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
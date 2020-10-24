import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

class WideLayer(nn.Module):
    def __init__(self, num_channels, num_out, dropout, stride):
        super(WideLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(num_channels, num_out, 3, padding = 1, bias = False, stride = stride)

        self.convx = nn.Conv2d(num_channels, num_out, 1, padding = 0, bias = False, stride = stride)

        self.bn2 = nn.BatchNorm2d(num_out)
        self.relu2 = nn.ReLU(inplace = True)

        self.dropout = dropout

        self.conv2 = nn.Conv2d(num_out, num_out, 3, padding = 1, bias = False, stride = 1)

        self.equal_size = (num_channels == num_out)
    
    def forward(self, x):
        output = self.relu1(self.bn1(x))
        if not(self.equal_size) :
            x = self.convx(output)
        output = self.conv1(output)
        output = self.relu2(self.bn2(output))
        if self.dropout > 0:
            output = F.dropout(output, p = self.dropout)
        output = self.conv2(output)
        return torch.add(x, output)

class WideBlock(nn.Module):
    def __init__(self, depth, num_channels, num_out, dropout, stride):
        super(WideBlock, self).__init__()
        layers = []
        layers.append(WideLayer(num_channels, num_out, dropout, stride))
        for i in range(1, depth):
            layers.append(WideLayer(num_out, num_out, dropout, 1))
        self.wide_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.wide_block(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes,depth = 28, width = 10, dropout = .3):
        super(WideResNet, self).__init__()
        d = (depth - 4) // 6
        self.conv = nn.Conv2d(3, 16, 3, bias = False)

        self.block1 = WideBlock(d, 16, 16 * width, dropout, 1)
        self.block2 = WideBlock(d, 16 * width, 32 * width, dropout, 2)
        self.block3 = WideBlock(d, 32 * width, 64 * width, dropout, 2)

        num_channels = 64 * width
        self.num_channels = num_channels

        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace = True)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.num_channels)
        x = self.fc(x)
        return x

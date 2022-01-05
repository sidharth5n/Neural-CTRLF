import torch
import torch.nn as nn

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(PreActResNet, self).__init__()
        self.in_channels = 64
        descriptor_size = 512
        self.model = nn.Sequential(nn.Conv2d(1, 64, 7, 2, 3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3, 2, 1),
                                   self._make_layer(block, 64, num_blocks[0], 1, 'first'),
                                   self._make_layer(block, 128, num_blocks[1], 2),
                                   self._make_layer(block, 256, num_blocks[2], 2),
                                   self._make_layer(block, 512, num_blocks[3], 2),
                                   nn.BatchNorm2d(self.in_channels),
                                   nn.ReLU(),
                                   nn.AvgPool2d((5, 2), 1))
        self.linear = nn.Linear(512 * block.expansion, descriptor_size)
    
    def forward(self, x):
        x = self.model(x)
        print(x.shape)
        return self.linear(x.squeeze())

    def _make_layer(self, block, features, count, stride, first = False):
        s = []
        for i in range(count):
            s.append(block(self.in_channels, features, stride, first and i == 0))
            self.in_channels = features * block.expansion
        return nn.Sequential(*s)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n, stride, first):
        super(BasicBlock, self).__init__()
        if first:
            layers = [nn.BatchNorm2d(in_channels), nn.ReLU()]
        else:
            layers = []

        layers.extend([nn.Conv2d(in_channels, n, 3, stride, 1),
                       nn.BatchNorm2d(n),
                       nn.ReLU(),
                       nn.Conv2d(n, n, 3, 1, 1)])
        
        self.layers = nn.Sequential(*layers)

        if in_channels != n*self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, n*self.expansion, 1, stride),
                                          nn.BatchNorm2d(n*self.expansion))

    def forward(self, x):
        x1 = self.layers(x)
        x2 = self.shortcut(x) if hasattr(self, 'shortcut') else x
        return x1 + x2

class BottleNeck(nn.Module):
    """
    The bottleneck residual layer for 50, 101, and 152 layer networks
    """
    expansion = 4

    def __init__(self, in_channels, n, stride, first):
        super(BottleNeck, self).__init__()
        if first:
            layers = [nn.BatchNorm2d(in_channels), nn.ReLU()]
        else:
            layers = []
        layers.extend([nn.Conv2d(in_channels, n, 1, 1),
                       nn.BatchNorm2d(n),
                       nn.ReLU(),
                       nn.Conv2d(n, n, 3, stride, 1),
                       nn.BatchNorm2d(n),
                       nn.ReLU(),
                       nn.Conv2d(n, n*self.expansion, 1, 1)])
        
        self.layers = nn.Sequential(*layers)

        if in_channels != n*self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, n*self.expansion, 1, stride),
                                          nn.BatchNorm2d(n*self.expansion))

    def forward(self, x):
        x1 = self.layer(x)
        x2 = self.shortcut(x) if hasattr(self, 'shortcut') else x
        return x1 + x2

def PreActResNet18():
    return PreActResNet(BasicBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(BasicBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(Bottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(Bottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(Bottleneck, [3,8,36,3])

def PreActResNet200():
    return PreActResNet(Bottleneck, [3,24,36,3])


if __name__ == '__main__':
    print(PreActResNet18().model[:2])
    print(len(PreActResNet18().model))
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1,
                conv_layer=None,
                norm_layer=None,
                activation_layer=None):
        super(wide_basic, self).__init__()
        self.norm1 = norm_layer(in_planes)
        self.relu = activation_layer(planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(BaseModel):
    def __init__(self, depth, widen_factor, num_classes=10,
                norm_layer_type = 'bn',
                conv_layer_type = 'conv2d',
                linear_layer_type = 'linear',
                activation_layer_type = 'relu'):
        super().__init__(norm_layer_type, conv_layer_type, linear_layer_type, activation_layer_type)
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.relu = self.activation_layer(self.in_planes)
        self.conv1 = self.conv_layer(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, stride=2)
        self.norm1 = self.norm_layer(nStages[3])
        self.linear = self.linear_layer(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                self.conv_layer, self.norm_layer,
                                self.activation_layer))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.norm1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def ResNet16_1(**kwargs):
    return Wide_ResNet(16,1, **kwargs)

def ResNet16_2(**kwargs):
    return Wide_ResNet(16,2, **kwargs)

def ResNet16_3(**kwargs):
    return Wide_ResNet(16,3, **kwargs)

def ResNet16_4(**kwargs):
    return Wide_ResNet(16,4, **kwargs)

def ResNet16_5(**kwargs):
    return Wide_ResNet(16,5, **kwargs)

def ResNet16_6(**kwargs):
    return Wide_ResNet(16,6, **kwargs)

    

def ResNet16_10(**kwargs):
    return Wide_ResNet(16,10, **kwargs)

def ResNet28_10(**kwargs):
    return Wide_ResNet(28,10,10, **kwargs)

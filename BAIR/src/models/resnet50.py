import os
from os.path import join, dirname
from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        size = m.weight.size()
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.fill_(0)


class ResNet(nn.Module):
    def __init__(self, block, layers, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512*block.expansion, classes)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        z = torch.flatten(x,start_dim=-2,end_dim=-1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x, z

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_path = join(dirname(__file__), 'pretrained_models/resnet18-5c106cde.pth')
    # model_path = 'pretrained_models/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path), strict=False)

    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model_path = join(dirname(__file__), 'pretrained_models/resnet50-19c8e357.pth')
    if pretrained:
        model.load_state_dict(torch.load(model_path), strict=False)
    return model



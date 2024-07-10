import torch
import torchvision.models
from torch import nn


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, dropout_rate=0.0):
        super(ResNet, self).__init__()
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048
        # adapt number of channels
        nc = 3
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, return_decode_feature=False):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))
        # # x 3, 224, 224
        # x = self.network.conv1(x) # 64, 112, 112
        # x = self.network.bn1(x)
        # x = self.network.relu(x)
        # x = self.network.maxpool(x) # 64, 56, 56
        # # x, return_indices = self.maxpool(x) # 64, 56, 56


        # decode = x
        # x = self.network.layer1(x) # 256, 56, 56
        # x = self.network.layer2(x) # 512, 28, 28
        # x = self.network.layer3(x) # 1024, 14, 14
        # x = self.network.layer4(x) # 2048, 7, 7
        # x = self.network.avgpool(x)
        # x = x.view(x.size(0), -1)
        # if return_decode_feature:
        #     return x, decode
        return x



    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

'''
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
'''

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, output_padding: int = 0) -> nn.Conv2d:
	"""3x3 convolution with padding"""
	return nn.ConvTranspose2d(
		in_planes,
		out_planes,
		kernel_size=3,
		stride=stride,
		padding=dilation,
		output_padding = output_padding,
		groups=groups,
		bias=False,
		dilation=dilation,
	)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0) -> nn.Conv2d:
	"""1x1 convolution"""
	return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, output_padding = output_padding)


class Bottleneck(nn.Module):
	# Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
	# while original implementation places the stride at the first 1x1 convolution(self.conv1)
	# according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
	# This variant is also known as ResNet V1.5 and improves accuracy according to
	# https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

	expansion: int = 4

	def __init__(
		self,
		inplanes: int,
		planes: int,
		stride: int = 1,
		output_padding: int = 0,
		upsample: Optional[nn.Module] = None,
		groups: int = 1,
		base_width: int = 64,
		dilation: int = 1,
		norm_layer: Optional[Callable[..., nn.Module]] = None,
	) -> None:
		super().__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.0)) * groups
		# Both self.conv2 and self.upsample layers upsample the input when stride != 1
		self.conv3 = conv1x1(planes * self.expansion, width)
		self.bn3 = norm_layer(planes)

		self.conv2 = conv3x3(width, width, stride, groups, dilation, output_padding)
		self.bn2 = norm_layer(width)

		self.conv1 = conv1x1(width, inplanes)
		self.bn1 = norm_layer(inplanes)
		
		
		self.relu = nn.ReLU(inplace=True)
		self.upsample = upsample
		self.stride = stride

	def forward(self, x: Tensor) -> Tensor:
		identity = x

		out = self.conv3(x)
		out = self.bn3(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv1(out)
		out = self.bn1(out)
		if self.upsample is not None:
			identity = self.upsample(x)
		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(
		self,
		block,
		layers: List[int],
		num_classes: int = 1000,
		zero_init_residual: bool = False,
		groups: int = 1,
		width_per_group: int = 64,
		norm_layer: Optional[Callable[..., nn.Module]] = None,
		# indices = None,
	) -> None:
		super().__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 2048
		self.dilation = 1
		self.groups = groups
		self.base_width = width_per_group
		self.de_conv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False)
		self.unpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
		self.bn1 = norm_layer(3)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.unsample = nn.Upsample(size=7, mode='nearest')

		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer1 = self._make_layer(block, 64, layers[0], output_padding = 0, last_block_dim=64)

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck) and m.bn3.weight is not None:
					nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
				elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
					nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

	def _make_layer(
		self,
		block,
		planes: int,
		blocks: int,
		stride: int = 1,
		output_padding: int = 1,
		last_block_dim: int = 0,
	) -> nn.Sequential:
		norm_layer = self._norm_layer
		upsample = None
		previous_dilation = self.dilation

		layers = []
		self.inplanes = planes * block.expansion
		if last_block_dim == 0:
			last_block_dim = self.inplanes//2
		if stride != 1 or self.inplanes != planes * block.expansion or output_padding==0:
			upsample = nn.Sequential(
				conv1x1(planes * block.expansion, last_block_dim, stride, output_padding),
				# norm_layer(planes * block.expansion),
				norm_layer(last_block_dim),
			)
		last_block = block(
				last_block_dim, planes, stride, output_padding, upsample, self.groups, self.base_width, previous_dilation, norm_layer
			)
		
		for _ in range(1, blocks):
			layers.append(
				block(
					self.inplanes,
					planes,
					groups=self.groups,
					base_width=self.base_width,
					dilation=self.dilation,
					norm_layer=norm_layer,
				)
			)
		layers.append(last_block)
		return nn.Sequential(*layers)

	def _forward_impl(self, x: Tensor, indices) -> Tensor:
		# See note [TorchScript super()]
		x = self.unsample(x)
		x = self.layer4(x)
		x = self.layer3(x)
		x = self.layer2(x)
		x = self.layer1(x)

		# print(x.shape)
		x = self.unpool(x, indices)
		x = self.de_conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		return x

	def _forward_cnns_only(self, x: Tensor) -> Tensor:
		# See note [TorchScript super()]
		x = self.unsample(x)
		x = self.layer4(x)
		x = self.layer3(x)
		x = self.layer2(x)
		x = self.layer1(x)
		return x

	def forward(self, x: Tensor, indices=None) -> Tensor:
		if indices is None:
			return self._forward_cnns_only(x)
		return self._forward_impl(x, indices)

	


class Decoder(nn.Module):
	def __init__(self, hidden_dim=2048):
		super(Decoder, self).__init__()
		self.hidden_dim = hidden_dim
		self.dfc4 = nn.Linear(hidden_dim, 2048)
		self.dfc3 = nn.Linear(2048, 4096)
		self.bn3 = nn.BatchNorm1d(4096)
		self.dfc2 = nn.Linear(4096, 4096)
		self.bn2 = nn.BatchNorm1d(4096)
		self.dfc1 = nn.Linear(4096, 256 * 6 * 6)
		self.bn1 = nn.BatchNorm1d(256*6*6)
		self.upsample1=nn.Upsample(scale_factor=2)
		self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
		self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
		self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
		self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
		self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)
		self.bn3.eval()
		self.bn2.eval()
		self.bn1.eval()

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
		

	def forward(self,x):#,i1,i2,i3):
		
		if x.shape[-1] < 2048:
			x = self.dfc4(x)
		
		x = self.dfc3(x)
		x = F.relu(self.bn3(x))
		x = self.dfc2(x)
		x = F.relu(self.bn2(x))
		x = self.dfc1(x)
		x = F.relu(self.bn1(x))

		# ---------------------------------
		# import pdb; pdb.set_trace()
		# x = self.dfc3(x)
		# x = F.relu(x)
		# x = self.dfc2(x)
		# x = F.relu(x)
		# x = self.dfc1(x)
		# x = F.relu(x)
		# ---------------------------------
		#print(x.size())
		batch_size = x.shape[0]
		x = x.view(batch_size, 256, 6, 6)
		#print (x.size())
		x=self.upsample1(x)
		#print x.size()
		x = self.dconv5(x)
		#print x.size()
		x = F.relu(x)
		#print x.size()
		x = F.relu(self.dconv4(x))
		#print x.size()
		x = F.relu(self.dconv3(x))
		#print x.size()		
		x=self.upsample1(x)
		#print x.size()		
		x = self.dconv2(x)
		#print x.size()		
		x = F.relu(x)
		x=self.upsample1(x)
		#print x.size()
		x = self.dconv1(x)
		#print x.size()
		# x = F.sigmoid(x)
		#print x
		return x

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.utils.model_zoo as model_zoo
import math
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule

from ..builder import BACKBONES

# from ..utils import ResLayer

webroot = 'http://dl.yf.io/drn/'

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=(1, 1),
                 residual=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()

        self.conv2_stride = stride
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation[1],
            dilation=dilation[1],
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class DRN(BaseModule):
    arch_settings = {
        54: (Bottleneck, (1, 1, 3, 4, 6, 3, 1, 1)),
        56: (Bottleneck, (1, 1, 3, 4, 6, 3, 2, 2)),
        105: (Bottleneck, (1, 1, 3, 4, 23, 3, 1, 1)),
        157: (Bottleneck, (1, 1, 3, 4, 23, 3, 2, 2))
    }

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 in_channels=3,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False,
                 out_middle=False,
                 pool_size=28,
                 arch='D',
                 depth=54):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        self.block, layers = self.arch_settings[depth]

        self.drn_layers = []
        if arch == 'D':
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                channels[0],
                kernel_size=7,
                stride=2,  # 需要注意
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, channels[0], postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

            self.layer0 = nn.Sequential(
                self.conv1,
                norm1,
                self.relu
            )

            self.layer1 = self._make_conv_layers(self.conv_cfg, channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(self.conv_cfg, channels[1], layers[1], stride=2)

            self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
            self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
            self.layer5 = self._make_layer(block, channels[4], layers[4],
                                           dilation=2, new_level=False)
            self.layer6 = None if layers[5] == 0 else \
                self._make_layer(block, channels[5], layers[5], dilation=4,
                                 new_level=False)

            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

            # if num_classes > 0:
            #     self.avgpool = nn.AvgPool2d(pool_size)
            #     self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
            #                         stride=1, padding=0, bias=True)
            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d):
            #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #         m.weight.data.normal_(0, math.sqrt(2. / n))
            #     elif isinstance(m, BatchNorm):
            #         m.weight.data.fill_(1)
            #         m.bias.data.zero_()

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:    # 因为第二项，此处必运行？
            self.drn_conv = build_conv_layer(
                self.conv_cfg,
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False)
            self.drn_norm_name, drn_norm = build_norm_layer(        # 是否需要加self.
                self.norm_cfg, planes * block.expansion, postfix=1)
            downsample = nn.Sequential(
                self.drn_conv,
                drn_norm,
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, conv_cfg, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            self.drn_conv = build_conv_layer(
                self.conv_cfg,
                self.inplanes,
                channels,
                kernel_size=3,
                stride=stride if i == 0 else 1,
                padding=dilation,
                bias=False,
                dilation=dilation)
            self.drn_norm_name, drn_norm = build_norm_layer(        # 是否需要加self.
                self.norm_cfg, channels, postfix=str(i+1))
            self.add_module(self.drn_norm_name, drn_norm)
            self.relu = nn.ReLU(inplace=True)

            modules.extend([self.drn_conv,
                            drn_norm,
                            self.relu])
            self.inplanes = channels

        return nn.Sequential(*modules)

    def forward(self, x):
        outs = []
        if self.arch == 'D':
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)
        x = self.layer5(x)
        outs.append(x)
        if self.layer6 is not None:
            x = self.layer6(x)
            outs.append(x)
        if self.layer7 is not None:
            x = self.layer7(x)
            outs.append(x)
        if self.layer8 is not None:
            x = self.layer8(x)
            outs.append(x)
        
        return tuple(outs)


def drn_d_54(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
    return model

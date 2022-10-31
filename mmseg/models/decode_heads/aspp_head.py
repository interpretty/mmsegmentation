# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

import numpy as np


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class ASPPHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(ASPPHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)

        # # aspp特征图展示
        # for feature_map in feats:
        #     # [N, C, H, W] -> [C, H, W]
        #     im = np.squeeze(feature_map.detach().cpu().numpy())
        #     # [C, H, W] -> [H, W, C]
        #     im = np.transpose(im, [1, 2, 0])
        #     # show top 12 feature maps
        #     for i in range(12):
        #         output_dir = 'D:/Yang/Py/mmsegmentation/checkpoints/BDVin/2000/temp/'
        #         output_file = output_dir + str(i) + '.jpg'
        #         cv2.imwrite(output_file, im[:, :, i] * 255.0)
        #
        #         ori_feat = cv2.imread(output_file, cv2.IMREAD_GRAYSCALE)
        #         # 二值化
        #         ret, binary = cv2.threshold(ori_feat, 200, 255, cv2.THRESH_BINARY)
        #         # 提轮廓
        #         contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #         # 计算轮廓面积
        #         keep_num = 0
        #         keep_list = []
        #         area_list = []
        #         mat_img = np.zeros(shape=ori_feat.shape, dtype=np.uint8)
        #         for j in contours:
        #             area = cv2.contourArea(j)
        #             if area > 10:
        #                 area_list.append(j)
        #             keep_list.append(area)
        #             keep_num = keep_num + 1
        #         if len(area_list) > 5:
        #             area_list = []
        #             y = sorted(keep_list)
        #             y.reverse()
        #             for j in contours:
        #                 area = cv2.contourArea(j)
        #                 if area > y[5]:
        #                     area_list.append(j)
        #         cv2.drawContours(mat_img, area_list, -1, (255, 255, 255), cv2.FILLED)
        #         contour_file = output_dir + str(i) + '_contour.jpg'
        #         cv2.imwrite(contour_file, mat_img)
        #
        #         #  绘制三区域
        #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素的形状和大小
        #         dst = cv2.dilate(mat_img, kernel)
        #         # 二值化
        #         ret, binary1 = cv2.threshold(dst, 200, 255, cv2.THRESH_BINARY)
        #         # 提轮廓
        #         contours1, hierarchy1 = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #         dst_file = output_dir + str(i) + '_dst.jpg'
        #         cv2.imwrite(dst_file, binary1)
        #         # 单通道转三通道
        #         mat_img_3 = np.zeros((mat_img.shape[0], mat_img.shape[1], 3), np.uint8, 'C')
        #         for m in range(mat_img.shape[0]):
        #             for n in range(mat_img.shape[1]):
        #                 mat_img_3[m, n][0] = mat_img[m, n]
        #                 mat_img_3[m, n][1] = mat_img_3[m, n][0]
        #                 mat_img_3[m, n][2] = mat_img_3[m, n][0]
        #         # 绘制轮廓
        #         cv2.drawContours(mat_img_3, contours1, -1, (0, 0, 255), 1)
        #         tri_contour_file = output_dir + str(i) + '_contour_tri.jpg'
        #         cv2.imwrite(tri_contour_file, mat_img_3)

        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)

        # im = np.squeeze(output.detach().cpu().numpy())
        # im = np.transpose(im, [1, 2, 0])
        # # show top 12 feature maps
        # for i in range(2):
        #     output_dir = 'D:/Yang/Py/mmsegmentation/checkpoints/BDVin/4000/temp/'
        #     output_file = output_dir + str(i) + '.jpg'
        #     cv2.imwrite(output_file, im[:, :, i] * 255.0)
        #
        #     ori_feat = cv2.imread(output_file, cv2.IMREAD_GRAYSCALE)
        #     # 二值化
        #     ret, binary = cv2.threshold(ori_feat, 20, 255, cv2.THRESH_BINARY)
        #     # 提轮廓
        #     contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     # 计算轮廓面积
        #     keep_num = 0
        #     keep_list = []
        #     area_list = []
        #     mat_img = np.zeros(shape=ori_feat.shape, dtype=np.uint8)
        #     for j in contours:
        #         area = cv2.contourArea(j)
        #         if area > 10:
        #             area_list.append(j)
        #         keep_list.append(area)
        #         keep_num = keep_num + 1
        #     if len(area_list) > 5:
        #         area_list = []
        #         y = sorted(keep_list)
        #         y.reverse()
        #         for j in contours:
        #             area = cv2.contourArea(j)
        #             if area > y[5]:
        #                 area_list.append(j)
        #     cv2.drawContours(mat_img, area_list, -1, (255, 255, 255), cv2.FILLED)
        #     contour_file = output_dir + str(i) + '_contour.jpg'
        #     cv2.imwrite(contour_file, mat_img)
        #
        #     #  绘制三区域
        #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素的形状和大小
        #     dst = cv2.dilate(mat_img, kernel)
        #     # 二值化
        #     ret, binary1 = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY)
        #     # 提轮廓
        #     contours1, hierarchy1 = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     dst_file = output_dir + str(i) + '_dst.jpg'
        #     cv2.imwrite(dst_file, binary1)
        #     # 单通道转三通道
        #     mat_img_3 = np.zeros((mat_img.shape[0], mat_img.shape[1], 3), np.uint8, 'C')
        #     for m in range(mat_img.shape[0]):
        #         for n in range(mat_img.shape[1]):
        #             mat_img_3[m, n][0] = mat_img[m, n]
        #             mat_img_3[m, n][1] = mat_img_3[m, n][0]
        #             mat_img_3[m, n][2] = mat_img_3[m, n][0]
        #     # 绘制轮廓
        #     cv2.drawContours(mat_img_3, contours1, -1, (0, 0, 255), 1)
        #     tri_contour_file = output_dir + str(i) + '_contour_tri.jpg'
        #     cv2.imwrite(tri_contour_file, mat_img_3)


        return output

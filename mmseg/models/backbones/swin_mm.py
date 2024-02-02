# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS
from ..utils.embed import PatchEmbed, PatchMerging

from mmengine.visualization import Visualizer

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # # # 查看attn
        # # # 1、储存位置
        # # attn_save_dir = 'C:/PY/mmsegmentation/checkpoints/vaihingen/unetformermm/tmp'
        # # # 创建attn的深拷贝
        # # attn_copy = attn.clone()
        # # # 指定保存路径
        # # save_path = attn_save_dir + '/attn_copy.pt'
        # # # 保存深拷贝到指定路径
        # # torch.save(attn_copy, save_path)
        #
        # # # 使用visualizer
        # # visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir=attn_save_dir)
        # # drawn_img = visualizer.draw_featmap(attn_copy[0], channel_reduction='squeeze_mean')
        # # visualizer.show(drawn_img)
        #
        # # 1、求swin中窗口的注意力比重
        # # （1）当前位置
        # ws = self.window_size[0]
        # # 假设attn_k是一个包含64个单个张量的列表
        # # 初始化一个变量来存储所有遍历得到的sum_attn_k
        # total_sum_attn_k = 0
        # count = 0
        # arr_1d_set = set()  # 创建一个空的集合
        # # （2）对应窗口的值
        # arr = np.zeros((8, 8))
        # attn_copy = attn.clone()
        # for batch in range(attn_copy.size(0)):
        #     for head in range(attn_copy.size(1)):
        #         for height in range(attn_copy.size(2)):
        #             # 计算数组的起始索引
        #             start_row = (height // (ws * 8)) * (ws * 8)
        #             start_col = (height % ws) // 8 * 8
        #             # 填充数组
        #             for m in range(8):
        #                 for n in range(8):
        #                     arr[m, n] = start_row + m * ws + start_col + n
        #             # 将arr展成一维
        #             arr_1d = arr.flatten()
        #             arr_1d_set.add(tuple(arr_1d))  # 将arr_1d转换为元组并添加到集合中
        #             # 生成新的一维数组attn_k
        #             attn_k = []
        #             for arr_ele in arr_1d:
        #                 # 根据arr_ele中的值取attn_copy第四维中的值
        #                 value = attn_copy[batch][head][height][int(arr_ele)]  # 这里的...表示前面的维度，具体根据你的张量维度来填写
        #                 attn_k.append(value)
        #             # print(attn_k)
        #             # 求和attn_k
        #             sum_attn_k = torch.stack(attn_k).sum()
        #             total_sum_attn_k += sum_attn_k
        #             count += 1
        # # 计算平均值
        # average_sum_attn_k = total_sum_attn_k / count
        # print("三层遍历完成后，每次遍历得到的sum_attn_k的平均值为:", average_sum_attn_k)
        # # 统计集合中元素的个数
        # num_unique_arr_1d = len(arr_1d_set)
        # print("总共有", num_unique_arr_1d, "种arr_1d")
        #
        # # 2、求优化后的注意力比重
        # total_sum_attn_k = 0
        # count = 0
        # arr_1d_set = set()  # 创建一个空的集合
        # for batch in range(attn_copy.size(0)):
        #     for head in range(attn_copy.size(1)):
        #         for height in range(attn_copy.size(2)):
        #
        #             # 1、计算当前宽高
        #             # 高由height // ws确定
        #             cur_row = height // ws
        #             # 宽由height % ws确定
        #             cur_col = height % ws
        #             # 2、判断高是否为上下位置
        #             if (cur_row - 3) < 0:
        #                 start_row = 0
        #             elif (cur_row + 6) > ws:
        #                 start_row = ws - 8
        #             else:
        #                 start_row = cur_row - 3
        #             # 3、判断宽是否为左右位置
        #             if (cur_col - 3) < 0:
        #                 start_col = 0
        #             elif (cur_col + 6) > ws:
        #                 start_col = ws - 8
        #             else:
        #                 start_col = cur_col - 3
        #             # 4、当前位置对应宽高数组
        #             for m in range(8):
        #                 for n in range(8):
        #                     arr[m, n] = start_row * ws + m * ws + start_col + n
        #             # 将arr展成一维
        #             arr_1d = arr.flatten()
        #             arr_1d_set.add(tuple(arr_1d))  # 将arr_1d转换为元组并添加到集合中
        #             # 生成新的一维数组attn_k
        #             attn_k = []
        #             for arr_ele in arr_1d:
        #                 # 根据arr_ele中的值取attn_copy第四维中的值
        #                 value = attn_copy[batch][head][height][int(arr_ele)]  # 这里的...表示前面的维度，具体根据你的张量维度来填写
        #                 attn_k.append(value)
        #             # print(attn_k)
        #             # 求和attn_k
        #             sum_attn_k = torch.stack(attn_k).sum()
        #             total_sum_attn_k += sum_attn_k
        #             count += 1
        # # 计算平均值
        # average_sum_attn_k = total_sum_attn_k / count
        # print("三层遍历完成后，每次遍历得到的sum_attn_k的平均值为:", average_sum_attn_k)
        # # 统计集合中元素的个数
        # num_unique_arr_1d = len(arr_1d_set)
        # print("总共有", num_unique_arr_1d, "种arr_1d")
        #
        # # 3、最大为1

        # 4、评价v的大小的影响

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

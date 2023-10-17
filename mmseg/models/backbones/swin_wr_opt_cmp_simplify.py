# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from mmengine.utils import to_2tuple
import math
import resources.swin_global as sw_global


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

    # init_weights的作用
    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, attn1=None, attn2=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        ws = self.window_size[0]
        B, N, C = x.shape
        # N为ws*ws，C统一为256，B为batch数量
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        # 生成相对位置特征
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if attn1 is None:
            attn = (q @ k.transpose(-2, -1))
            attn = attn + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            # attn对角线置为0
            diagonal_attn1 = torch.diagonal(attn, dim1=-2, dim2=-1)
            diagonal_attn1.fill_(0)
            attn1 = attn.contiguous()

        else:
            _, _, L, _ = attn1.shape

            # 1、求对角区域的attn
            # 以2为最小置换尺寸对q、k进行置换
            # 一次置换
            q = q.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                B, self.num_heads, ws // 2 * ws // 2, 4, -1)
            k = k.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                B, self.num_heads, ws // 2 * ws // 2, 4, -1)
            # 二次置换
            if attn2 is not None:
                q = q.view(B, self.num_heads, ws // 4, 2, ws // 4, 2, 4, -1).permute(0, 1, 2, 4, 3, 5, 6, 7).reshape(
                    B, self.num_heads, ws // 4 * ws // 4 * 2 * 2, 4, -1)
                k = k.view(B, self.num_heads, ws // 4, 2, ws // 4, 2, 4, -1).permute(0, 1, 2, 4, 3, 5, 6, 7).reshape(
                    B, self.num_heads, ws // 4 * ws // 4 * 2 * 2, 4, -1)
            # 求对角线区域的细化attn
            attn = q @ k.transpose(-2, -1)

            # 2、相对位置矩阵的分区域切分
            # 相对位置参数
            # 分区域切分
            # relative_position_bias进行permute符合原对应关系
            # 以2为最小置换尺寸进行置换
            # 一次置换
            relative_position_bias_per = relative_position_bias.contiguous().view(
                self.num_heads, ws // 2, 2, ws // 2, 2, ws // 2, 2, ws // 2, 2).permute(
                0, 1, 3, 2, 4, 5, 7, 6, 8).reshape(
                self.num_heads, ws // 2 * ws // 2, 4, ws // 2 * ws // 2, 4).permute(0, 1, 3, 2, 4)
            # 二次置换
            if attn2 is not None:
                relative_position_bias_per = relative_position_bias_per.reshape(
                    self.num_heads, ws // 4, 2, ws // 4, 2, 4, ws // 4, 2, ws // 4, 2, 4).permute(
                    0, 1, 3, 2, 4, 5, 6, 8, 7, 9, 10).reshape(
                    self.num_heads, ws // 4 * ws // 4 * 2 * 2, 4, ws // 4 * ws // 4 * 2 * 2, 4).permute(0, 1, 3, 2, 4)

            # 取relative_position_bias斜对角区域
            # Extract the diagonal blocks
            rpbp_diag = relative_position_bias_per.contiguous().diagonal(dim1=1, dim2=2)
            # Reshape the diagonal blocks tensor into the desired shape (N, 4, 4)
            rpbp_diag = rpbp_diag.reshape(self.num_heads, -1, 4, 4).unsqueeze(0)

            # 取relative_position_bias求取均值
            # 此处可以优化
            # rpbp_major1 = relative_position_bias_per.mean(dim=(3, 4))      # the wrong implementation
            rpbp_major = relative_position_bias_per.contiguous()
            diagonal_rpbp = torch.diagonal(rpbp_major, dim1=1, dim2=2)
            diagonal_rpbp.fill_(0)
            rpbp_major = rpbp_major.permute(0, 1, 3, 2, 4).reshape(self.num_heads, ws * ws, ws * ws).unsqueeze(0)

            # 3、分多个区域切分的attn
            # 第一层attn
            attn1

            # 第二层
            if attn2 is None:
                attn
            else:
                attn2
                # 第三层
                attn

            # 4、分别与relative_position_bias和attn相乘的v
            # 一次置换
            v = v.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 6, 2, 4, 3, 5).reshape(
                B, self.num_heads, -1, ws // 2 * ws // 2, 4)
            # 二次置换
            if attn2 is not None:
                # 需要进行转换
                v = v.view(B, self.num_heads, ws // 4, 2, ws // 4, 2, 4, -1).permute(
                    0, 1, 2, 4, 3, 5, 6, 7).reshape(B, self.num_heads, ws // 4 * ws // 4 * 2 * 2, 4, -1)

            if attn2 is None:
                # （1）先变形为基础形式，方便转换和求和
                v = v.view(B, self.num_heads, ws // 2 * ws // 2, 4, -1)
                # （2）与diag(attn与rpbp_diag)相乘的v
                v_diag = v.clone().reshape(B, self.num_heads, ws // 4 * ws // 4 * 2 * 2, 4, -1)
                # （3）与major(rpbp_major)相乘的v
                v_rpbp_major = v.clone().reshape(B, self.num_heads, ws // 4 * ws // 4 * 2 * 2 * 4, -1)
                # （4）与attn1相乘的v
                v_attn1 = torch.sum(v, dim=3, keepdim=False) / 4
            else:
                # （1）先变形为基础形式，方便转换和求和
                v = v.view(B, self.num_heads, ws // 4 * ws // 4, 2 * 2, 4, -1)
                # （2）与diag(attn与rpbp_diag)相乘的v
                v_diag = v.clone().reshape(B, self.num_heads, ws // 4 * ws // 4 * 2 * 2, 4, -1)
                # （3）与major(rpbp_major)相乘的v
                v_rpbp_major = v.clone().reshape(B, self.num_heads, ws // 4 * ws // 4 * 2 * 2 * 4, -1)
                # （4）与attn1相乘的v
                v_attn1 = torch.sum(v, dim=(3, 4), keepdim=False) / (4 * 4)
                # （5）与attn2相乘的v
                v_attn2 = torch.sum(v, dim=4, keepdim=False) / 4

            # 5、相乘
            # （1）
            attn_rpbp_diag = attn + rpbp_diag
            attn_rpbp_diag = self.softmax(attn_rpbp_diag)       # 仅能保证每次attention_map在归一化范围内


            # wrong:/(ws*ws) 为正则化，相当于每个attn_rpbp_diag元素对应大小是上一层次带下的1/16，但四个相加后为1/4，
            # 上一层次共(ws/2*ws/2)个元素和为1，故当前层次为1/(ws/2*ws/2)/16*4
            # right:每四个元素加起来，和应为1/(ws/2*ws/2),
            x_diag = (attn_rpbp_diag @ v_diag) / (ws // 2 * ws // 2)
            x_diag = x_diag.view(B, self.num_heads, ws * ws, -1)

            # （2）
            x_rpbp_major = rpbp_major @ v_rpbp_major

            # （3）
            x_attn1 = attn1 @ v_attn1

            # （4）
            if attn2 is not None:
                x_attn2 = attn2 @ v_attn2

            # 求和
            x = x_diag.contiguous().reshape(B, self.num_heads, L, -1, C // self.num_heads)\
                + x_rpbp_major.contiguous().reshape(B, self.num_heads, L, -1, C // self.num_heads)\
                + x_attn1.contiguous().reshape(B, self.num_heads, L, -1, C // self.num_heads)
            if attn2 is not None:
                x = x.contiguous().reshape(B, self.num_heads, L, 4, 4, C // self.num_heads)\
                    + x_attn2.contiguous().reshape(B, self.num_heads, L, 4, 1, C // self.num_heads)

            # 置换回去
            # 一次置换
            if attn2 is None:
                x = x.view(B, self.num_heads, ws // 2, ws // 2, 2, 2, C // self.num_heads).permute(
                    0, 2, 4, 3, 5, 1, 6).reshape(B, ws * ws, -1)
            # 两次置换
            else:
                x = x.view(B, self.num_heads, ws // 4, ws // 4, 2, 2, 2, 2, C // self.num_heads).permute(
                    0, 2, 4, 6, 3, 5, 7, 1, 8).reshape(B, ws * ws, -1)

            # 传导
            if attn2 is None:
                diagonal_attn2 = torch.diagonal(attn_rpbp_diag, dim1=1, dim2=2)
                diagonal_attn2.fill_(0)
                attn2 = attn_rpbp_diag.contiguous()

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn1, attn2

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
                 dropout_layer=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if dropout_layer is None:
            dropout_layer = dict(type='DropPath', drop_prob=0.)
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

    def forward(self, query, hw_shape, attn1=None, attn2=None):
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
        shifted_query = query
        attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows, attn1, attn2 = self.w_msa(query_windows, mask=attn_mask, attn1=attn1, attn2=attn2)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x, attn1, attn2

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

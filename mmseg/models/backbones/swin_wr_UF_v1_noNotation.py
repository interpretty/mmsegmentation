# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.model import BaseModule
from mmengine.utils import to_2tuple


class WindowMSA(BaseModule):
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
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, global_tuple=None):
        ws = self.window_size[0]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn2 = None
        if len(global_tuple) == 1:
            relative_position_bias = global_tuple[0]
            attn = (q @ k.transpose(-2, -1))
            attn = attn + relative_position_bias.unsqueeze(0)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            mask_side = self.window_size[0] * self.window_size[1]
            mask_diag = torch.ones(mask_side, mask_side, device='cuda:0')
            indices = torch.arange(mask_side)
            mask_diag[indices, indices] = 0
            attn1 = attn * mask_diag
            global_tuple.append(attn1)
        elif len(global_tuple) == 2:
            relative_position_bias, attn1 = global_tuple
            _, _, L, _ = attn1.shape
            q = q.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                B, self.num_heads, ws // 2 * ws // 2, 4, -1).contiguous()
            k = k.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                B, self.num_heads, ws // 2 * ws // 2, 4, -1).contiguous()
            attn = q @ k.transpose(-2, -1)
            v = v.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                B, self.num_heads, ws // 2 * ws // 2, 4, -1).contiguous()
            if attn2 is None:
                v_attn1 = torch.sum(v, dim=3, keepdim=False) / 4
                v_diag = v
            rpbp_diag = relative_position_bias.unsqueeze(1).unsqueeze(0)
            attn_rpbp_diag = attn + rpbp_diag
            attn_rpbp_diag = self.softmax(attn_rpbp_diag)  # 仅能保证每次attention_map在归一化范围内
            attn_rpbp_diag = attn_rpbp_diag / (ws // 2 * ws // 2)
            x_diag = attn_rpbp_diag @ v_diag
            x_diag = x_diag.view(B, self.num_heads, ws * ws, -1)
            x_attn1 = attn1 @ v_attn1
            x = x_diag.reshape(B, self.num_heads, L, -1, C // self.num_heads) \
                + x_attn1.reshape(B, self.num_heads, L, -1, C // self.num_heads)
            if attn2 is None:
                x = x.view(B, self.num_heads, ws // 2, ws // 2, 2, 2, C // self.num_heads).permute(
                    0, 2, 4, 3, 5, 1, 6).reshape(B, ws * ws, -1).contiguous()
            if attn2 is None:
                mask_side = 2 * 2
                mask_diag = torch.ones(mask_side, mask_side, device='cuda:0')
                indices = torch.arange(mask_side)
                mask_diag[indices, indices] = 0
                attn2 = attn_rpbp_diag * mask_diag
            global_tuple.append(attn2)
        else:
            relative_position_bias, attn1, attn2 = global_tuple
            _, _, L, _ = attn1.shape
            q = q.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                B, self.num_heads, ws // 2 * ws // 2, 4, -1).contiguous()
            k = k.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                B, self.num_heads, ws // 2 * ws // 2, 4, -1).contiguous()
            if attn2 is not None:
                q = q.view(B, self.num_heads, ws // 4, 2, ws // 4, 2, 4, -1).permute(0, 1, 2, 4, 3, 5, 6, 7).reshape(
                    B, self.num_heads, ws // 4 * ws // 4 * 2 * 2, 4, -1).contiguous()
                k = k.view(B, self.num_heads, ws // 4, 2, ws // 4, 2, 4, -1).permute(0, 1, 2, 4, 3, 5, 6, 7).reshape(
                    B, self.num_heads, ws // 4 * ws // 4 * 2 * 2, 4, -1).contiguous()
            attn = q @ k.transpose(-2, -1)
            v = v.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                B, self.num_heads, ws // 2 * ws // 2, 4, -1).contiguous()
            if attn2 is not None:
                v = v.view(B, self.num_heads, ws // 4, 2, ws // 4, 2, 4, -1).permute(
                    0, 1, 2, 4, 3, 5, 6, 7).reshape(B, self.num_heads, ws // 4 * ws // 4, 2 * 2, 4, -1).contiguous()
            if attn2 is None:
                pass
            else:
                v_attn1 = torch.sum(v, dim=(3, 4), keepdim=False) / (4 * 4)
                v_attn2 = torch.sum(v, dim=4, keepdim=False) / 4
                v_diag = v.reshape(B, self.num_heads, ws // 4 * ws // 4 * 2 * 2, 4, -1)
            rpbp_diag = relative_position_bias.unsqueeze(1).unsqueeze(0)
            attn_rpbp_diag = attn + rpbp_diag
            attn_rpbp_diag = self.softmax(attn_rpbp_diag)  # 仅能保证每次attention_map在归一化范围内
            attn_rpbp_diag = attn_rpbp_diag / (ws // 2 * ws // 2)
            x_diag = attn_rpbp_diag @ v_diag
            x_diag = x_diag.view(B, self.num_heads, ws * ws, -1)
            x_attn1 = attn1 @ v_attn1
            if attn2 is not None:
                x_attn2 = attn2 @ v_attn2
            x = x_diag.reshape(B, self.num_heads, L, -1, C // self.num_heads) \
                + x_attn1.reshape(B, self.num_heads, L, -1, C // self.num_heads)
            if attn2 is not None:
                x = x.reshape(B, self.num_heads, L, 4, 4, C // self.num_heads) \
                    + x_attn2.reshape(B, self.num_heads, L, 4, 1, C // self.num_heads)
            if attn2 is not None:
                x = x.view(B, self.num_heads, ws // 4, ws // 4, 2, 2, 2, 2, C // self.num_heads).permute(
                    0, 2, 4, 6, 3, 5, 7, 1, 8).reshape(B, ws * ws, -1).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, global_tuple


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

    def forward(self, query, hw_shape, global_tuple=None):
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
        attn_windows, global_tuple = self.w_msa(query_windows, mask=attn_mask, global_tuple=global_tuple)

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
        return x, global_tuple

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

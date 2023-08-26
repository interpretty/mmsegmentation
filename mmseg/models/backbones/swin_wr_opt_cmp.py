# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from mmengine.utils import to_2tuple


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

    def forward(self, x, mask=None, sim_map=None):
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

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # relative_position_bias进行permute符合原对应关系
        relative_position_bias_per = relative_position_bias.contiguous().view(
            self.num_heads, ws // 2, 2, ws // 2, 2, ws // 2, 2,ws // 2, 2).permute(
            0, 1, 3, 2, 4, 5, 7, 6, 8).reshape(
            -1, ws // 2 * ws // 2, 4, ws // 2 * ws // 2, 4).permute(0, 1, 3, 2, 4)

        # 取relative_position_bias斜对角区域
        # Extract the diagonal blocks
        diagonal_blocks = relative_position_bias_per.diagonal(dim1=1, dim2=2)
        # Reshape the diagonal blocks tensor into the desired shape (N, 4, 4)
        diagonal_blocks = diagonal_blocks.reshape(self.num_heads, -1, 4, 4)

        # 取relative_position_bias求取均值
        # Calculate the average of each block
        block_averages = relative_position_bias_per.mean(dim=(3, 4))
        # Reshape the block averages tensor into shape (N, M/4, M/4)
        block_averages = block_averages.reshape(self.num_heads, ws // 2 * ws // 2, ws // 2 * ws // 2)

        if sim_map is not None:
            _, _, L, _ = sim_map.shape

            # q、k分块
            q = q.view(B, self.num_heads, ws, ws, -1)
            k = k.view(B, self.num_heads, ws, ws, -1)
            q = q.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6)
            k = k.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6)
            q = q.reshape(B, self.num_heads, ws // 2 * ws // 2, 4, -1)
            k = k.reshape(B, self.num_heads, ws // 2 * ws // 2, 4, -1)

            attn = q @ k.transpose(-2, -1)

            # 令attn、sim_map分别与v相乘
            # 1、sim_map与v相乘
            # （1）v聚合
            v_sep = v.clone().view(B, self.num_heads, ws, ws, -1)
            v_sep = v_sep.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6)
            v_sep = v_sep.reshape(B, self.num_heads, ws // 2 * ws // 2, 4, -1)
            v_sep = torch.mean(v_sep, -2)
            # （2）sim_map加入位置因素
            attn_sim = sim_map.contiguous()
            # attn_sim = sim_map + block_averages.unsqueeze(0)

            # 令attn_sim对角线为0
            attn_sim = attn_sim.reshape(B, self.num_heads, ws // 2 * ws // 2, ws // 2 * ws // 2).clone()
            diagonal = torch.diagonal(attn_sim, dim1=-2, dim2=-1)
            diagonal.fill_(0)
            attn_sim = attn_sim.reshape(B, self.num_heads, ws // 2 * ws // 2, ws // 2 * ws // 2)

            attn_sim = attn_sim @ v_sep

            # 2、attn与v相乘
            # （1）v进行permute
            v_diag = v.view(B, self.num_heads, ws // 2, 2, ws // 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6)
            v_diag = v_diag.reshape(B, self.num_heads, ws // 2 * ws // 2, 4, -1)
            # （2）attn加入位置因素
            # attn_v = attn + diagonal_blocks.unsqueeze(0)
            attn_v = attn.contiguous()

            attn_v = attn_v @ v_diag

            # 3、合并attn_sim与attn_v

            # 扩张attn_sim
            # 不扩张，直接相加
            attn_sim = attn_sim.permute(0, 1, 3, 2)  # B, self.num_heads, features, ws // 2 * ws // 2
            attn_v = attn_v.permute(0, 1, 4, 2, 3)  # B, self.num_heads, features, ws // 2 * ws // 2, 4
            attn_merge = attn_sim.unsqueeze(-1) + attn_v
            attn_merge = attn_merge.permute(0, 1, 3, 4, 2)
            attn_merge = attn_merge.reshape(B, self.num_heads, ws // 2, ws // 2, 2, 2, -1).permute(0, 2, 4, 3, 5, 1, 6)
            attn_merge = attn_merge.reshape(B, ws * ws, -1)

            # sim_map扩建，通过expand函数实现
            sim_map = sim_map.unsqueeze(-1).unsqueeze(-1)
            sim_map = sim_map.expand(-1, -1, -1, -1, 4, 4) / 4

            # sim_map填充
            # Get the shapes of sim_map1 and attn1
            s1, s2, s3, _, s4, s5 = sim_map.shape
            # Reshape sim_map1 and attn1 to match the required dimensions
            sim_map = sim_map.view(s1 * s2 * s3 * s3, s4 * s5)
            attn = attn.view(s1 * s2 * s3, s4 * s5)
            # Create indices for indexing into sim_map1
            indices_0 = torch.arange(0, s3 * s3, s3 + 1).cuda()
            indices_1 = torch.cat([indices_0 + i * s3 * s3 for i in range(s1 * s2)]).cuda()
            # Use index_copy_ to update sim_map1 with attn1 values
            sim_map.index_copy_(0, indices_1, attn)
            # Reshape sim_map1 back to its original shape
            sim_map = sim_map.view(s1, s2, s3, s3, s4, s5)

            attn = []
            attn = sim_map.permute(0, 1, 2, 4, 3, 5)
            attn = attn.reshape(B, self.num_heads, ws // 2, ws // 2, 2, 2, ws // 2, ws // 2, 2, 2)
            attn = attn.permute(0, 1, 2, 4, 3, 5, 6, 8, 7, 9)
            attn = attn.reshape(B, self.num_heads, L * 4, L * 4)

            # attn = attn + relative_position_bias.unsqueeze(0)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            # 判断attn_merge和是否一致
            are_equal = torch.equal(attn_merge, x)

            # difference = attn_merge - x
            # Convert the difference tensor to a numpy array
            numpy_array1 = attn_merge.cpu().detach().numpy()
            numpy_array2 = x.cpu().detach().numpy()
            import pandas as pd
            # Create DataFrames from the numpy arrays
            df1 = pd.DataFrame(numpy_array1[0,:,:])
            df2 = pd.DataFrame(numpy_array2[0,:,:])
            # Save the DataFrames as Excel files
            df1.to_excel(r'C:\Users\WIN\Desktop\array1.xlsx', index=False)
            df2.to_excel(r'C:\Users\WIN\Desktop\array2.xlsx', index=False)

            print(are_equal)


        else:
            attn = (q @ k.transpose(-2, -1))

        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

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

    def forward(self, query, hw_shape, sim_map=None):
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
        attn_windows, sim_map = self.w_msa(query_windows, mask=attn_mask, sim_map=sim_map)

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
        return x, sim_map

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

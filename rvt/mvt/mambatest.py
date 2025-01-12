# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from rvt.diffuser_actor.utils.layers import ParallelAttention, FFWRelativeCrossAttentionModule
from rvt.diffuser_actor.utils.position_encodings import SinusoidalPosEmb, RotaryPositionEncoding, \
    RotaryPositionEncoding3D
from rvt.diffuser_actor.utils.Hilbert3d import Hilbert3d
from rvt.diffuser_actor.utils.mambablock import MambaLayerglobal, MambaLayerlocal, MambaTotalBlock
import rvt.mvt.utils as mvt_utils
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}


class CustomDETRHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super(CustomDETRHead, self).__init__()

        # 3D point 的预测头，输出 3 个值 (x, y, z)
        self.bbox_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            # nn.Sigmoid()  # 将边界框归一化到 [0, 1]
        )

        # 四个像素坐标的预测头，每个坐标使用一个线性层，将 hidden_dim 转换成 2 个值 (x, y)
        # 使用 sigmoid 激活使像素坐标归一化到 [0, 1] 范围
        self.pixel_coord_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
                nn.Sigmoid()  # 将坐标归一化到 [0, 1]
            ) for _ in range(4)
        ])

    def forward(self, fusion_output):
        """
        输入:
        - fusion_output: [batch_size, 5, hidden_dim] 的特征张量

        输出:
        - bbox_preds: [batch_size, 6]，3D 边界框预测结果
        - pixel_coords_preds: [batch_size, 4, 2]，四个像素坐标的预测结果
        """

        # 选择第一个位置的特征，用于 3D 边界框预测
        bbox_preds = self.bbox_layer(fusion_output[:, 0, :])  # [batch_size, 6]

        # 选择后四个位置的特征，用于四个像素坐标预测
        pixel_coords_preds = torch.stack([
            layer(fusion_output[:, i, :]) for i, layer in enumerate(self.pixel_coord_layers, start=1)
        ], dim=1)  # [batch_size, 4, 2]

        return bbox_preds, pixel_coords_preds


class HeatmapPredictor(nn.Module):
    def __init__(self, input_channels=192, output_channels=4, target_size=(128, 128)):
        super(HeatmapPredictor, self).__init__()

        # 第一个卷积层，减少通道数并扩展空间维度
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 从 4x4 -> 8x8

        # 第二个卷积层，进一步减少通道数
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 从 8x8 -> 16x16

        # 第三个卷积层，继续减少通道数
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 从 16x16 -> 64x64

        # 最终卷积层，输出为 4 个 heatmap
        self.conv4 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(output_channels)
        self.upsample4 = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)  # 64x64 -> 128x128

    def forward(self, x):
        # 输入形状 [bs, 4, 4, 192]，先转换成 [bs, 192, 4, 4]
        x = x.permute(0, 3, 1, 2)  # 将通道放在第二维

        # 1. 第一次卷积和上采样，从 4x4 -> 8x8
        x = self.upsample1(torch.relu(self.bn1(self.conv1(x))))  # [bs, 128, 8, 8]

        # 2. 第二次卷积和上采样，从 8x8 -> 16x16
        x = self.upsample2(torch.relu(self.bn2(self.conv2(x))))  # [bs, 64, 16, 16]

        # 3. 第三次卷积和上采样，从 16x16 -> 64x64
        x = self.upsample3(torch.relu(self.bn3(self.conv3(x))))  # [bs, 32, 64, 64]

        # 4. 最终卷积层输出 heatmap，从 64x64 -> 128x128
        x = self.upsample4(torch.relu(self.bn4(self.conv4(x))))  # [bs, 4, 128, 128]

        return x


class MyMambaPipeline(nn.Module):
    def __init__(self, num_features=192, num_frames=4, num_camera=4, img_height=128, img_width=128, patch_size=16,
                 depth_of_rgb_mamba=24, depth_of_pcd_mamba=24, depth_rgb=4, depth_pcd=4, depth_fusionatten=4, depth_crossatten=4, fusion_layer_num=4):
        super().__init__()
        self.model_feature_dim = num_features
        self.num_his_frames = num_frames
        self.num_camera = num_camera
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        assert (self.img_height == self.img_width), "w should equal to h"
        self.num_patch = self.img_height // patch_size
        self.RGB_mamba_block = VisionMamba(
            patch_size=patch_size,
            embed_dim=num_features,
            depth=depth_of_rgb_mamba,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            num_frames=num_frames*num_camera).cuda()

        self.pcd_mamba_block = VisionMamba(
            patch_size=patch_size,
            embed_dim=num_features,
            depth=depth_of_pcd_mamba,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            num_frames=num_frames * num_camera).cuda()

        #######################
        self.h_l = torch.tensor(self.hilbert_curve_large_scale()["hilbert_curve_large_scale"]).to("cuda")  # [256]
        self.mamba_blocks_rgb = nn.ModuleList(
            [MambaTotalBlock(dim=num_features, total_layers_num=depth_rgb) for _ in range(self.num_camera)]
        )
        self.mamba_blocks_pcd = nn.ModuleList(
            [MambaTotalBlock(dim=num_features, total_layers_num=depth_pcd) for _ in range(self.num_camera)]
        )
        self.rgb_cross_atten2each_came = nn.ModuleList(
            [FFWRelativeCrossAttentionModule(num_features, 8, depth_crossatten) for _ in range(self.num_camera)]
        )
        self.pcd_cross_atten2each_came = nn.ModuleList(
            [FFWRelativeCrossAttentionModule(num_features, 8, depth_crossatten) for _ in range(self.num_camera)]
        )

        self.fusion_layer_num = fusion_layer_num
        self.fusion_cross_atten = nn.ModuleList(
            [FFWRelativeCrossAttentionModule(num_features, 8, depth_fusionatten) for _ in range(self.fusion_layer_num * 2)]
        )

        self.pre_head = CustomDETRHead(num_features).cuda()

        self.learnable_query = nn.Parameter(torch.randn(1, 1, self.model_feature_dim))

        self.abs_emb = SinusoidalPosEmb(self.model_feature_dim)
        self.rot_emb_3D = RotaryPositionEncoding3D(self.model_feature_dim)
        self.rot_emb_2D = RotaryPositionEncoding(self.model_feature_dim)

        #######################
        # self.conv1d_on_camera_dim = nn.Conv1d(in_channels=self.num_camera * 2, out_channels=1, kernel_size=1).cuda()
        #
        # self.fusion_attention = ParallelAttention(
        #     num_layers=1,
        #     d_model=192, n_heads=8,
        #     self_attention1=True, self_attention2=True,
        #     cross_attention1=True, cross_attention2=True,
        #     rotary_pe=False, apply_ffn=False
        # ).cuda()  # todo 可选是否用旋转编码
        #
        # self.lang_cross_atten = FFWRelativeCrossAttentionModule(192, 8, 2).cuda()
        #
        # self.linear1 = nn.Linear(512, 192).cuda()
        # # 64 * 64 * 128
        # self.GlobalMambaBlock1 = MambaLayerglobal(dim=num_features)
        # self.LocalMambaBlock1 = MambaLayerlocal(dim=num_features)
        # self.GlobalMambaBlock2 = MambaLayerglobal(dim=num_features)
        # self.LocalMambaBlock2 = MambaLayerlocal(dim=num_features)
        #
        # self.conv_down_sample = nn.Conv3d(192, 192 * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2),
        #                                   padding=(1, 1, 1)).cuda()
        #
        # self.GlobalMambaBlockLowRes1 = MambaLayerglobal(dim=num_features * 2)
        # self.LocalMambaBlockLowRes1 = MambaLayerlocal(dim=num_features * 2)
        # self.GlobalMambaBlockLowRes2 = MambaLayerglobal(dim=num_features * 2)
        # self.LocalMambaBlockLowRes2 = MambaLayerlocal(dim=num_features * 2)
        #
        # self.conv_down_sample2 = nn.Conv3d(192 * 2, 192, kernel_size=(4, 3, 3), stride=(4, 1, 1),
        #                                    padding=(0, 1, 1)).cuda()
        # self.head = HeatmapPredictor().cuda()

        self.proprio_dim = 4
    def forward(
            self, input
    ):
        """
        vis_img_feat: [bs, channel, nframe*ncam, H, W]
        vis_pc_feat: [bs, channel, nframe*ncam, H, W]
        lang_goal_embs:[bs, 77, 512]
        """

        torch.autograd.set_detect_anomaly(True)

        vis_img_feat, vis_pc_feat, lang_goal_embs = input
        bs = vis_img_feat.shape[0]

        pc_feat = self.scale_point_cloud_features(vis_pc_feat.transpose(1, 2).flatten(start_dim=0, end_dim=1))
        pc_rel_pe = self.rot_emb_3D(pc_feat.flatten(start_dim=1, end_dim=3)).view(bs, self.num_his_frames,
                                                                                  self.num_camera, self.num_patch,
                                                                                  self.num_patch,
                                                                                  self.model_feature_dim, 2)
        abs_pe = self.abs_emb(
            torch.arange(0, self.num_his_frames * self.num_camera * self.num_patch * self.num_patch,
                         device=vis_img_feat.device)
        )[None].repeat(bs, 1, 1).view(bs, self.num_his_frames,
                                      self.num_camera, self.num_patch,
                                      self.num_patch,
                                      self.model_feature_dim)

        rgb_mamba_out = self.RGB_mamba_block(vis_img_feat)  #

        pcd_mamba_out = self.pcd_mamba_block(
            vis_pc_feat)  # [bs, ch, 4*his, 128, 128] -> [bs, dim, n_hisframe*n_cam, height/patch_size, height/patch_size] -> [bs, seq_len, dim]  seq_len = n_cam * n_hisframe * height/patch_size * height/patch_size

        # pos_emb = self.abs_emb(torch.arange(0, pcd_mamba_out.size(1), device=pcd_mamba_out.device))[None].repeat(
        #     len(pcd_mamba_out), 1, 1)  # [bs, seq_len, feature_dim]

        # rel_pcd_pe = self.rot_emb_3D(vis_pc_feat.flatten(start_dim=2).transpose(1, 2)[:,:100,:])
        # rel_pcd_pe = self.rot_emb_2D(rgb_mamba_out[:, :96, 1])  # torch.Size([4, 96, 2, 2])
        # rel_rgb_pe = self.rot_emb_2D(vis_img_feat)

        # seq1, seq2 = self.fusion_attention(
        #     seq1=rgb_mamba_out, seq1_key_padding_mask=None,
        #     seq2=pcd_mamba_out, seq2_key_padding_mask=None,
        #     seq1_pos=None, seq2_pos=None,
        #     seq1_sem_pos=pos_emb, seq2_sem_pos=pos_emb
        # )  # [4, 256, 192]
        #
        # x = torch.cat((seq1.view(bs, self.num_frames, self.num_camera, self.num_patch, self.num_patch, self.model_feature_dim),
        #                seq2.view(bs, self.num_frames, self.num_camera, self.num_patch, self.num_patch, self.model_feature_dim)),
        #               dim=2)  # torch.Size([bs, num_his, pcd_cam+img_cam , num_patch, num_patch, dim])


        # RGB branch
        RGB_cam_features = []
        latest_rbg_frame = torch.zeros(bs, self.model_feature_dim, self.num_patch, self.num_patch).to("cuda")
        rgb_feat_1 = rgb_mamba_out.view(bs, self.num_his_frames, self.num_camera, self.num_patch, self.num_patch,
                                        self.model_feature_dim).permute(0, 1, 2, 5, 3, 4)  # B, nf, nc, C, H, W
        for cam_idx in range(self.num_camera):
            cam_feature = self.mamba_blocks_rgb[cam_idx](
                rgb_feat_1[:, :, cam_idx, :, :, :], self.h_l)  # input x: B, nf, C, H, W output: x: B, nf, C, H, W

            cam_feature += abs_pe[:, :, cam_idx].permute(0, 1, 4, 2, 3)

            latest_rbg_frame += cam_feature[:, -1]  # [B, D, H, W] todo

            rgb_cross_atten_output = \
                self.rgb_cross_atten2each_came[cam_idx](query=latest_rbg_frame.flatten(start_dim=2).permute(2, 0, 1),
                                                        value=cam_feature.permute(2, 0, 1, 3, 4).flatten(
                                                            start_dim=2).transpose(0, 2),
                                                        query_pos=pc_rel_pe[:, -1, cam_idx].flatten(start_dim=1,
                                                                                                    end_dim=2),
                                                        value_pos=pc_rel_pe[:, :, cam_idx].flatten(start_dim=1,
                                                                                                   end_dim=3)) \
                    [-1].view(self.num_patch, self.num_patch, bs, self.model_feature_dim).permute(2, 3, 0, 1)

            latest_rbg_frame += rgb_cross_atten_output  # [B, D, H, W] todo
            RGB_cam_features.append(rgb_cross_atten_output.flatten(start_dim=2).transpose(1, 2))  # [[B, L, D]]

        RGB_cam_features_for_fusion = torch.concat(RGB_cam_features, dim=1)  # [b, ncam*l ,d]

        # pcd branch
        pcd_cam_features = []
        latest_pcd_frame = torch.zeros(bs, self.model_feature_dim, self.num_patch, self.num_patch).to("cuda")
        pcd_feat_1 = pcd_mamba_out.view(bs, self.num_his_frames, self.num_camera, self.num_patch, self.num_patch,
                                        self.model_feature_dim).permute(0, 1, 2, 5, 3, 4)  # B, nf, nc, C, H, W
        for cam_idx in range(self.num_camera):
            cam_feature = self.mamba_blocks_pcd[cam_idx](
                pcd_feat_1[:, :, cam_idx, :, :, :], self.h_l)

            cam_feature += abs_pe[:, :, cam_idx].permute(0, 1, 4, 2, 3)
            latest_pcd_frame += cam_feature[:, -1]  #

            pcd_cross_atten_output = \
                self.pcd_cross_atten2each_came[cam_idx](query=latest_pcd_frame.flatten(start_dim=2).permute(2, 0, 1),
                                                        value=cam_feature.permute(2, 0, 1, 3, 4).flatten(
                                                            start_dim=2).transpose(0, 2),
                                                        query_pos=pc_rel_pe[:, -1, cam_idx].flatten(start_dim=1,
                                                                                                    end_dim=2),
                                                        value_pos=pc_rel_pe[:, :, cam_idx].flatten(start_dim=1,
                                                                                                   end_dim=3)) \
                    [-1].view(self.num_patch, self.num_patch, bs, self.model_feature_dim).permute(2, 3, 0, 1)
            latest_pcd_frame += pcd_cross_atten_output  # [B, D, H, W] todo
            pcd_cam_features.append(pcd_cross_atten_output.flatten(start_dim=2).transpose(1, 2))  # [[B, L, D]]
        pcd_cam_features_for_fusion = torch.concat(pcd_cam_features, dim=1)  # [b, ncam*l ,d]

        # fusion part
        fusion_query = self.learnable_query.repeat(bs, 5, 1)  # [b, 5, d]
        mask = torch.zeros(bs, fusion_query.shape[1], RGB_cam_features_for_fusion.shape[
            1])  # attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        pre_3D_bbox_list = []
        pre_points_list = []
        for idx in range(self.fusion_layer_num):
            fusion_output = self.fusion_cross_atten[idx](query=fusion_query.transpose(0, 1),
                                                         value=RGB_cam_features_for_fusion.transpose(0, 1),attn_mask=mask)[
                -1].transpose(0, 1)

            pre_3D_bbox, pre_pixel_points = self.pre_head(fusion_output)  # pre_res:[bs, list[6个数 3dbbox,4个像素点]]
            mask = self.generate_mask(pre_pixel_points, mask.shape[1:])
            pre_3D_bbox_list.append(pre_3D_bbox)
            pre_pixel_points = pre_pixel_points*self.img_height
            pre_points_list.append(pre_pixel_points)

            fusion_output = self.fusion_cross_atten[idx + 1](query=fusion_query.transpose(0, 1),
                                                             value=pcd_cam_features_for_fusion.transpose(0, 1),attn_mask=mask)[
                -1].transpose(0, 1)

            pre_3D_bbox, pre_pixel_points = self.pre_head(fusion_output)
            mask = self.generate_mask(pre_pixel_points, mask.shape[1:])
            pre_3D_bbox_list.append(pre_3D_bbox)
            pre_pixel_points = pre_pixel_points * self.img_height
            pre_points_list.append(pre_pixel_points)

        return  [pre_3D_bbox_list[-2], pre_3D_bbox_list[-1]], [pre_points_list[-2], pre_points_list[-1]]

        # # 一维卷积
        #
        # x = x.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2)  # [4, 8, 4, 8, 8, 192]
        # x = self.conv1d_on_camera_dim(x)
        # x = x.view(bs, 1, 4, 8, 8, 192).squeeze(1).permute(0, 1, 4, 2, 3)  # ->[bs, num_his, dim, num_patch, num_patch]
        #
        # # lang_goal_embs
        # x = x.permute(2, 0, 1, 3, 4).flatten(start_dim=2).transpose(0, 2)  # [dim, bs, num_his, num_patch, num_patch]->[dim, bs, num_his*num_patch*num_patch]-> [L, bs, dim]
        # lang_goal_embs = self.linear1(lang_goal_embs)  # [bs, 77, 512]->[bs, 77, 192]
        # x = self.lang_cross_atten(query=x, value=lang_goal_embs.transpose(0, 1))[-1]  # ->[seq_len, bs, dim]
        # x = x.transpose(0, 1).view(bs, 4, 8, 8, 192).permute(0, 1, 4, 2, 3)  # [seq_len, bs, dim] -> [bs, nframe, dim, npatch, npatch]
        #
        # # global + local mamba
        #
        # h_l = torch.tensor(self.hilbert_curve_large_scale()["hilbert_curve_large_scale"]).to("cuda")  # [256]
        # h_s = torch.tensor(self.hilbert_curve_small_scale()["hilbert_curve_small_scale"]).to("cuda")  # [64]
        #
        # M1 = self.GlobalMambaBlock1(x)  # ->[bs, nframe, dim, num_patch, num_patch]
        # M1 = self.LocalMambaBlock1(M1, h_l)
        # M1 = self.GlobalMambaBlock2(M1)
        # M1 = self.LocalMambaBlock2(M1, h_l)  # ->[bs, nframe, dim, num_patch, num_patch]
        #
        # x_down = rearrange(M1, 'n d c h w ->  n c d h w')
        # x_down = torch.nn.functional.relu(self.conv_down_sample (x_down))
        # x_down = rearrange(x_down, 'n c d h w ->  n d c h w')  # ->[bs, nframe, dim*2, num_patch/2, num_patch/2]
        #
        # M2 = self.GlobalMambaBlockLowRes1(x_down)
        # M2 = self.LocalMambaBlockLowRes1(M2, h_s)
        # M2 = self.GlobalMambaBlockLowRes2(M2)
        # M2 = self.LocalMambaBlockLowRes2(M2, h_s)
        #
        #
        # x_re = rearrange(M2, 'n d c h w ->  n c d h w')
        # x_re = self.conv_down_sample2(x_re).squeeze(2)
        # x_re = rearrange(x_re, 'n c h w ->  n h w c')  # ->[bs, num_patch/2, num_patch/2, dim]
        #
        # hm = self.head(x_re)  # ->[bs,ncamera, H, W]
        # return hm

    def scale_point_cloud_features(self, pcd):
        """
        缩放并重新排列点云特征。

        参数:
        - pcd: 点云特征张量，形状为 (batch_size * num_cameras, channels, original_height, original_width)
        - target_height: 缩放后的高度
        - target_width: 缩放后的宽度
        - num_cameras: 相机数量

        返回:
        - 缩放并重新排列后的点云特征张量，形状为 (batch_size, num_cameras * target_height * target_width, channels)
        """

        # 使用双线性插值缩放到目标宽高
        pcd_scaled = torch.nn.functional.interpolate(pcd, size=(self.num_patch, self.num_patch), mode='bilinear')

        # 重新排列点云特征
        pcd_rearranged = rearrange(
            pcd_scaled,
            "(bt nframe) c h w -> bt nframe h w c",
            nframe=self.num_camera * self.num_his_frames
        )

        return pcd_rearranged

    def generate_mask(self, coord_list, attention_mask_size, neighborhood_size=1):
        # coord像素坐标换算patch位置
        # 计算该patch邻域位置
        # 生成对应的attention mask,将对应patch的位置给与更高权重，patch顺序为按行展开
        """
        生成 attention mask，将目标 patch 及其邻域位置赋予更高权重。

        参数:
        - coord: tuple，目标像素坐标 (x, y)。
        - attention_mask_size: int，attention mask 的尺寸 (假设为方形)。
        - patch_size: int，每个 patch 的尺寸，默认为 16。
        - neighborhood_size: int，邻域范围，默认为 1，表示包括目标 patch 和其一阶邻域。
        - high_weight: int，目标 patch 和邻域的权重值，默认为 10。

        返回:
        - mask: Tensor，attention mask，大小为 [attention_mask_size, attention_mask_size]。
        """
        # 初始化 mask，所有位置权重默认为 1
        batch_size = coord_list.shape[0]
        mask = torch.zeros(batch_size, attention_mask_size[0], attention_mask_size[1])
        high_weight = 1
        switch2index = lambda x, y, cam_id: y * self.num_patch + x + cam_id * self.num_patch ** 2
        #
        # import pdb
        # pdb.set_trace()
        for bs in range(batch_size):
            for idx, coord in enumerate(coord_list[bs]):
                # 将像素坐标转化为 patch 坐标
                patch_x, patch_y = int(coord[0] * self.img_height // self.patch_size), int(
                    coord[1] * self.img_height // self.patch_size)

                # 将 patch 坐标展开为线性索引

                # 确定邻域 patch 的范围
                for i in range(-neighborhood_size, neighborhood_size + 1):
                    for j in range(-neighborhood_size, neighborhood_size + 1):
                        # 计算邻域 patch 的坐标
                        neighbor_x, neighbor_y = patch_x + i, patch_y + j

                        # 确保邻域 patch 坐标在有效范围内
                        if 0 <= neighbor_x < self.num_patch and 0 <= neighbor_y < self.num_patch:
                            # 计算邻域 patch 的线性索引
                            neighbor_index = switch2index(neighbor_x, neighbor_y, idx)

                            # 将邻域 patch 的位置赋予高权重
                            mask[bs, :, neighbor_index] = high_weight

        return mask

    def hilbert_curve_large_scale(self, ):
        # B, nf, C, H, W = x.shape

        nf = self.num_his_frames
        H = self.num_patch
        W = self.num_patch

        hilbert_curve = list(
            Hilbert3d(width=H, height=W, depth=nf))
        hilbert_curve = torch.tensor(hilbert_curve).long()
        hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

        return {
            'hilbert_curve_large_scale': hilbert_curve
        }

    def hilbert_curve_small_scale(self, ):
        # B, nf, C, H, W = x.shape

        nf = 4
        H = 4
        W = 4

        hilbert_curve = list(
            Hilbert3d(width=H, height=W, depth=nf))
        hilbert_curve = torch.tensor(hilbert_curve).long()
        hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

        return {
            'hilbert_curve_small_scale': hilbert_curve
        }

    def save_hilbert_curve_large_scale(self, hilbert_curve_large_scale,
                                       filename='./Hilbert/hilbert_curve_large_scale.pt'):
        torch.save(hilbert_curve_large_scale, filename)

    def save_hilbert_curve_small_scale(self, hilbert_curve_small_scale,
                                       filename='./Hilbert/hilbert_curve_small_scale.pt'):
        torch.save(hilbert_curve_small_scale, filename)

    @torch.no_grad()
    def get_action_trans_single_pcd(self, pre_point, target_point, cam_pcd, keyframe_pic, img_width, img_height, if_vis=False):
        """
        计算与target point欧式距离最近的campcd中的点是哪一个，并返回其图像坐标
        target_point (torch.Tensor): 世界坐标系下的点 [3]
        cam_pcd: [3,128,128]
        img_width (int): 图像的宽度
        img_height (int): 图像的高度
        """
        # 计算 target_point 与 cam_pcd 每个点的欧式距离
        # 将 target_point 扩展成 [3, img_width, img_height] 以匹配 cam_pcd 的形状
        target_point_expanded = target_point.view(3, 1, 1).expand_as(cam_pcd)

        # 计算距离的平方
        dist_squared = torch.sum((cam_pcd - target_point_expanded) ** 2, dim=0)  # [img_width, img_height]

        # 找到最小距离的索引
        min_dist_index = torch.argmin(dist_squared)

        # 将线性索引转换为图像坐标 (x, y)
        y, x = divmod(min_dist_index.item(), img_width)  # 注意：这里的顺序为 (y, x)，因为 torch.argmin 返回的索引是按行展开的

        img = keyframe_pic.permute(1, 2, 0).numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
        img = (img * 255).astype(np.uint8)  # 转为 [0, 255] 范围

        # 绘制点
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转为 BGR
        cv2.circle(img, (0, 128), 2, (255, 0, 0), -1)  # blue: y max
        cv2.circle(img, (128, 0), 2, (0, 255, 0), -1)  # green: x max
        cv2.circle(img, (0, 0), 2, (0, 0, 0), -1)  # black: origin
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)  # red: projected point
        cv2.circle(img, (int(pre_point[0]), int(pre_point[1])), 5, (0, 255, 255), -1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 调整为 [C, H, W]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        # 归一化到 [0, 1]
        img_tensor /= 255.0
        

        # 显示图片（如果需要）
        if if_vis:
            cv2.imshow("Projected Point on Image", img)
            cv2.waitKey(2)
            cv2.destroyAllWindows()

        return torch.tensor([x, y]), img_tensor

    @torch.no_grad()
    def get_action_trans_single(self, target_point, cam_info_ex, cam_info_in, keyframe_pic, img_width, img_height):
        """
        计算世界坐标系下的单个点在单张图像中的像素位置，处理超出图像范围的情况
        Args:
            target_point (torch.Tensor): 世界坐标系下的点 [3]
            cam_info (list): 包含相机的外参 [ 4, 4] 和内参 [3, 3]
            keyframe_pic ：[img_width,img_height,channel]
            img_width (int): 图像的宽度
            img_height (int): 图像的高度
        Returns:
            pixel_coords (torch.Tensor): 图像中的像素坐标 [bs, 2]，超出范围的点设置为 (0, 0)
        """
        # 添加一个齐次坐标分量
        target_point_h = torch.cat([target_point, torch.tensor([1.0])], dim=0)  # [4]

        # 将世界坐标转换到相机坐标系
        cam_coords = torch.matmul(cam_info_ex, target_point_h)[:3]  # [3]

        # 如果点在相机后面，直接返回 (0, 0)
        if cam_coords[2] <= 0:
            print("z坐标小于0")
            return torch.tensor([0, 0])

        # 将相机坐标投影到图像平面
        pixel_coords_h = torch.matmul(cam_info_in, cam_coords)  # [3]

        # 归一化以得到像素坐标
        pixel_coords_k = pixel_coords_h[:2] / pixel_coords_h[2]  # [2]

        # 将坐标转换为整数，方便图像索引
        pixel_coords = pixel_coords_k.round().long()

        # 检查是否在图像范围内，超出范围的点设置为 (0, 0)
        if pixel_coords[0] < 0 or pixel_coords[0] >= img_width or pixel_coords[1] < 0 or pixel_coords[1] >= img_height:
            print("超出限制")
        print(pixel_coords)
        # 转换图像为numpy格式
        img = keyframe_pic.permute(1, 2, 0).numpy()  # 假设图像的形状为 [channel, height, width]
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
        # 绘制图像和点
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.scatter(0, 128, color='blue', s=50, label='y max')
        plt.scatter(128, 0, color='green', s=50, label='x max')
        plt.scatter(0, 0, color='black', s=50, label='origin')
        if pixel_coords[0] != 0 or pixel_coords[1] != 0:  # 点在图像范围内
            plt.scatter(23 + 62 + pixel_coords[0].item(), pixel_coords[1].item() - 16, color='red', s=50,
                        label=f'Projected Point:{pixel_coords[0]},{pixel_coords[1]}')

            # plt.scatter(pixel_coords[1].item(), pixel_coords[0].item(), color='red', s=50, label=f'Projected Point:{pixel_coords[0]},{pixel_coords[1]}')
        else:
            print("Point is out of bounds")
        plt.title("Projected Point on Image")
        plt.legend()
        plt.show()

        import pdb;
        pdb.set_trace()

    @torch.no_grad()
    def get_action_trans(self, target_point, keyframe_pic, img_width, img_height, cam_info=None, if_vis=False):
        """
        计算世界坐标系下的点在图像中的像素位置，处理超出图像范围的情况
        Args:
            target_point (torch.Tensor): 世界坐标系下的点 [bs, 3]
            cam_info (list): 包含相机的外参 [bs, 4, 4, 4] 和内参 [bs, 4, 3, 3]
                - cam_info['extrinsics']: 相机外参 [bs, 4, 4]
                - cam_info['intrinsics']: 相机内参 [bs, 3, 3]
            img_width (int): 图像的宽度
            img_height (int): 图像的高度
        Returns:
            pixel_coords (torch.Tensor): 图像中的像素坐标 [bs, 2]，超出范围的点设置为 (0, 0)
        """
        bs = keyframe_pic.shape[0]
        pixel_coords_list = []

        # for n in range(bs):
        #     # 获取相机的外参（旋转和平移矩阵）
        #     extrinsics = cam_info[0][n]  # [4, 4, 4]
        #
        #     # 获取相机的内参（用于投影到像素坐标）
        #     intrinsics = cam_info[1][n]  # [4, 3, 3]
        #
        #     # 将世界坐标转换到相机坐标系
        #     # 增加齐次坐标 [bs, 3] -> [bs, 4]
        #     target_point_homo = torch.cat([target_point[n].unsqueeze(0).repeat(extrinsics.size(0), 1),
        #                                    torch.ones(extrinsics.size(0), 1).to(target_point.device)], dim=-1)  # [ 4]
        #
        #     # 将世界坐标转换到相机坐标系，相机外参为 [bs, 4, 4]
        #     cam_coords_homo = torch.bmm(extrinsics, target_point_homo.unsqueeze(-1)).squeeze(-1)  # [bs, 4]
        #
        #     # 保留相机坐标系下的 3D 坐标 (x, y, z)，忽略最后的齐次坐标
        #     cam_coords = cam_coords_homo[:, :3]  # [bs, 3]
        #
        #     # 将相机坐标系下的点投影到图像平面上
        #     # 图像平面的齐次坐标 (x_proj, y_proj, z_proj) = intrinsics @ [X, Y, Z]
        #     pixel_coords_homo = torch.bmm(intrinsics, cam_coords.unsqueeze(-1)).squeeze(-1)  # [bs, 3]
        #
        #     # 归一化像素坐标 (x/z, y/z)，将 (x, y, z) 转为 (x', y')
        #     pixel_coords = pixel_coords_homo[:, :2] / pixel_coords_homo[:, 2:3]  # [bs, 2]
        #
        #     # 检查是否超出图像范围
        #     # 获取图像边界
        #     x_max, y_max = img_width - 1, img_height - 1
        #
        #     # 检查是否超出图像边界
        #     x_out_of_bounds = (pixel_coords[:, 0] < 0) | (pixel_coords[:, 0] > x_max)
        #     y_out_of_bounds = (pixel_coords[:, 1] < 0) | (pixel_coords[:, 1] > y_max)
        #
        #     # 找到所有超出图像范围的点
        #     out_of_bounds = x_out_of_bounds | y_out_of_bounds
        #
        #     # 打印提示，并将超出范围的坐标设置为 (0, 0)
        #     if out_of_bounds.any():
        #         print(f"Warning: {out_of_bounds.sum().item()} points are out of bounds and will be set to (0, 0)")
        #         pixel_coords[out_of_bounds] = 0
        #     pixel_coords_list.append(pixel_coords)

        pixel_coords = target_point.view(bs, -1, 2)

        action_trans = mvt_utils.generate_hm_from_pt(
            pixel_coords.reshape(-1, 2),  # torch.Size([20, 2])
            (img_height, img_width),
            sigma=1.5,  # 1.5
            thres_sigma_times=3,
        )
        if if_vis:
            vis_point = pixel_coords.clone().to("cpu")
            vis_action_trans = action_trans.clone()
            vis_action_trans = vis_action_trans.view(bs, 4, img_height, img_width)

            for _bs in range(0, bs):
                for _nc in range(0, 4):
                    # 获取要可视化的二维矩阵 (h, w)
                    heatmap = vis_action_trans[_bs, _nc, :, :].cpu().detach().numpy()
                    image = keyframe_pic[_bs, _nc].cpu().detach().numpy().transpose(1, 2, 0)

                    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)
                    # 绘制热图
                    plt.imshow(image)
                    plt.plot(vis_point[_bs][_nc][0], vis_point[_bs][_nc][1], 'o', color='b')
                    # plt.plot(vis_point[_bs * 4 + _nc], 'o', color='r')
                    plt.imshow(heatmap, cmap='hot', alpha=0.5, interpolation='nearest')
                    plt.colorbar()  # 添加颜色条，显示不同值的范围
                    plt.title(f"Heatmap for batch {_bs}, num_camera {_nc}")
                    plt.show()

        action_trans = action_trans.view(bs, 4, img_height * img_width).transpose(1, 2).clone()
        return action_trans.to("cuda:0")


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
            use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        bimamba=True,
        device=None,
        dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)  # [1,3,1,224,224]
        return x  # [1,192,1,14,14]


class VisionMamba(nn.Module):
    def __init__(
            self,
            img_size=128,
            patch_size=16,
            depth=24,
            embed_dim=192,
            channels=3,
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True,
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1,
            num_frames=8,
            fc_drop_rate=0.,
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}  # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        """
        x: [bs, channel, n_cam*n_hisframe, width, height]

        """

        # import pdb
        # pdb.set_trace()
        x = self.patch_embed(
            x)  # [bs, channel, n_cam*n_hisframe, width, height]->[bs, dim, n_cam*n_hisframe, height/patch_size, height/patch_size]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)  # [bs*seq_len, w*h, dim]

        # cls_token = self.cls_token.expand(x.shape[0], -1,
        #                                   -1)  # stole cls_tokens impl from Phil Wang, thanks  # [1,1,192]
        # x = torch.cat((cls_token, x), dim=1)  # [1,197,192]
        x = x + self.pos_embed  # [1, patch_num, dim] 每张图的patch的位置嵌入 #todo

        # temporal pos
        # cls_tokens = x[:B, :1, :]  # [1,1,192]
        # x = x[:, 1:]  # [1,196,192]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)  # [bs*w*h, seq_len, dim]
        x = x + self.temporal_pos_embedding  # [1,1,192] + [196,1,192]= [196,1,192]  序列长度上的嵌入  #todo
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)  # [bs, w*h*seq_len, dim]
        # x = torch.cat((cls_tokens, x), dim=1)  # [1,197,192]

        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )  # [1,197,192]

        return hidden_states  # [1,192]

    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)

        # x = self.head(self.head_drop(x))  # [1,1000]
        return x


def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


@register_model
def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_small(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=576,
        depth=32,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def my_mamba_model(pretrained=False, **kwargs):
    model = MyMambaPipeline(
        num_features=192).cuda()
    # model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table, parameter_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 1
    img_size = 224

    # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
    # model = videomamba_tiny(num_frames=num_frames).cuda()
    # flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda())
    # s = time.time()
    # print(flop_count_table(flops, max_depth=1))
    # print(time.time() - s)
    # print("FLOPs: ", flops.total())
    # print(parameter_count_table(model))

    inputs1 = torch.randn(1, 3, 16, 128, 128, device='cuda:0')
    inputs2 = torch.randn(1, 3, 16, 128, 128, device='cuda:0')
    inputs3 = torch.randn(1, 77, 512, device='cuda:0')
    model = my_mamba_model().cuda()
    inputs = [inputs1, inputs2, inputs3]
    flops = FlopCountAnalysis(model, inputs)
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time() - s)

    # model = eval(model_name)(img_size=resolution, num_frames=frame, num_classes=400)
    # replace_layernorm(model)
    # model.to('cuda:0')
    # model.eval()
    # model = torch.jit.trace(model, inputs)
    # compute_throughput(model_name, model, device, batch_size, frame=frame, resolution=resolution)

# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from yacs.config import CfgNode as CN

_C = CN()

# mamba param
_C.mymamba = CN()
_C.mymamba.num_features = 192
_C.mymamba.num_frames = 4
_C.mymamba.num_camera = 4
_C.mymamba.img_height = 128
_C.mymamba.img_width = 128
_C.mymamba.patch_size = 16
_C.mymamba.depth_of_rgb_mamba = 24
_C.mymamba.depth_of_pcd_mamba = 24
_C.mymamba.depth_rgb = 4
_C.mymamba.depth_pcd = 4
_C.mymamba.depth_fusionatten = 4
_C.mymamba.depth_crossatten = 4
_C.mymamba.fusion_layer_num = 4


##
_C.depth = 8
_C.img_size = 220
_C.add_proprio = True
_C.proprio_dim = 4
_C.add_lang = True
_C.lang_dim = 512
_C.lang_len = 77
_C.img_feat_dim = 3
_C.feat_dim = (72 * 3) + 2 + 2
_C.im_channels = 64
_C.attn_dim = 512
_C.attn_heads = 8
_C.attn_dim_head = 64
_C.activation = "lrelu"
_C.weight_tie_layers = False
_C.attn_dropout = 0.1
_C.decoder_dropout = 0.0
_C.img_patch_size = 11
_C.final_dim = 64
_C.self_cross_ver = 1
_C.add_corr = True
_C.norm_corr = False
_C.add_pixel_loc = True
_C.add_depth = True
_C.rend_three_views = False
_C.use_point_renderer = False
_C.pe_fix = True
_C.feat_ver = 0
_C.wpt_img_aug = 0.01
_C.inp_pre_pro = True
_C.inp_pre_con = True
_C.cvx_up = False
_C.xops = False
_C.rot_ver = 0
_C.num_rot = 72
_C.stage_two = False
_C.st_sca = 4
_C.st_wpt_loc_aug = 0.05
_C.st_wpt_loc_inp_no_noise = False
_C.img_aug_2 = 0.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()

# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import copy
import torch

from torch import nn
from torch.cuda.amp import autocast

import rvt.mvt.utils as mvt_utils

from rvt.mvt.mvt_single import MVT as MVTSingle
from rvt.mvt.config import get_cfg_defaults
from rvt.mvt.renderer import BoxRenderer


class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        rend_three_views,
        use_point_renderer,
        pe_fix,
        feat_ver,
        wpt_img_aug,
        inp_pre_pro,
        inp_pre_con,
        cvx_up,
        xops,
        rot_ver,
        num_rot,
        stage_two,
        st_sca,
        st_wpt_loc_aug,
        st_wpt_loc_inp_no_noise,
        img_aug_2,
        renderer_device="cuda:0",
    ):
        """MultiView Transfomer
        :param stage_two: whether or not there are two stages
        :param st_sca: scaling of the pc in the second stage
        :param st_wpt_loc_aug: how much noise is to be added to wpt_local when
            transforming the pc in the second stage while training. This is
            expressed as a percentage of total pc size which is 2.
        :param st_wpt_loc_inp_no_noise: whether or not to add any noise to the
            wpt_local location which is fed to stage_two. This wpt_local
            location is used to extract features for rotation prediction
            currently. Other use cases might also arise later on. Even if
            st_wpt_loc_aug is True, this will compensate for that if set to
            True.
        :param img_aug_2: similar to img_aug in rvt repo but applied only to
            point feat and not the whole point cloud
        """
        super().__init__()

        self.use_point_renderer = use_point_renderer
        if self.use_point_renderer:
            from point_renderer.rvt_renderer import RVTBoxRenderer as BoxRenderer
        else:
            from mvt.renderer import BoxRenderer
        global BoxRenderer

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]
        del args["stage_two"]
        del args["st_sca"]
        del args["st_wpt_loc_aug"]
        del args["st_wpt_loc_inp_no_noise"]
        del args["img_aug_2"]

        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.stage_two = stage_two
        self.st_sca = st_sca
        self.st_wpt_loc_aug = st_wpt_loc_aug
        self.st_wpt_loc_inp_no_noise = st_wpt_loc_inp_no_noise
        self.img_aug_2 = img_aug_2

        # for verifying the input
        self.feat_ver = feat_ver
        self.img_feat_dim = img_feat_dim
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )
        self.num_img = self.renderer.num_img
        self.proprio_dim = proprio_dim
        self.img_size = img_size

        self.mvt1 = MVTSingle(
            **args,
            renderer=self.renderer,
            no_feat=self.stage_two,
        )
        if self.stage_two:
            self.mvt2 = MVTSingle(**args, renderer=self.renderer)

    def get_pt_loc_on_img(self, pt, mvt1_or_mvt2, dyn_cam_info, out=None):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local (see forward()),
            even for mvt2
        :param out: output from mvt, when using mvt2, we also need to provide the
            origin location where where the point cloud needs to be shifted
            before estimating the location in the image
        """
        assert len(pt.shape) == 3
        bs, _np, x = pt.shape
        assert x == 3

        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            assert out is None
            out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)
        else:
            assert self.stage_two
            assert out is not None
            assert out['wpt_local1'].shape == (bs, 3)
            pt, _ = mvt_utils.trans_pc(pt, loc=out["wpt_local1"], sca=self.st_sca)
            pt = pt.view(bs, _np, 3)
            out = self.mvt2.get_pt_loc_on_img(pt, dyn_cam_info)

        return out

    def get_wpt(self, out, mvt1_or_mvt2, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        :param y_q: refer to the definition in mvt_single.get_wpt
        """
        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            wpt = self.mvt1.get_wpt(
                out, dyn_cam_info, y_q,
            )
        else:
            assert self.stage_two
            wpt = self.mvt2.get_wpt(
                out["mvt2"], dyn_cam_info, y_q
            )
            wpt = out["rev_trans"](wpt)

        return wpt

    def render(self, pc, img_feat, img_aug, mvt1_or_mvt2, dyn_cam_info):
        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            mvt = self.mvt1
        else:
            mvt = self.mvt2

        with torch.no_grad():
            with autocast(enabled=False):
                if dyn_cam_info is None:
                    dyn_cam_info_itr = (None,) * len(pc)
                else:
                    dyn_cam_info_itr = dyn_cam_info
                import pdb
                pdb.set_trace()
                if mvt.add_corr:  # 是否将pc与imgfeat进行concat再进行render处理
                    if mvt.norm_corr:
                        img = []
                        for _pc, _img_feat, _dyn_cam_info in zip(
                            pc, img_feat, dyn_cam_info_itr
                        ):
                            # fix when the pc is empty
                            max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                            img.append(
                                self.renderer(
                                    _pc,
                                    torch.cat((_pc / max_pc, _img_feat), dim=-1),  # pc归一化
                                    fix_cam=True,
                                    dyn_cam_info=(_dyn_cam_info,)
                                    if not (_dyn_cam_info is None)
                                    else None,
                                ).unsqueeze(0)
                            )
                    else:
                        img = [
                            self.renderer(
                                _pc,
                                torch.cat((_pc, _img_feat), dim=-1),
                                fix_cam=True,
                                dyn_cam_info=(_dyn_cam_info,)
                                if not (_dyn_cam_info is None)
                                else None,
                            ).unsqueeze(0)
                            for (_pc, _img_feat, _dyn_cam_info) in zip(
                                pc, img_feat, dyn_cam_info_itr
                            )
                        ]
                else:
                    img = [
                        self.renderer(
                            _pc,
                            _img_feat,
                            fix_cam=True,
                            dyn_cam_info=(_dyn_cam_info,)
                            if not (_dyn_cam_info is None)
                            else None,
                        ).unsqueeze(0)
                        for (_pc, _img_feat, _dyn_cam_info) in zip(
                            pc, img_feat, dyn_cam_info_itr
                        )
                    ]



        img = torch.cat(img, 0)  # torch.Size([4, 5, 220, 220, 7])

        #####################################################################
        import matplotlib.pyplot as plt
        # 假设 img 是形状 [4, 5, 220, 220, 7] 的张量
        # 提取各部分
        xyz_images = img[..., :3].cpu().numpy()  # 3通道 xyz 图像
        rgb_images = img[..., 3:6].cpu().numpy()  # 3通道 rgb 图像
        depth_images = img[..., 6].cpu().numpy()  # 1通道 深度图像
        # 选择要可视化的 batch (例如 batch 0)
        batch_idx = 0
        num_cam = img.shape[1]  # 获取相机的数量
        # 创建用于显示的子图 (num_cam 行, 3 列)
        fig, axs = plt.subplots(num_cam, 3, figsize=(12, num_cam * 4))
        for c in range(num_cam):
            # 可视化 xyz 图像
            axs[c, 0].imshow(xyz_images[batch_idx, c])
            axs[c, 0].set_title(f"Cam {c} - XYZ")
            axs[c, 0].axis('off')
            # 可视化 rgb 图像
            axs[c, 1].imshow(rgb_images[batch_idx, c])
            axs[c, 1].set_title(f"Cam {c} - RGB")
            axs[c, 1].axis('off')
            # 可视化 深度图像
            axs[c, 2].imshow(depth_images[batch_idx, c], cmap='gray')
            axs[c, 2].set_title(f"Cam {c} - Depth")
            axs[c, 2].axis('off')
        plt.tight_layout()
        plt.show()
        #####################################################################


        # for visualization purposes
        if mvt.add_corr:  # true
            mvt.img = img[:, :, 3:].clone().detach()  # torch.Size([4, 5, 4, 220, 220])
        else:
            mvt.img = img.clone().detach()


        img = img.permute(0, 1, 4, 2, 3)  # torch.Size([4, 5, 7, 220, 220])
        # image augmentation
        # 如果 img_aug 不为 0，那么会进行图像增强，方法是给图像添加随机噪声。
        # 首先生成一个范围在 [0, img_aug] 之间的随机值 stdv，然后生成形状与图像相同的噪声 noise，噪声的值范围在 [-stdv, stdv] 之间。
        # 最后通过 torch.clamp 将添加噪声后的图像限制在 [-1, 1] 范围内，防止值超出范围。
        if img_aug != 0:  # 0.1
            stdv = img_aug * torch.rand(1, device=img.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
            img = torch.clamp(img + noise, -1, 1)

        # 如果 mvt.add_pixel_loc 为真，则会在图像中添加像素位置信息。
        if mvt.add_pixel_loc:  # true
            bs = img.shape[0]
            pixel_loc = mvt.pixel_loc.to(img.device)  # torch.Size([5, 3, 220, 220]) [-1,1]
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )

        return img

    def verify_inp(
        self,
        pc,
        img_feat,
        proprio,
        lang_emb,
        img_aug,
        wpt_local,
        rot_x_y,
    ):
        bs = len(pc)
        assert bs == len(img_feat)

        if not self.training:
            # no img_aug when not training
            assert img_aug == 0
            assert rot_x_y is None, f"rot_x_y={rot_x_y}"

        if self.training:
            assert (
                (not self.feat_ver == 1)
                or (not wpt_local is None)
            )

            if self.rot_ver == 0:
                assert rot_x_y is None, f"rot_x_y={rot_x_y}"
            elif self.rot_ver == 1:
                assert rot_x_y.shape == (bs, 2), f"rot_x_y.shape={rot_x_y.shape}"
                assert (rot_x_y >= 0).all() and (
                    rot_x_y < self.num_rot
                ).all(), f"rot_x_y={rot_x_y}"
            else:
                assert False

        for _pc, _img_feat in zip(pc, img_feat):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape

            assert np == np2
            assert x1 == 3
            assert x2 == self.img_feat_dim

        if self.add_proprio:
            bs3, x3 = proprio.shape
            assert bs == bs3
            assert (
                x3 == self.proprio_dim
            ), "Does not support proprio of shape {proprio.shape}"
        else:
            assert proprio is None, "Invalid input for proprio={proprio}"

        if self.add_lang:
            bs4, x4, x5 = lang_emb.shape
            assert bs == bs4
            assert (
                x4 == self.lang_max_seq_len
            ), "Does not support lang_emb of shape {lang_emb.shape}"
            assert (
                x5 == self.lang_emb_dim
            ), "Does not support lang_emb of shape {lang_emb.shape}"
        else:
            assert (lang_emb is None) or (
                torch.all(lang_emb == 0)
            ), f"Invalid input for lang={lang}"

        if not (wpt_local is None):
            bs5, x6 = wpt_local.shape
            assert bs == bs5
            assert x6 == 3, "Does not support wpt_local of shape {wpt_local.shape}"

        if self.training:
            assert (not self.stage_two) or (not wpt_local is None)

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param wpt_local: gt location of the wpt in 3D, tensor of shape
            (bs, 3)
        :param rot_x_y: (bs, 2) rotation in x and y direction
        """
        # 数据格式检查
        self.verify_inp(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_emb,
            img_aug=img_aug,
            wpt_local=wpt_local,
            rot_x_y=rot_x_y,
        )
        with torch.no_grad():
            if self.training and (self.img_aug_2 != 0):
                for x in img_feat:  # 加噪增强
                    stdv = self.img_aug_2 * torch.rand(1, device=x.device)
                    # values in [-stdv, stdv]
                    noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                    x = x + noise

            # 渲染虚拟图像
            img = self.render(
                pc=pc,
                img_feat=img_feat,
                img_aug=img_aug,
                mvt1_or_mvt2=True,
                dyn_cam_info=None,
            )  # torch.Size([4, 5, 10, 220, 220])   bs, num_img, img_feat_dim, h, w

        if self.training:
            wpt_local_stage_one = wpt_local
            wpt_local_stage_one = wpt_local_stage_one.clone().detach()
        else:
            wpt_local_stage_one = wpt_local

        out = self.mvt1(
            img=img,
            proprio=proprio,
            lang_emb=lang_emb,
            wpt_local=wpt_local_stage_one,
            rot_x_y=rot_x_y,
            **kwargs,
        )

        # if self.stage_two:
        #     with torch.no_grad():
        #         # adding then noisy location for training
        #         if self.training:
        #             # noise is added so that the wpt_local2 is not exactly at
        #             # the center of the pc
        #             wpt_local_stage_one_noisy = mvt_utils.add_uni_noi(
        #                 wpt_local_stage_one.clone().detach(), 2 * self.st_wpt_loc_aug
        #             )
        #             pc, rev_trans = mvt_utils.trans_pc(
        #                 pc, loc=wpt_local_stage_one_noisy, sca=self.st_sca
        #             )
        #
        #             if self.st_wpt_loc_inp_no_noise:
        #                 wpt_local2, _ = mvt_utils.trans_pc(
        #                     wpt_local, loc=wpt_local_stage_one_noisy, sca=self.st_sca
        #                 )
        #             else:
        #                 wpt_local2, _ = mvt_utils.trans_pc(
        #                     wpt_local, loc=wpt_local_stage_one, sca=self.st_sca
        #                 )
        #
        #         else:
        #             # bs, 3
        #             wpt_local = self.get_wpt(
        #                 out, y_q=None, mvt1_or_mvt2=True,
        #                 dyn_cam_info=None,
        #             )
        #             pc, rev_trans = mvt_utils.trans_pc(
        #                 pc, loc=wpt_local, sca=self.st_sca
        #             )
        #             # bad name!
        #             wpt_local_stage_one_noisy = wpt_local
        #
        #             # must pass None to mvt2 while in eval
        #             wpt_local2 = None
        #
        #         img = self.render(
        #             pc=pc,
        #             img_feat=img_feat,
        #             img_aug=img_aug,
        #             mvt1_or_mvt2=False,
        #             dyn_cam_info=None,
        #         )
        #
        #     out_mvt2 = self.mvt2(
        #         img=img,
        #         proprio=proprio,
        #         lang_emb=lang_emb,
        #         wpt_local=wpt_local2,
        #         rot_x_y=rot_x_y,
        #         **kwargs,
        #     )
        #
        #     out["wpt_local1"] = wpt_local_stage_one_noisy
        #     out["rev_trans"] = rev_trans
        #     out["mvt2"] = out_mvt2

        return out

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        if not self.use_point_renderer:
            print("Freeing up some memory")
            self.renderer.free_mem()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    mvt = MVT(**cfg)
    breakpoint()

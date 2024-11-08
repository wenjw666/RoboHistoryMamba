# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import pprint

import clip
import torch
import torchvision
import numpy as np
import torch.nn as nn
import bitsandbytes as bnb

from scipy.spatial.transform import Rotation
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

import rvt.utils.peract_utils as peract_utils
import rvt.mvt.utils as mvt_utils
import rvt.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils

from rvt.mvt.augmentation import apply_se3_aug_con, aug_utils
from peract_colab.arm.optim.lamb import Lamb
from yarr.agents.agent import ActResult
from rvt.utils.dataset import _clip_encode_text
from rvt.utils.lr_sched_utils import GradualWarmupScheduler

import rvt.mvt.mambatest as mambatest
from rvt.mvt.mambatest import MyMambaPipeline
from rvt.diffuser_actor.utils.layers import ParallelAttention
from rvt.diffuser_actor.utils.position_encodings import SinusoidalPosEmb

import matplotlib.pyplot as plt


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
        wpt,
        pred_wpt,
        action_rot,
        pred_rot_quat,
        action_grip_one_hot,
        grip_q,
        action_collision_one_hot,
        collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i: i + 1], pred_wpt[i: i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i: i + 1].argmax(-1),
                grip_q[i: i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i: i + 1].argmax(-1),
                collision_q[i: i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
        self,
        tasks,
        wpt,
        pred_wpt,
        action_rot,
        pred_rot_quat,
        action_grip_one_hot,
        grip_q,
        action_collision_one_hot,
        collision_q,
        reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
        agent,
        loss_log,
        reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val))
    pprint.pprint(out)
    return out


class RVTAgent:
    def __init__(
            self,
            network: nn.Module,
            num_rotation_classes: int,
            stage_two: bool,
            add_lang: bool,
            amp: bool,
            bnb: bool,
            move_pc_in_bound: bool,
            lr: float = 0.0001,
            lr_cos_dec: bool = False,
            cos_dec_max_step: int = 60000,
            warmup_steps: int = 0,
            image_resolution: list = None,
            lambda_weight_l2: float = 0.0,
            transform_augmentation: bool = True,
            transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
            transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
            place_with_mean: bool = True,
            transform_augmentation_rot_resolution: int = 5,
            optimizer_type: str = "lamb",
            gt_hm_sigma: float = 1.5,
            img_aug: bool = False,
            add_rgc_loss: bool = False,
            scene_bounds: list = peract_utils.SCENE_BOUNDS,
            cameras: list = peract_utils.CAMERAS,
            rot_ver: int = 0,
            rot_x_y_aug: int = 2,
            log_dir="",
    ):
        """
        :param gt_hm_sigma: the std of the groundtruth hm, currently for for
            2d, if -1 then only single point is considered
        :type gt_hm_sigma: float
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param rot_x_y_aug: only applicable when rot_ver is 1, it specifies how
            much error we should add to groundtruth rotation while training
        :param log_dir: a folder location for saving some intermediate data
        """

        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.amp = amp
        self.bnb = bnb
        self.stage_two = stage_two
        self.add_lang = add_lang
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps
        self.lr_cos_dec = lr_cos_dec
        self.cos_dec_max_step = cos_dec_max_step
        self.scene_bounds = scene_bounds
        self.cameras = cameras
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

        self.scaler = GradScaler(enabled=self.amp)

        #######################################################
        self.mamba_pipeline = MyMambaPipeline(num_features=192).cuda()

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if self._optimizer_type == "lamb":
            if self.bnb:
                print("Using 8-Bit Optimizer")
                self._optimizer = bnb.optim.LAMB(
                    self._network.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                )
            else:
                # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                self._optimizer = Lamb(
                    self._network.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception("Unknown optimizer")

        if self.lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self.cos_dec_max_step,
                eta_min=self._lr / 100,  # mininum lr
            )
        else:
            after_scheduler = None
        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=after_scheduler,
        )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    # copied from per-act and removed the translation part
    def _get_one_hot_expert_actions(
            self,
            batch_size,
            action_rot,
            action_grip,
            action_ignore_collisions,
            device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            # q
            # if self.stage_two:
            #     out = out["mvt2"]
            #     q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            #     if not only_pred:
            #         q_trans2 = q_trans2.clone()
            #     q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            # if self.stage_two:
            #     out = out["mvt2"]

        if self.rot_ver == 0:
            # (bs, 218)
            rot_q = out["feat"].view(bs, -1)[:, 0: self.num_all_rot]
            grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot: self.num_all_rot + 2]
            # (bs, 2)
            collision_q = out["feat"].view(bs, -1)[
                          :, self.num_all_rot + 2: self.num_all_rot + 4
                          ]
        elif self.rot_ver == 1:
            rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                              dim=-1).view(bs, -1)
            grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
        else:
            assert False

        y_q = None

        return q_trans, rot_q, grip_q, collision_q, y_q, pts

    def update(
            self,
            step: int,
            replay_sample: dict,
            backprop: bool = True,
            eval_log: bool = False,
            reset_log: bool = False,
    ) -> dict:
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        assert replay_sample["lang_goal_embs"].shape[1:] == (1, 77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (
            1,
            self._net_mod.proprio_dim,
        )

        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
                          :, -1
                          ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
                                   :, -1
                                   ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        lang_goal_embs = replay_sample["lang_goal_embs"][:, -1].float()
        tasks = replay_sample["tasks"]

        proprio = arm_utils.stack_on_channel(replay_sample["low_dim_state"])  # (b, 4)
        return_out = {}

        # 提取obs与pcd
        obs, pcd = peract_utils._preprocess_inputs(replay_sample,
                                                   self.cameras)  #  len(obs)=4  len(obs[0])=2  obs[0][0].shape torch.Size([4, 3, 128, 128]) # len(pcd)=4  pcd[0].size() torch.Size([4, 3, 128, 128])

        history_obs, history_pcds = [], []
        num_history_frame = 4
        _norm_rgb = lambda x: (x.float() / 255.0) * 2.0 - 1.0
        for his_idx in range(num_history_frame):
            obs_temp = []
            pcds_temp = []
            for n in self.cameras:
                rgb_temp = torch.cat(torch.split(replay_sample[f"%s_rgb_hisId{his_idx}" % n], 1, dim=1), dim=2).squeeze(
                    1)
                pcd_temp = torch.cat(torch.split(replay_sample[f"%s_point_cloud_hisId{his_idx}" % n], 1, dim=1),
                                     dim=2).squeeze(1)

                rgb_temp = _norm_rgb(rgb_temp)

                obs_temp.append(
                    [rgb_temp, pcd_temp]
                )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
                pcds_temp.append(pcd_temp)  # only pointcloud
            history_obs.append(obs_temp)
            history_pcds.append(pcds_temp)

        # ###################可视化###################
        #
        # # 假设 obs 的结构是 obs[0][0] 为一个形状 [4, 3, 128, 128] 的张量
        # num_outer = len(obs)  # 外层列表的长度
        # num_inner = len(obs[0])  # 内层列表的长度
        # # 假设每个 obs[i][j] 都是形状为 [4, 3, 128, 128] 的张量
        # total_images = num_outer * num_inner * obs[0][0].shape[0]  # 总图像数量
        # images_per_group = obs[0][0].shape[0]  # 每组中包含的图像数量
        # # 创建网格，计算网格的行和列数
        # grid_rows = num_outer
        # grid_cols = num_inner * images_per_group
        # # 创建一个子图，大小根据图像数量自动调整
        # fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
        # # 遍历 obs 列表并可视化图像
        # for outer_idx in range(num_outer):
        #     for inner_idx in range(num_inner):
        #         images = obs[outer_idx][inner_idx]  # 获取 [4, 3, 128, 128] 形状的图像
        #         for img_idx in range(images.shape[0]):
        #             img = images[img_idx].cpu().detach().permute(1, 2, 0).numpy()  # 转换图像格式
        #
        #             # 计算在 grid 中的位置
        #             row = outer_idx
        #             col = inner_idx * images_per_group + img_idx
        #
        #             # 在子图中绘制图像
        #             axes[row, col].imshow(img)
        #             axes[row, col].axis('off')  # 关闭坐标轴
        # # 显示所有图像
        # plt.tight_layout()
        # plt.show()

        with torch.no_grad():

            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )  # pc,img_feat:torch.Size([4, 65536, 3])  65536=128x128x4

            history_pc = []
            history_img_feats = []
            for his_idx in range(num_history_frame):
                pc_temp, img_feat_temp = rvt_utils.get_pc_img_feat(
                    history_obs[his_idx],
                    history_pcds[his_idx],
                )
                history_pc.append(pc_temp)
                history_img_feats.append(img_feat_temp)

            ################### 可视化 ################
            vis_img_feat = img_feat.clone().view(img_feat.shape[0], 4, 128, 128, 3)
            vis_pc_feat = pc.clone().view(img_feat.shape[0], 4, 128, 128, 3)

            def vis_func(feature: torch.tensor):
                # 假设 vis_img_feat 是形状为 [4, 4, 128, 128, 3] 的张量
                vis_img_feat = feature.cpu().detach().numpy()  # 将 Tensor 转换为 NumPy 数组
                # 创建一个 4x4 的网格来显示 16 张图像
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # 每个图像的大小为 2x2
                # 遍历每个位置，将图像显示出来
                for i in range(4):
                    for j in range(4):
                        # 从 vis_img_feat 中获取第 i 行第 j 列的图像
                        img = vis_img_feat[i, j, :, :, :]  # 形状为 [128, 128, 3]

                        # 在子图的第 i 行第 j 列中显示图像
                        axes[i, j].imshow(img)
                        axes[i, j].axis('off')  # 关闭坐标轴
                # 显示所有图像
                plt.tight_layout()
                plt.show()

            # vis_func(vis_img_feat)
            # vis_func(vis_pc_feat)

            ################### 可视化 ################

            # action真值处理，进行增强
            # if self._transform_augmentation and backprop:
            #     action_trans_con, action_rot, pc = apply_se3_aug_con(
            #         pcd=pc,
            #         action_gripper_pose=action_gripper_pose,
            #         bounds=torch.tensor(self.scene_bounds),
            #         trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
            #         rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
            #     )
            #     action_trans_con = torch.tensor(action_trans_con).to(pc.device)
            #     action_rot = torch.tensor(action_rot).to(pc.device)

            ################### 可视化 ################

            # vis_pc_feat = pc.clone().view(img_feat.shape[0], 4, 128, 128, 3)
            # vis_func(vis_pc_feat)

            def vis_point_cloud(pc, marker_point=None):
                """
                可视化点云并添加一个特殊标记点。

                参数:
                - pc: 点云数据 (Tensor)，形状为 [N, 3]
                - marker_point: 特殊标记点 (list 或 ndarray)，形状为 [3]，默认值为 None。
                """
                import open3d as o3d
                import numpy as np
                # 将 Tensor 转换为 NumPy 数组
                point_cloud_tensor = pc.cpu()
                point_cloud = point_cloud_tensor.numpy()

                # 创建 open3d 点云对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud)

                # 创建可视化窗口
                vis = o3d.visualization.Visualizer()
                vis.create_window()

                # 添加点云到可视化对象
                vis.add_geometry(pcd)

                # 如果提供了 marker_point，则添加特殊标记点
                if marker_point is not None:
                    # 确保 marker_point 为 NumPy 数组
                    marker_point = np.array(marker_point).reshape(1, 3)

                    # 创建单独的点云对象用于标记点
                    marker_pcd = o3d.geometry.PointCloud()
                    marker_pcd.points = o3d.utility.Vector3dVector(marker_point)
                    marker_pcd.paint_uniform_color([1, 0, 0])  # 使用红色标记

                    # 设置点大小（使用 Sphere 代替，或其他方式增加点大小）
                    marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # 调整半径以改变点大小
                    marker_sphere.translate(marker_point[0])  # 将球体位置移动到标记点
                    marker_sphere.paint_uniform_color([1, 0, 0])  # 红色

                    # 添加标记点对象到可视化
                    vis.add_geometry(marker_sphere)

                # 运行可视化
                vis.run()
                # 关闭窗口并销毁资源
                vis.destroy_window()

            # vis_point_cloud(pc[0,:,:])
            ################### 可视化 ################

            # 旋转标准化
            # TODO: vectorize
            # action_rot = action_rot.cpu().numpy()
            # for i, _action_rot in enumerate(action_rot):
            #     _action_rot = aug_utils.normalize_quaternion(_action_rot)
            #     if _action_rot[-1] < 0:
            #         _action_rot = -_action_rot
            #     action_rot[i] = _action_rot
            #
            # # 点云图像处理
            # pc, img_feat = rvt_utils.move_pc_in_bound(
            #     pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            # )
            # # 历史点云图像处理
            # history_pc_in_bound = []
            # history_img_feat_in_bound = []
            # for n in range(num_history_frame):
            #     pc_temp, img_feat_temp = rvt_utils.move_pc_in_bound(
            #         history_pc[n], history_img_feats[n], self.scene_bounds, no_op=not self.move_pc_in_bound
            #     )
            #     history_pc_in_bound.append(pc_temp)
            #     history_img_feat_in_bound.append(img_feat_temp)

            wpt = [x[:3] for x in action_trans_con]  # 动作平移  真值

            temp_vis_pc = pc

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                ### 该代码的目的是将点云缩放到一个固定范围（例如单位立方体），同时定义了一个反向变换函数，可以将点云从标准化后的空间恢复到原始的坐标系。
                # 如果你有一个大的场景数据或多个场景，你可以用这种方法将它们标准化，这样可以更方便地在不同场景中处理点云数据。
                # rev_trans 函数可以用于将标准化后的点云恢复到原始的场景中，确保你能够以可控的方式对点云进行标准化和逆标准化操作。
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))  # 夹爪位置真值
                rev_trans.append(b)  # 将处理后的pc变换回原坐标系

            wpt_local = torch.cat(wpt_local, axis=0)  # torch.Size([4, 3])

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug  # 0.1
            else:
                img_aug = 0

            dyn_cam_info = None

        with autocast(enabled=self.amp):

            # # 真值向量生成
            # (
            #     action_rot_x_one_hot,  # torch.Size([4, 72])
            #     action_rot_y_one_hot,  # torch.Size([4, 72])
            #     action_rot_z_one_hot,  # torch.Size([4, 72])
            #     action_grip_one_hot,  # (bs, 2)
            #     action_collision_one_hot,  # (bs, 2)
            # ) = self._get_one_hot_expert_actions(
            #     bs, action_rot, action_grip, action_ignore_collisions, device=self._device
            # )
            #
            # if self.rot_ver == 1:
            #     rot_x_y = torch.cat(  # 该值应为真值
            #         [
            #             action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
            #             action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
            #         ],
            #         dim=-1,
            #     )
            #
            #     # xy轴旋转数据增强
            #     if self.rot_x_y_aug != 0:
            #         # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
            #         rot_x_y += torch.randint(
            #             -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
            #         ).to(rot_x_y.device)
            #         rot_x_y %= self._num_rotation_classes

            # out = self._network(
            #     pc=pc,  # 当前观察
            #     img_feat=img_feat,  # 当前观察
            #     proprio=proprio,  # 当前观察
            #     lang_emb=lang_goal_embs,  # 当前观察
            #     img_aug=img_aug,  # 当前观察
            #     wpt_local=wpt_local if self._network.training else None,  # 真值
            #     rot_x_y=rot_x_y if self.rot_ver == 1 else None,  # 真值
            # )

            # out = None

            # q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            #     out, dims=(bs, nc, h, w)  # (4,5,220,220)
            # )
            # pts = None

            #  wpt_local为真值
            # action_trans = self.get_action_trans(
            #     wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
            # )

            # loss_log = {}

            ########################################

            for n in range(num_history_frame):
                history_pc[n] = history_pc[n].view(bs, 4, 128, 128, 3).permute(0, 4, 1, 2, 3).cuda()
                history_img_feats[n] = history_img_feats[n].view(bs, 4, 128, 128, 3).permute(0, 4, 1, 2, 3).cuda()
            history_pc = torch.cat(history_pc, dim=2)
            history_img_feats = torch.cat(history_img_feats, dim=2)

            # vis_img_feat = vis_img_feat.permute(0, 4, 1, 2, 3).repeat(1, 1, 4, 1, 1).cuda()  # [4, 3, 4, 128, 128]
            # vis_pc_feat = vis_pc_feat.permute(0, 4, 1, 2, 3).repeat(1, 1, 4, 1, 1).cuda()  # [4, 3, 4, 128, 128]

            hm_result = self.mamba_pipeline([history_img_feats, history_pc, lang_goal_embs])
            hm_result = hm_result.view(bs, 4, -1).transpose(1, 2).clone()

            # cam_info = [torch.concat([replay_sample["front_camera_keyframe_extrinsics"],
            #                           replay_sample["left_shoulder_camera_keyframe_extrinsics"],
            #                           replay_sample["right_shoulder_camera_keyframe_extrinsics"],
            #                           replay_sample["wrist_camera_keyframe_extrinsics"]], dim=1),
            #             torch.concat([replay_sample["front_camera_keyframe_intrinsics"],
            #                           replay_sample["left_shoulder_camera_keyframe_intrinsics"],
            #                           replay_sample["right_shoulder_camera_keyframe_intrinsics"],
            #                           replay_sample["wrist_camera_keyframe_intrinsics"]], dim=1)
            #             ]
            target_rgb = torch.concat([replay_sample["front_keyframe_rgb"],
                                       replay_sample["left_shoulder_keyframe_rgb"],
                                       replay_sample["right_shoulder_keyframe_rgb"],
                                       replay_sample["wrist_keyframe_rgb"]], dim=1)
            # import pdb
            # pdb.set_trace()
            target_point = []
            for n in range(4):
                for cname in ["front", "left_shoulder", "right_shoulder", "wrist"]:
                    target_point.append(self.mamba_pipeline.get_action_trans_single_pcd(wpt[n].to("cpu"),
                                                                                        replay_sample[
                                                                                            "%s_keyframe_point_cloud" % cname][
                                                                                            n, 0].to("cpu"),
                                                                                        replay_sample[
                                                                                            "%s_keyframe_rgb" % cname][
                                                                                            n, 0].to("cpu"),
                                                                                        128,
                                                                                        128))
            target_point = torch.concat(target_point, dim=0)

            # self.mamba_pipeline.get_action_trans_single(wpt[n].to("cpu"),
            #                                             replay_sample["front_camera_keyframe_extrinsics"][n, 0, :,
            #                                             :].to("cpu"),
            #                                             replay_sample["front_camera_keyframe_intrinsics"][n, 0, :,
            #                                             :].to("cpu"),
            #                                             replay_sample["front_keyframe_rgb"][n, 0].to("cpu"), 128,
            #                                             128)
            action_trans = self.mamba_pipeline.get_action_trans(target_point,
                                                                target_rgb, 128,
                                                                128)
            # for n in range(4):
            #     vis_point_cloud(temp_vis_pc[n], wpt[n].to("cpu"))
        #########################################
        if backprop:
            with autocast(enabled=self.amp):
                # cross-entropy loss
                trans_loss = self._cross_entropy_loss(hm_result, action_trans).mean()
                print(f"loss:{trans_loss}")
                # trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0

                self.add_rgc_loss = False  # todo
                if self.add_rgc_loss:
                    rot_loss_x = self._cross_entropy_loss(
                        rot_q[
                        :,
                        0 * self._num_rotation_classes: 1 * self._num_rotation_classes,
                        ],
                        action_rot_x_one_hot.argmax(-1),
                    ).mean()

                    rot_loss_y = self._cross_entropy_loss(
                        rot_q[
                        :,
                        1 * self._num_rotation_classes: 2 * self._num_rotation_classes,
                        ],
                        action_rot_y_one_hot.argmax(-1),
                    ).mean()

                    rot_loss_z = self._cross_entropy_loss(
                        rot_q[
                        :,
                        2 * self._num_rotation_classes: 3 * self._num_rotation_classes,
                        ],
                        action_rot_z_one_hot.argmax(-1),
                    ).mean()

                    grip_loss = self._cross_entropy_loss(
                        grip_q,
                        action_grip_one_hot.argmax(-1),
                    ).mean()

                    collision_loss = self._cross_entropy_loss(
                        collision_q, action_collision_one_hot.argmax(-1)
                    ).mean()

                total_loss = (
                        trans_loss
                        + rot_loss_x
                        + rot_loss_y
                        + rot_loss_z
                        + grip_loss
                        + collision_loss
                )

            self._optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()
            self._lr_sched.step()

            # loss_log = {
            #     "total_loss": total_loss.item(),
            #     "trans_loss": trans_loss.item(),
            #     "rot_loss_x": rot_loss_x.item(),
            #     "rot_loss_y": rot_loss_y.item(),
            #     "rot_loss_z": rot_loss_z.item(),
            #     "grip_loss": grip_loss.item(),
            #     "collision_loss": collision_loss.item(),
            #     "lr": self._optimizer.param_groups[0]["lr"],
            # }

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": 0,
                "rot_loss_y": 0,
                "rot_loss_z": 0,
                "grip_loss": 0,
                "collision_loss": 0,
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        if eval_log:
            with torch.no_grad():
                wpt = torch.cat([x.unsqueeze(0) for x in wpt])
                pred_wpt, pred_rot_quat, _, _ = self.get_pred(
                    out,
                    rot_q,
                    grip_q,
                    collision_q,
                    y_q,
                    rev_trans,
                    dyn_cam_info=dyn_cam_info,
                )

                return_log = manage_eval_log(
                    self=self,
                    tasks=tasks,
                    wpt=wpt,
                    pred_wpt=pred_wpt,
                    action_rot=action_rot,
                    pred_rot_quat=pred_rot_quat,
                    action_grip_one_hot=action_grip_one_hot,
                    grip_q=grip_q,
                    action_collision_one_hot=action_collision_one_hot,
                    collision_q=collision_q,
                    reset_log=reset_log,
                )

                return_out.update(return_log)

        return return_out

    @torch.no_grad()
    def act(
            self, step: int, observation: dict, deterministic=True, pred_distri=False
    ) -> ActResult:
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
            lang_goal_embs = lang_goal_embs.float()
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )

        proprio = arm_utils.stack_on_channel(observation["low_dim_state"])

        obs, pcd = peract_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )

        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )

        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=0,  # no img augmentation while acting
        )
        _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
            out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
        )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )

        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
            )
        )
        if pred_distri:
            x_distri = rot_grip_q[
                       0,
                       0 * self._num_rotation_classes: 1 * self._num_rotation_classes,
                       ]
            y_distri = rot_grip_q[
                       0,
                       1 * self._num_rotation_classes: 2 * self._num_rotation_classes,
                       ]
            z_distri = rot_grip_q[
                       0,
                       2 * self._num_rotation_classes: 3 * self._num_rotation_classes,
                       ]
            return ActResult(continuous_action), (
                x_distri.cpu().numpy(),
                y_distri.cpu().numpy(),
                z_distri.cpu().numpy(),
            )
        else:
            return ActResult(continuous_action)

    def get_pred(
            self,
            out,
            rot_q,
            grip_q,
            collision_q,
            y_q,
            rev_trans,
            dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q
        )

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                :,
                0 * self._num_rotation_classes: 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                :,
                1 * self._num_rotation_classes: 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                :,
                2 * self._num_rotation_classes: 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    @torch.no_grad()
    def get_action_trans(
            self,
            wpt_local,  # torch.Size([4, 3])
            pts,
            out,
            dyn_cam_info,
            dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),  # torch.Size([4, 1,3])
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )  # returns the location of a point on the image of the cameras
        assert wpt_img.shape[1] == 1  # torch.Size([4, 1, 5, 2])
        # if self.stage_two:
        #     wpt_img2 = self._net_mod.get_pt_loc_on_img(
        #         wpt_local.unsqueeze(1),
        #         mvt1_or_mvt2=False,
        #         dyn_cam_info=dyn_cam_info,
        #         out=out,
        #     )
        #     assert wpt_img2.shape[1] == 1
        #
        #     # (bs, 1, 2 * num_img, 2)
        #     wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
        #     nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)  # torch.Size([4, 5, 2])

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),  # torch.Size([20, 2])
            (h, w),
            sigma=self.gt_hm_sigma,  # 1.5
            thres_sigma_times=3,
        )  # torch.Size([20, 220, 220])

        vis_action_trans = action_trans.clone()
        vis_action_trans = vis_action_trans.view(bs, nc, h, w)

        # import pdb;
        # pdb.set_trace()
        # import matplotlib.pyplot as plt
        #
        # for _bs in range(0, bs):
        #     for _nc in range(0, nc):
        #         # 获取要可视化的二维矩阵 (h, w)
        #         heatmap = vis_action_trans[_bs, _nc, :, :].cpu().detach().numpy()
        #         # 绘制热图
        #         plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        #         plt.colorbar()  # 添加颜色条，显示不同值的范围
        #         plt.title(f"Heatmap for batch {_bs}, num_camera {_nc}")
        #         plt.show()

        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()  # torch.Size([4, 48400, 5])

        return action_trans

    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

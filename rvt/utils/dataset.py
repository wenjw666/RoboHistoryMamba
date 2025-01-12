# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
import os
import torch
import pickle
import logging
import numpy as np
from typing import List

import clip
import peract_colab.arm.utils as utils

from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from rvt.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS
from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from rvt.libs.peract.helpers.utils import extract_obs

import random


def create_replay(
        batch_size: int,
        timesteps: int,
        disk_saving: bool,
        cameras: list,
        voxel_sizes,
        replay_size=3e5,
):
    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # 历史相机观测
    for history_idx in range(4):
        observation_elements.append(
            ObservationElement(f"low_dim_state_hisId{history_idx}", (LOW_DIM_SIZE,), np.float32)
        )
        for cname in cameras:
            observation_elements.append(
                ObservationElement(
                    f"%s_rgb_hisId{history_idx}" % cname,
                    (
                        3,
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                    ),
                    np.float32,
                )
            )
            observation_elements.append(
                ObservationElement(
                    f"%s_depth_hisId{history_idx}" % cname,
                    (
                        1,
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                    ),
                    np.float32,
                )
            )
            observation_elements.append(
                ObservationElement(
                    f"%s_point_cloud_hisId{history_idx}" % cname,
                    (
                        3,
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                    ),
                    np.float32,
                )
            )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
            observation_elements.append(
                ObservationElement(
                    f"%s_camera_extrinsics_hisId{history_idx}" % cname,
                    (
                        4,
                        4,
                    ),
                    np.float32,
                )
            )
            observation_elements.append(
                ObservationElement(
                    f"%s_camera_intrinsics_hisId{history_idx}" % cname,
                    (
                        3,
                        3,
                    ),
                    np.float32,
                )
            )

    # 关键帧信息
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_camera_keyframe_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_keyframe_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                f"%s_keyframe_rgb" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                f"%s_keyframe_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            # ReplayElement("ignore_collisions_tm1", (ignore_collisions_size,), np.int32),

            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement(
                "lang_goal_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
    ]

    replay_buffer = (
        UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
        obs_tp1: Observation,
        obs_tm1: Observation,
        rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])  # 正则化四元数
    if quat[-1] < 0:
        quat = -quat  # 最后一位为正
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)  # 得到离散的旋转角度
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)  # 当前帧碰撞位
    for depth, vox_size in enumerate(
            voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)  # 夹爪位置坐标体素化
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,  # 关键帧体素坐标
        rot_and_grip_indicies,  # 关键帧旋转与夹爪状态
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),  # 平移旋转和夹爪状态
        attention_coordinates,  # 缩放后的平移坐标
    )


# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb


# add individual data points to a replay
def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        task: str,
        task_replay_storage_folder: str,
        episode_idx: int,
        sample_frame: int,
        inital_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool,
        next_keypoint_idx: int,
        description: str = "",
        clip_model=None,
        device="cpu",
):
    def sample_between_ids(id_start, id_end, n):
        """
        在两个顺序 ID 值之间随机均匀采样 n 个整数

        参数:
        id_start (int): 起始 ID 值
        id_end (int): 结束 ID 值
        n (int): 采样的数量

        返回:
        list: 随机采样后的整数 ID 列表
        """
        available_range = list(range(id_start + 1, id_end))
        # 如果间隔内的数量小于 n
        if len(available_range) < n:
            # 先将间隔内所有值加入结果
            samples = available_range
            # 用 id_start 补齐到 n 个
            samples += [id_start] * (n - len(samples))
        else:
            # 间隔内数量足够，随机采样 n 个
            samples = random.sample(available_range, n)
        samples.sort()  # 可选：如果需要结果按顺序排列
        return samples

    global obs_tp1, obs_tm1
    prev_action = None
    obs = inital_obs
    for k in range(
            next_keypoint_idx, len(episode_keypoints)
    ):  # confused here, it seems that there are many similar samples in the replay
        # 打印帧id信息
        print(
            f"使用episode_idx:{episode_idx}\n 共有关键帧如下{episode_keypoints}\n next_keypoint_idx:{next_keypoint_idx}\t "
            f"for循环k:{k},所使用的关键帧obs_tp1 id:{episode_keypoints[k]},历史帧为obs_tm1 id:{max(0, episode_keypoints[k] - 1)}\n"
            f"保存动作为当前关键帧动作，观测为上一关键帧观测")

        # 采样历史帧
        history_id = sample_between_ids(episode_keypoints[k - 1] if k != 0 else 0, episode_keypoints[k], 4)
        print(f"采样历史帧id为：{history_id}")
        # 提取历史帧观测
        # history_obs = [demo[history_idx] for history_idx in history_id]
        history_obs_dict_temp = [extract_obs(
            demo[history_idx],  # 上一关键帧的观测
            CAMERAS,
            t=k - next_keypoint_idx,  # 当前关键帧在所有关键帧中的位置 #TODO 时间戳暂不修改
            prev_action=prev_action,  # 没用
            episode_length=25,  # ？
        )
            for history_idx in history_id]
        history_obs_dict = []
        for idx, his_obs_dictin in enumerate(history_obs_dict_temp):
            history_obs_dict.append({f"{key}_hisId{idx}": value for key, value in his_obs_dictin.items()})
            history_obs_dict[idx].pop(f"ignore_collisions_hisId{idx}", None)

        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]  # 关键帧数据
        obs_tm1 = demo[max(0, keypoint - 1)]  # 上一针或第一针数据
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,  # 从obs_tm1中提取
            action,
            attention_coordinates,
        ) = _get_action(
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        terminal = k == len(episode_keypoints) - 1  # 最后一帧
        reward = float(terminal) * 1.0 if terminal else 0  # 最后一帧reward

        # TODO: 修改此处的图像读取
        obs_dict = extract_obs(
            obs,  # 上一关键帧的观测
            CAMERAS,
            t=k - next_keypoint_idx,  # 当前关键帧在所有关键帧中的位置
            prev_action=prev_action,  # 没用
            episode_length=25,  # ？
        )
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        if k == 0:  # 第一个关键帧之前的第一次循环
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        others = {
            "demo": True,
            "keypoint_idx": k,  # 第几个关键帧
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,  # 上一个关键帧索引
            "next_keypoint_frame": keypoint,  # 下一个关键帧索引
            "sample_frame": sample_frame,  # 观测来源关键帧索引
        }
        final_obs = {  # 关键帧动作
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
        }

        obs_dict_tp1 = extract_obs(
            obs_tp1,  # 上一关键帧的观测
            CAMERAS,
            t=k - next_keypoint_idx,  # 当前关键帧在所有关键帧中的位置
            prev_action=prev_action,  # 没用
            episode_length=25,  # ？
        )
        keyframe_camera_info = {
            "front_camera_keyframe_extrinsics": obs_dict_tp1["front_camera_extrinsics"],
            "front_camera_keyframe_intrinsics": obs_dict_tp1["front_camera_intrinsics"],
            "left_shoulder_camera_keyframe_extrinsics": obs_dict_tp1["left_shoulder_camera_extrinsics"],
            "left_shoulder_camera_keyframe_intrinsics": obs_dict_tp1["left_shoulder_camera_intrinsics"],
            "right_shoulder_camera_keyframe_extrinsics": obs_dict_tp1["right_shoulder_camera_extrinsics"],
            "right_shoulder_camera_keyframe_intrinsics": obs_dict_tp1["right_shoulder_camera_intrinsics"],
            "wrist_camera_keyframe_extrinsics": obs_dict_tp1["wrist_camera_extrinsics"],
            "wrist_camera_keyframe_intrinsics": obs_dict_tp1["wrist_camera_intrinsics"]
        }
        keyframe_camera_obs = {
            "left_shoulder_keyframe_rgb": obs_dict_tp1["left_shoulder_rgb"],
            "right_shoulder_keyframe_rgb": obs_dict_tp1["right_shoulder_rgb"],
            "wrist_keyframe_rgb": obs_dict_tp1["wrist_rgb"],
            "front_keyframe_rgb": obs_dict_tp1["front_rgb"],
            "left_shoulder_keyframe_point_cloud": obs_dict_tp1["left_shoulder_point_cloud"],
            "right_shoulder_keyframe_point_cloud": obs_dict_tp1["right_shoulder_point_cloud"],
            "wrist_keyframe_point_cloud": obs_dict_tp1["wrist_point_cloud"],
            "front_keyframe_point_cloud": obs_dict_tp1["front_point_cloud"],
        }

        others.update(final_obs)
        others.update(obs_dict)  # 采样帧观测
        others.update(keyframe_camera_info)  # 关键帧相机参数
        others.update(keyframe_camera_obs)

        # 添加历史帧观测
        # obs_dict_tm1 = extract_obs(
        #     obs_tm1,  # 上一关键帧的观测
        #     CAMERAS,
        #     t=k - next_keypoint_idx,  # 当前关键帧在所有关键帧中的位置
        #     prev_action=prev_action,  # 没用
        #     episode_length=25,  # ？
        # )
        # obs_dict_tm1 = {f"{key}_tm1": value for key, value in obs_dict_tm1.items()}
        for his_obs_dict in history_obs_dict:
            others.update(his_obs_dict)

        #保存replay文件
        timeout = False
        replay.add(  # 每add一回就回生成一个replay文件
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        obs = obs_tp1
        sample_frame = keypoint

    # 打印帧id信息
    print(
        f"[FINAL replay] 使用episode_idx:{episode_idx}\n 保存最后一个关键帧观测:{episode_keypoints[-1]}\n 保存动作为最后一个关键帧动作,历史帧为demo["
        f"max(0, keypoint - 1)]:{max(0, episode_keypoints[-1] - 1)}")

    # final stepe
    obs_dict_tp1 = extract_obs(
        obs_tp1,  # 最后一个关键帧的观测
        CAMERAS,
        t=k + 1 - next_keypoint_idx,
        prev_action=prev_action,
        episode_length=25,
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
    obs_dict_tp1.pop("wrist_world_to_cam", None)
    keyframe_camera_info = {
        "front_camera_keyframe_extrinsics": obs_dict_tp1["front_camera_extrinsics"],
        "front_camera_keyframe_intrinsics": obs_dict_tp1["front_camera_intrinsics"],
        "left_shoulder_camera_keyframe_extrinsics": obs_dict_tp1["left_shoulder_camera_extrinsics"],
        "left_shoulder_camera_keyframe_intrinsics": obs_dict_tp1["left_shoulder_camera_intrinsics"],
        "right_shoulder_camera_keyframe_extrinsics": obs_dict_tp1["right_shoulder_camera_extrinsics"],
        "right_shoulder_camera_keyframe_intrinsics": obs_dict_tp1["right_shoulder_camera_intrinsics"],
        "wrist_camera_keyframe_extrinsics": obs_dict_tp1["wrist_camera_extrinsics"],
        "wrist_camera_keyframe_intrinsics": obs_dict_tp1["wrist_camera_intrinsics"],
    }
    keyframe_camera_obs = {
        "left_shoulder_keyframe_rgb": obs_dict_tp1["left_shoulder_rgb"],
        "right_shoulder_keyframe_rgb": obs_dict_tp1["right_shoulder_rgb"],
        "wrist_keyframe_rgb": obs_dict_tp1["wrist_rgb"],
        "front_keyframe_rgb": obs_dict_tp1["front_rgb"],
        "left_shoulder_keyframe_point_cloud": obs_dict_tp1["left_shoulder_point_cloud"],
        "right_shoulder_keyframe_point_cloud": obs_dict_tp1["right_shoulder_point_cloud"],
        "wrist_keyframe_point_cloud": obs_dict_tp1["wrist_point_cloud"],
        "front_keyframe_point_cloud": obs_dict_tp1["front_point_cloud"],

    }
    # 添加历史帧观测
    # obs_dict_tm1 = extract_obs(
    #     obs_tm1,  # 上一关键帧的观测
    #     CAMERAS,
    #     t=k - next_keypoint_idx,  # 当前关键帧在所有关键帧中的位置
    #     prev_action=prev_action,  # 没用
    #     episode_length=25,  # ？
    # )
    # obs_dict_tm1 = {f"{key}_tm1": value for key, value in obs_dict_tm1.items()}
    for his_obs_dict in history_obs_dict:
        obs_dict_tp1.update(his_obs_dict)
    obs_dict_tp1.update(keyframe_camera_info)
    obs_dict_tp1.update(final_obs)  # 最后一个关键帧的动作
    obs_dict_tp1.update(keyframe_camera_obs)

    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)  # 每add一回就回生成一个replay文件


def fill_replay(
        replay: ReplayBuffer,
        task: str,
        task_replay_storage_folder: str,
        start_idx: int,
        num_demos: int,
        demo_augmentation: bool,
        demo_augmentation_every_n: int,
        cameras: List[str],
        rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool,
        data_path: str,
        episode_folder: str,
        variation_desriptions_pkl: str,
        clip_model=None,
        device="cpu",
):
    import pdb;
    pdb.set_trace()
    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...")
        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )  # 'data/train/open_drawer/all_variations/episodes/episode0/variation_descriptions.pkl'
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(
                    f)  # ['open the bottom drawer', 'grip the bottom handle and pull the bottom drawer open', 'slide the bottom drawer open']

            # extract keypoints
            episode_keypoints = keypoint_discovery(demo)
            next_keypoint_idx = 0
            for i in range(len(demo) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame 整除10的帧进行处理
                    continue

                obs = demo[i]
                desc = descs[0]
                # if our starting point is past one of the keypoints, then remove it
                while (
                        next_keypoint_idx < len(episode_keypoints)
                        and i >= episode_keypoints[next_keypoint_idx]  # 当帧数i超过当前指示的下一个关键点索引，将关键点索引加一，索引不超过关键点列表长度
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):  #  相等说明已处理完
                    break
                _add_keypoints_to_replay(
                    replay,
                    task,  # 'open_drawer'
                    task_replay_storage_folder,  # 'replay/replay_train/open_drawer'
                    d_idx,  # episode id
                    i,  # 帧数
                    obs,  # 当前帧观测
                    demo,  # 当前帧demo数据
                    episode_keypoints,  # 关键帧索引列表
                    cameras,  # ['front', 'left_shoulder', 'right_shoulder', 'wrist'] 所用相机名
                    rlbench_scene_bounds,  # [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]  #深度范围
                    voxel_sizes,  # [100]
                    rotation_resolution,  # 5
                    crop_augmentation,  # false
                    next_keypoint_idx=next_keypoint_idx,
                    description=desc,
                    clip_model=clip_model,
                    device=device,
                )

        # save TERMINAL info in replay_info.npy
        task_idx = replay._task_index[task]
        with open(
                os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        ) as fp:
            np.save(
                fp,
                replay._store["terminal"][
                replay._task_replay_start_index[task_idx]: replay._task_replay_start_index[task_idx] +
                                                           replay._task_add_count[task_idx].value],
            )

        print("Replay filled with demos.")

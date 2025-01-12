# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import time
import tqdm
import random
import yaml
import argparse

from collections import defaultdict
from contextlib import redirect_stdout

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import config as exp_cfg_mod
import rvt.models.rvt_agent as rvt_agent
import rvt.utils.ddp_utils as ddp_utils
import rvt.mvt.config as mvt_cfg_mod
import rvt.mvt.config_mamba as mymamba_cfg_mod

from rvt.mvt.mvt import MVT
from rvt.mvt.mambatest import MyMambaPipeline

from rvt.models.my_agent import print_eval_log, print_loss_log, print_evaluation_log
from rvt.utils.get_dataset import get_dataset
from rvt.utils.rvt_utils import (
    TensorboardManager,
    short_name,
    get_num_feat,
    load_agent,
    RLBENCH_TASKS,
)
from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
)

import rvt.models.my_agent as my_agent

def val(agent, dataset, training_iterations, rank=0):
    """
    测试agent，返回测试效果
    """
    agent.train()  # 设置训练模式
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):

        raw_batch = next(data_iter)
       
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }  # 处理tensor类型数据
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args = {
            "step": iteration,
        }
        update_args.update(
            {
                "replay_sample": batch,
                "backprop": False,
                "reset_log": (iteration == 0),
                "eval_log": True,
            }
        )
        agent.update(**update_args)
        torch.cuda.empty_cache()

    if rank == 0:
        log = print_evaluation_log(agent)

    return log

# new train takes the dataset as input
def train(agent, dataset, training_iterations, rank=0):
    agent.train()  # 设置训练模式
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):

        raw_batch = next(data_iter)


        # 假设 data 是你的数据集，确保它是可以序列化的格式
        # import json
        # data = {}
        # for key, value in raw_batch.items():
        #     if isinstance(value, torch.Tensor):
        #         data[key] = value.tolist()
        # with open('dataset.json', 'w') as f:
        #     json.dump(data, f, indent=4)
        # print(type(raw_batch))
        # for key, value in raw_batch.items():
        #     print(
        #         f"{key}: type = {type(value)}, shape = {value.size() if isinstance(value, torch.Tensor) else 'N/A'}, dtype = {value.dtype if isinstance(value, torch.Tensor) else 'N/A'}")


        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }  # 处理tensor类型数据
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args = {
            "step": iteration,
        }
        update_args.update(
            {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
                "eval_log": False,
            }
        )
        
        agent.update(**update_args)
        torch.cuda.empty_cache()

    if rank == 0:
        log = print_loss_log(agent)
        
    return log


def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )


def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks


def get_logdir(cmd_args, exp_cfg):
    log_dir = os.path.join(cmd_args.log_dir, exp_cfg.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    """
    保存当前训练参数文件
    """
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


def experiment(rank, cmd_args, devices, port):
    """experiment.

    :param rank: 显卡id
    :param cmd_args:
    :param devices: list or int. if list, we use ddp else not
    """
    device = devices[rank]
    device = f"cuda:{device}"
    ddp = len(devices) > 1  # 多卡
    ddp_utils.setup(rank, world_size=len(devices), port=port)  # 分配端口

    exp_cfg = exp_cfg_mod.get_cfg_defaults()  # 主要加载训练参数
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    if ddp:
        print(f"Running DDP on rank {rank}.")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    exp_cfg.peract.lr *= len(devices) * exp_cfg.bs
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.exp_cfg_opts)}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.mvt_cfg_opts)}"

    if rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    # Things to change
    BATCH_SIZE_TRAIN = exp_cfg.bs
    BATCH_SIZE_VAL = exp_cfg.bs_val
    NUM_TRAIN = 10
    NUM_VAL = 10
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * len(devices)))
    EPOCHS = exp_cfg.epochs
    TRAIN_REPLAY_STORAGE_DIR = "replay/replay_train"
    TEST_REPLAY_STORAGE_DIR = "replay/replay_test"
    VAL_REPLAY_STORAGE_DIR = "replay/replay_val"
    VAL_ITERATIONS = int(exp_cfg.val_iter // (exp_cfg.bs * len(devices)))    
    IF_ONLY_TRAIN = exp_cfg.only_train
    log_dir = get_logdir(cmd_args, exp_cfg)
    tasks = get_tasks(exp_cfg)


    print("Training on {} tasks: {}".format(len(tasks), tasks))

    t_start = time.time()
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        BATCH_SIZE_VAL,
        TRAIN_REPLAY_STORAGE_DIR,
        VAL_REPLAY_STORAGE_DIR,
        DATA_FOLDER,
        NUM_TRAIN,
        NUM_VAL,
        cmd_args.refresh_replay,
        device,
        num_workers=exp_cfg.num_workers,
        only_train=IF_ONLY_TRAIN,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
    )
    train_dataset, val_dataset = get_dataset_func()
    t_end = time.time()

    print("=== train_dataset 关键属性 ===")
    print(f"数据集对象: {train_dataset.dataset}")
    # print(f"数据集总样本数: {len(train_dataset.dataset)}")
    # print(f"总批次数: {len(train_dataset)}")
    print(f"每批次大小 (batch_size): {train_dataset.batch_size}")
    # print(f"是否随机打乱 (shuffle): {train_dataset.shuffle}")
    print(f"是否丢弃最后不完整的批次 (drop_last): {train_dataset.drop_last}")
    print(f"数据采样器 (sampler): {train_dataset.sampler}")
    print(f"加载数据使用的子进程数 (num_workers): {train_dataset.num_workers}")
    print(f"是否使用固定内存加速 (pin_memory): {train_dataset.pin_memory}")
    print(f"预取因子 (prefetch_factor): {getattr(train_dataset, 'prefetch_factor', 'N/A')}")
    print(f"子进程初始化函数 (worker_init_fn): {train_dataset.worker_init_fn}")
    if not IF_ONLY_TRAIN:
        print("=== val_dataset 关键属性 ===")
        print(f"数据集对象: {val_dataset.dataset}")
        # print(f"数据集总样本数: {len(val_dataset.dataset)}")
        # print(f"总批次数: {len(val_dataset)}")
        print(f"每批次大小 (batch_size): {val_dataset.batch_size}")
        # print(f"是否随机打乱 (shuffle): {val_dataset.shuffle}")
        print(f"是否丢弃最后不完整的批次 (drop_last): {val_dataset.drop_last}")
        print(f"数据采样器 (sampler): {val_dataset.sampler}")
        print(f"加载数据使用的子进程数 (num_workers): {val_dataset.num_workers}")
        print(f"是否使用固定内存加速 (pin_memory): {val_dataset.pin_memory}")
        print(f"预取因子 (prefetch_factor): {getattr(val_dataset, 'prefetch_factor', 'N/A')}")
        print(f"子进程初始化函数 (worker_init_fn): {val_dataset.worker_init_fn}")
    print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    # 读取参数
    if exp_cfg.agent == "our":
        # 加载自定义模型的模型参数
        mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
        if cmd_args.mvt_cfg_path != "":
            mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
        if cmd_args.mvt_cfg_opts != "":
            mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

        mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
        mvt_cfg.freeze()

        # for maintaining backward compatibility
        assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
            mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
        )

        mymamba_cfg = mymamba_cfg_mod.get_cfg_defaults()
        mymamba_cfg.freeze()


        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        # 实例化模型
        # rvt = MVT(
        #     renderer_device=device,
        #     **mvt_cfg,
        # ).to(device)

        mamba_pipeline = MyMambaPipeline(**mymamba_cfg.mymamba).to(device)

        # if ddp:
        #     rvt = DDP(rvt, device_ids=[device])

        # 加载agent

        # agent = rvt_agent.RVTAgent(
        #     network=rvt,
        #     image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        #     add_lang=mvt_cfg.add_lang,
        #     stage_two=mvt_cfg.stage_two,
        #     rot_ver=mvt_cfg.rot_ver,
        #     scene_bounds=SCENE_BOUNDS,
        #     cameras=CAMERAS,
        #     log_dir=f"{log_dir}/test_run/",
        #     cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
        #     **exp_cfg.peract,
        #     **exp_cfg.rvt,
        # )
        agent = my_agent.MyAgent(
            network=mamba_pipeline,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            stage_two=mvt_cfg.stage_two,
            rot_ver=mvt_cfg.rot_ver,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"{log_dir}/test_run/",
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        # 初始化优化器设置
        agent.build(training=True, device=device)
    else:
        assert False, "Incorrect agent"

    start_epoch = 0
    end_epoch = EPOCHS
    if exp_cfg.resume != "":
        agent_path = exp_cfg.resume
        print(f"Recovering model and checkpoint from {exp_cfg.resume}")
        epoch = load_agent(agent_path, agent, only_epoch=False)
        start_epoch = epoch + 1
    dist.barrier()

    if rank == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
        tb = TensorboardManager(log_dir)

    import pdb
    pdb.set_trace()
    print("Start training ...")
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")

        out = train(agent, train_dataset, TRAINING_ITERATIONS, rank)

        # if rank == 0:
        tb.update("train", i, out)

        if not IF_ONLY_TRAIN:
            print("----- val -----")
            out = val(agent, val_dataset, VAL_ITERATIONS, rank)

            tb.update("val", i, out)  
        
        # if rank == 0:
            # TODO: add logic to only save some models
        # save_agent(agent, f"{log_dir}/model_{i}.pth", i)
        # save_agent(agent, f"{log_dir}/model_last.pth", i)
        i += 1


    if rank == 0:
        tb.close()
        print("[Finish]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mvt_cfg_path", type=str, default="")
    parser.add_argument("--exp_cfg_path", type=str, default="")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")

    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--with-eval", action="store_true", default=False)

    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    devices = cmd_args.device.split(",")
    devices = [int(x) for x in devices]

    port = (random.randint(0, 3000) % 3000) + 27000
    # mp.spawn(experiment, args=(cmd_args, devices, port), nprocs=len(devices), join=True)
    device = int(cmd_args.device)
    experiment(0, cmd_args, [device], port)
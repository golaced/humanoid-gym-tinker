# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *
import copy
import torch
from tqdm import tqdm
from datetime import datetime
from global_config import ROBOT_SEL,PLAY_DIR,ROOT_DIR

import torch
import tvm
from tvm import relay


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5


    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations() #1,705 isaac

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        #path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')#任務對於/policy_1.pt
        path = "./model_jitt"

        print('***',path)
        print(ppo_runner.alg)
        print('++++')

        print(ppo_runner.alg.actor_critic)
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)

        # path = os.path.join(path, "policy_origin.pt")
        length= env_cfg.env.num_observations
        # model = copy.deepcopy(ppo_runner.alg.actor_critic.actor).to("cpu")
        # obs_demo_input = torch.rand(1,length).to(device="cpu")
        # traced_script_module = torch.jit.trace(model,(obs_demo_input))
        # traced_script_module.save(path)

        if 0:# convet TVM
            inputSize=length
            print("Convert Net Inputs:",inputSize)
            # 已经jit trace后的模型导入
            model=torch.jit.load("./model_jitt/policy_origin.pt")
            # print(model)
            model=model.float() #确保模型的所有参数使用浮动点精度（float32），这里进行强制转换为浮动点类型。
            #model=model.half()  
            shape_list = [("input0",((inputSize,),"float32"))]
            mod,param=relay.frontend.from_pytorch(model,shape_list)#TVM 的一个前端API，用于将PyTorch模型转换为TVM的Relay IR (Intermediate Representation) 格式。不仅转换了模型的计算图，还包括了参数。
            # # ==x64=======
            target = tvm.target.Target("llvm", host="llvm")#指定目标架构为x64 CPU，llvm是TVM支持的后端编译器工具链，host="llvm" 指定主机也使用LLVM。
            with tvm.transform.PassContext(opt_level=3):#指定优化级别为 3（最高优化级别），以尽可能优化模型的执行性能
                lib = relay.build(mod, target=target, params=param)#将 Relay 中间表示编译成目标平台（x64）上可执行的共享库（so文件）
            lib.export_library("./model_jitt/policy_x64_cpu.so")#将生成的共享库导出为.so文件，这个文件可以在x64架构的 CPU 上运行

            #==arm64=======
            target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")#指定目标架构为ARM64。-mtriple=aarch64-linux-gnu用于告诉LLVM编译器目标是ARM64(aarch64)平台上的 Linux 系统
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=param)
            lib.export_library("./model_jitt/policy_arm64_cpu.so", cc='/usr/bin/aarch64-linux-gnu-g++-9')#指定了编译器cc参数为aarch64-linux-gnu-g++-9，这是一个适用于ARM64的C++编译器，通常用于交叉编译

        if 0:# convet TVM
            # 已经jit trace后的模型导入
            inputSize=length
            model=torch.jit.load("./model_jitt/policy_origin.pt")
            model=model.float() #确保模型的所有参数使用浮动点精度（float32），这里进行强制转换为浮动点类型。

            shape_list = [("input0",((inputSize,),"float32"))]
            mod,param=relay.frontend.from_pytorch(model,shape_list)#TVM 的一个前端API，用于将PyTorch模型转换为TVM的Relay IR (Intermediate Representation) 格式。不仅转换了模型的计算图，还包括了参数。

            # # ==x64=======
            target = tvm.target.Target("llvm", host="llvm")#指定目标架构为x64 CPU，llvm是TVM支持的后端编译器工具链，host="llvm" 指定主机也使用LLVM。
            with tvm.transform.PassContext(opt_level=3):#指定优化级别为 3（最高优化级别），以尽可能优化模型的执行性能
                lib = relay.build(mod, target=target, params=param)#将 Relay 中间表示编译成目标平台（x64）上可执行的共享库（so文件）
            lib.export_library("./model_jitt/policy_x64_cpu.so")#将生成的共享库导出为.so文件，这个文件可以在x64架构的 CPU 上运行

            #==arm64=======
            target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")#指定目标架构为ARM64。-mtriple=aarch64-linux-gnu用于告诉LLVM编译器目标是ARM64(aarch64)平台上的 Linux 系统
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=param)
            lib.export_library("./model_jitt/policy_arm64_cpu.so", cc='/usr/bin/aarch64-linux-gnu-g++-9')#指定了编译器cc参数为aarch64-linux-gnu-g++-9，这是一个适用于ARM64的C++编译器，通常用于交叉编译

        if 1:
            inputSize=length
            model=torch.jit.load("./model_jitt/policy_origin.pt")
            model=torch.jit.trace(model,torch.randn(1,inputSize)).eval()
            model=torch.jit.optimize_for_inference(torch.jit.script(model))
            shape_list = [("input0",((inputSize,),"float32"))]
            mod,param=relay.frontend.from_pytorch(model,shape_list)

            # ==x64=======
            from tvm.relay.transform import ToMixedPrecision
            mixed_precision_mod = ToMixedPrecision("float16")(mod)
            # from tvm.relay import quantize
            # with quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            #     mixed_precision_mod = quantize.quantize(mod, params=param)
            #mixed_precision_mod = mod
            target = tvm.target.Target("llvm", host="llvm")
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mixed_precision_mod, target=target, params=param)
            lib.export_library("./model_jitt/policy_x64_cpu.so")

            #==arm64=======
            target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mixed_precision_mod, target=target, params=param)
            lib.export_library("./model_jitt/policy_arm64_cpu.so", cc='/usr/bin/aarch64-linux-gnu-g++-9')

        # action_demo_input = torch.rand(1,5,12).to(device=env.device)
        # obs_demo_input = torch.rand(1,5,11).to(device=env.device)
        # model_jit = torch.jit.trace(ppo_runner.alg.actor_critic,(obs_demo_input,action_demo_input))
        # model_jit.save(path)

        #torch.save( ppo_runner.alg.actor_critic.actor,'model_jitt.pt')
        print('dddExported policy as jit script to: ', path)

        # model_dict = torch.load(os.path.join(LEGGED_GYM_ROOT_DIR, PLAY_DIR))#《---------------------调用的网络模型doghome
        # policy.load_state_dict(model_dict['model_state_dict'])
        # policy.half()
        # policy = policy.to(env.device)
        # policy.save_torch_jit_policy('model_jitt.pt',env.device)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 1200 # number of steps before plotting states
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    for i in tqdm(range(stop_state_log)):

        actions = policy(obs.detach()) # * 0.

        if FIX_COMMAND:
            env.commands[:, 0] = 0.5    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        logger.log_states(
            {
                'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            }
            )
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)

    logger.print_rewards()
    logger.plot_states()

    if RENDER:
        video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = False
    args = get_args()
    args.task= ROBOT_SEL#'humanoid_ppo'
    args.run_name='v1'
    play(args)
import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime

import torch
import tvm
from tvm import relay

import configparser # export config to ini files

def export_policy_as_jit_hw(actor_critic, path, input_nums):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_origin.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.trace(model,torch.randn(1,input_nums))
    traced_script_module.save(path)

args = get_args()
args.headless=True
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task) # env_cfg corresponds to e.g. LumosCfg_s2(LeggedRobotCfg), train_cfg corresponds to e.g. LumosCfgPPO_s2(LeggedRobotCfgPPO)

# prepare environment
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg) 

# load policy
train_cfg.runner.resume = True
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
policy = ppo_runner.get_inference_policy(device=env.device)

print(env_cfg.env.frame_stack)
path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
export_policy_as_jit_hw(ppo_runner.alg.actor_critic, path, env_cfg.env.num_observations)
export_policy_as_jit(ppo_runner.alg.actor_critic, path)

print('Exported policy as jit script to: ', path)

inputSize=env_cfg.env.num_observations
path_to_policy = os.path.join(path, "./model_jitt/policy_origin.pt")
print(path_to_policy)
model_trace=torch.jit.load(path_to_policy)
model_trace=torch.jit.trace(model_trace,torch.randn(1,inputSize)).eval()
shape_list = [("input0",((inputSize,),"float32"))]
mod,param=relay.frontend.from_pytorch(model_trace,shape_list)

# ==x64=======
path_to_policy_hw=os.path.join(path, "policy_x64_cpu.so")

target = tvm.target.Target("llvm", host="llvm")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=param)
lib.export_library(path_to_policy_hw)

#==arm64=======
path_to_policy_hw_arm=os.path.join(path, "policy_arm64_cpu.so")
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=param)
lib.export_library(path_to_policy_hw_arm, cc='/usr/bin/aarch64-linux-gnu-g++-9')

# 创建 ConfigParser 对象
config = configparser.ConfigParser()
path_to_policy_ini=os.path.join(path, "policyConfig.ini")

# 添加配置段落和键值对
config.write(";[ctrl_para]")
config['ctrl_para'] = {
    'frame_stack': env_cfg.env.frame_stack,
    'num_single_obs': env_cfg.env.num_single_obs,
    'num_actions': env_cfg.env.num_actions,
    'control_dt': env_cfg.sim.dt*env_cfg.control.decimation,
    'cycle_time': env_cfg.rewards.cycle_time,
    'nominal_base_height': env_cfg.init_state.pos[2],
    'scales_action': env_cfg.control.action_scale,
    'scales_lin_vel': env_cfg.normalization.obs_scales.lin_vel,
    'scales_ang_vel': env_cfg.normalization.obs_scales.ang_vel,
    'scales_dof_pos': env_cfg.normalization.obs_scales.dof_pos,
    'scales_dof_vel': env_cfg.normalization.obs_scales.dof_vel,
    'clip_actions': env_cfg.normalization.clip_actions,
    'clip_observations': env_cfg.normalization.clip_observations
}

config['default_joint_angles'] = {key: str(value) for key, value in env_cfg.init_state.default_joint_angles.items()}

config['default_joint_stiffness'] = {f"{key}-kp": str(value) for key, value in env_cfg.control.stiffness.items()}

config['default_joint_damping'] = {f"{key}-kd": str(value) for key, value in env_cfg.control.damping.items()}

# 导出到 ini 文件
with open(path_to_policy_ini, 'w') as configfile:
    config.write(configfile)

configfile.close()

# comment segment titles
input_file = path_to_policy_ini
output_file = input_file

with open(input_file, 'r') as infile:
    lines = infile.readlines()

with open(output_file, 'w') as configfile:
    for line in lines:
        stripped_line = line.strip()
        # 如果是段落标题，即以 [ 开始并以 ] 结束，则进行注释
        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            configfile.write(f";{line}")
        else:
            configfile.write(line)

print("配置已导出到 ini 文件")



## to compare the output between jit trace and jit script
# path_to_policy = os.path.join(path, "policy_hw.pt")
# print(path_to_policy)
# model_trace=torch.jit.load(path_to_policy)
# model_trace=torch.jit.trace(model_trace,torch.randn(1,inputSize)).eval()

# path_to_policy = os.path.join(path, "policy_1.pt")
# print(path_to_policy)
# model_script=torch.jit.load(path_to_policy)
# model_script=torch.jit.script(model_script,torch.randn(1,inputSize)).eval()

# input_1 = torch.randn(1, inputSize)
# input_2 = torch.zeros(1, inputSize)

# # 使用 torch.jit.trace
# traced_model = torch.jit.trace(model_trace, input_1)
# print("Trace (input_1):", traced_model(input_1))
# print("Trace (input_2):", traced_model(input_2))

# # 使用 torch.jit.script
# scripted_model = torch.jit.script(model_script)
# print("Script (input_1):", scripted_model(input_1))
# print("Script (input_2):", scripted_model(input_2))
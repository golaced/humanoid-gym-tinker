# SPDX-License-Identifier: BSD-3-Clause
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# ... (license text omitted for brevity) ...

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import TinkerCfg
from global_config import SPD_X, SPD_Y, SPD_YAW
import tvm
from tvm.contrib import graph_executor

# Default joint positions
default_dof_pos = [0.0, 0.1, 0.56, -1.12, -0.57, 0.0, -0.1, -0.56, 1.12, 0.57]

class cmd:
    vx = SPD_X
    vy = SPD_Y
    dyaw = SPD_YAW

def quaternion_to_euler_array(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy_module, cfg):
    global default_dof_pos
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating"):
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:15] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            obs[0, 15:25] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 25:35] = action
            obs[0, 35:38] = omega * cfg.normalization.obs_scales.ang_vel
            obs[0, 38:41] = eu_ang * cfg.normalization.obs_scales.quat
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs:(i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

            policy_module.set_input("input0", tvm.nd.array(policy_input))
            policy_module.run()
            tvm_output = policy_module.get_output(0).asnumpy()
            action[:] = tvm_output
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            target_q = action * cfg.control.action_scale + default_dof_pos

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()

def load_tvm_graph_module(model_path):
    lib = tvm.runtime.load_module(model_path)
    dev = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](dev))
    return module

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default='./model_jitt/policy_x64_cpu.so', help='TVM model path.')
    parser.add_argument('--terrain', action='store_true', help='Use terrain or plane.')
    args = parser.parse_args()

    class Sim2simCfg(TinkerCfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/tinker/xml/world_terrain.xml' if args.terrain else f'{LEGGED_GYM_ROOT_DIR}/resources/robots/tinker/xml/world.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 20
        class robot_config:
            kp_all = 15.0
            kd_all = 0.65
            kps = np.array([kp_all] * 10, dtype=np.double)
            kds = np.array([kd_all] * 10, dtype=np.double)
            tau_limit = 30. * np.ones(10, dtype=np.double)

    policy_module = load_tvm_graph_module(args.load_model)
    run_mujoco(policy_module, Sim2simCfg())
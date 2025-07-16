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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from global_config import MAX_ITER,SAVE_DIV,PLAY_DIR

class TinkerCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 41 #依据URDF修改 依据观测进行修改
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 55+10+3*10   #依据URDF修改 依据观测进行修改 RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x203 and 219x768)
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 10 #依据URDF修改
        num_envs = 1024
        episode_length_s = 20 #24  # episode length in seconds  训练时间长度
        use_ref_actions = False #前馈动作
        env_spacing = 1.  # not used with heightfields/trimeshes  doghome

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tinker/urdf/tinker.urdf'

        name = "Tinker-L"
        foot_name = "ankle"
        knee_name = "ankle"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter 自己碰撞
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = True
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.33]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'J_L0':   0.0,   # [rad]
            'J_L1':  0.08,   # [rad]
            'J_L2':  0.56,   # [rad]
            'J_L3':  -1.12,   # [rad]
            'J_L4_ankle': -0.57,   # [rad]

            'J_R0':   0.0,   # [rad]
            'J_R1':  -0.08,   # [rad]
            'J_R2':  -0.56,   # [rad]
            'J_R3':  1.12,   # [rad]
            'J_R4_ankle': 0.57,   # [rad] 
        }

    class control(LeggedRobotCfg.control):
        use_filter = True
        control_type= "P"
        # PD Drive parameters:
        stiffness = {'J_L0': 15, 'J_L1': 15,'J_L2': 15, 'J_L3':15, 'J_L4_ankle':15,
                     'J_R0': 15, 'J_R1': 15,'J_R2': 15, 'J_R3':15, 'J_R4_ankle':15}
        damping = {'J_L0': 0.65, 'J_L1': 0.65,'J_L2': 0.65, 'J_L3':0.65, 'J_L4_ankle':0.65,
                   'J_R0': 0.65, 'J_R1': 0.65,'J_R2': 0.65, 'J_R3':0.65, 'J_R4_ankle':0.65}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  #1 4=50hz 

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.3, 2.5]
        randomize_base_mass = True
        added_mass_range = [-0.15, 0.5]
        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_noise = 0.015

        randomize_base_com = True
        added_com_range = [-0.02, 0.02]

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]#比例系数

        randomize_kpkd = True
        kp_range = [0.8,1.2]#比例系数
        kd_range = [0.8,1.2]

        randomize_lag_timesteps = True
        add_lag = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 40]
        #old mass randomize new------------------------------
        randomize_all_mass = True
        rd_mass_range = [0.9, 1.1]

        randomize_com = True #link com
        rd_com_range = [-0.02, 0.02]
    
        random_inertia = True
        inertia_range = [0.9, 1.1]

        randomize_motor_offset = True
        motor_offset_range = [-0.045, 0.045] # Offset to add to the motor angles

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.25
        min_dist = 0.2
        max_dist = 0.6
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.2    # rad 前馈弧度
        target_feet_height = 0.08      # m
        cycle_time = 0.5                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 1
        max_contact_force = 180  # Forces above this value are penalized

        feet_off_z=0.0487
        class scales:
            joint_pos = 0.1
            #feet_clearance = 1
            feet_swing_trajectory_z = 2
            feet_swing_height =  1
            feet_contact_number = 1.5
            # gait
            feet_air_time = 1.5
            foot_slip = -0.05
            # contact
            feet_contact_forces = -0.01
            feet_rotation1 = 0.5
            feet_rotation2 = 0.5
            # vel tracking
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.1
            #vel_mismatch_exp = 0.5  # lin_z; ang x,y
            lin_vel_z = -2.0
            ang_vel_xy = -0.02
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            #default_joint_pos = 0.5
            hip_pos = -2
            #orientation = 1.
            lin_vel_z = -2.0
            ang_vel_xy = -0.02
            orientation_eular=1.0 # 0.05可以探索爬行
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.02
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -2e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
 
class TinkerCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001
        num_learning_epochs = 5
        num_mini_batches = 4    # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 1.e-4
        schedule = 'adaptive'   # could be adaptive, fixed
        gamma = 0.98
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        # entropy_coef = 0.001
        # learning_rate = 1e-5
        # num_learning_epochs = 2
        # gamma = 0.994
        # lam = 0.9
        # num_mini_batches = 4       

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = MAX_ITER  # number of policy updates

        # logging
        save_interval = SAVE_DIV  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'Tinker_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1 #-1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt


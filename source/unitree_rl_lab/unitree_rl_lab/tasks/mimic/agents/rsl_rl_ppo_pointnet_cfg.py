# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PointNetActorCriticCfg(RslRlPpoActorCriticCfg):
    """扩展 ActorCritic 配置，添加 PointNet 所需的额外参数。"""
    num_proprioception: int = 30
    num_points: int = 256


# ==============================================================================
# Phase 1: 基础运动学预训练配置 (Kinematic Pre-training)
# ==============================================================================
@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ==============================================================================
# Phase 2: 残差策略与视觉闭环配置 (Residual Policy + PointNet)
# ==============================================================================
@configclass
class PointNetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """专门针对带有 PointNet 视觉前端和残差动作的 PPO 训练配置"""

    num_steps_per_env = 24
    # 残差微调阶段通常收敛更快，不需要 30000 这么多次迭代
    max_iterations = 6000
    save_interval = 400
    experiment_name = ""
    empirical_normalization = False

    # 【核心修改】：替换默认的 Policy 为我们自定义的 PointNet Policy
    policy = PointNetActorCriticCfg(
        class_name="unitree_rl_lab.tasks.mimic.robots.piper.turn_handkerchief_rpl.pointnet_policy.PointNetActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # motion_command(12) + joint_pos_rel(6) + joint_vel_rel(6) + last_action(6) = 30
        num_proprioception=30,
        num_points=256,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 视觉任务初期需要更多的探索，适当调高 entropy_coef
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        # 残差微调阶段的初始学习率可以比 Phase 1 稍微大一点或保持一致
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ==============================================================================
# Phase 2 Ablation: 残差策略，无 Point Cloud / PointNet actor 输入
# ==============================================================================
@configclass
class NoPointCloudPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """SoftMimic w/o Point Cloud: residual PPO with a plain MLP actor."""

    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 400
    experiment_name = "piper_turn_handkerchief_rpl_no_pointcloud"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

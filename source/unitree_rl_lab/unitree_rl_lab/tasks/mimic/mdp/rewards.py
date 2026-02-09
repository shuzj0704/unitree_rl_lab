from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def motion_relative_joint_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, k: float, std: float) -> torch.Tensor:
    """关节位置追踪奖励
    公式: exp[-k * Σ||q_t^j ⊖ q̂_t^j||² / σ²] , 其中 k=1.0
    """
    from isaaclab.assets import Articulation
    
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[command.cfg.asset_name]
    
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    joint_indices = [asset.joint_names.index(name) for name in joint_names]
    
    target_joint_pos = command.joint_pos
    current_joint_pos = asset.data.joint_pos[:, joint_indices]
    
    pos_error = torch.sum(torch.square(target_joint_pos - current_joint_pos), dim=-1)
    
    return torch.exp(-k * pos_error / (std**2))


def motion_relative_joint_velocity_tracking_exp(env: ManagerBasedRLEnv, command_name: str, k: float, std: float) -> torch.Tensor:
    """关节速度追踪奖励
    公式: exp[-k * Σ(Δq̇)² / σ²], 其中 k=0.3
    """
    from isaaclab.assets import Articulation
    
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[command.cfg.asset_name]
    
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    joint_indices = [asset.joint_names.index(name) for name in joint_names]
    
    target_joint_vel = command.joint_vel
    current_joint_vel = asset.data.joint_vel[:, joint_indices]
    
    vel_error = torch.sum(torch.square(target_joint_vel - current_joint_vel), dim=-1)
    
    return torch.exp( -k * vel_error / (std**2))
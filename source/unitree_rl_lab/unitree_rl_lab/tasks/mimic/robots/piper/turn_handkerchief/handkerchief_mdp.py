"""Handkerchief-specific MDP terms for the turn_handkerchief task.

This module contains observation, reward, termination, and event functions
related to the deformable handkerchief object. Kept separate from the shared
mimic MDP module to avoid mixing task-specific logic with general-purpose code.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, DeformableObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==============================================================================
# Helper
# ==============================================================================

STICK_LENGTH = 0.17  # length of the stick attached to link6 (metres)


def _get_stick_tip_state(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the stick-tip position and velocity in world frame.

    Returns:
        stick_tip_pos: (num_envs, 3)
        stick_tip_vel: (num_envs, 3)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    end_body_idx = robot.find_bodies("link6")[0][0]

    pos_link_w = robot.data.body_pos_w[:, end_body_idx]
    quat_link_w = robot.data.body_quat_w[:, end_body_idx]
    com_pos_b = robot.data.com_pos_b[:, end_body_idx]
    lin_vel_w = robot.data.body_lin_vel_w[:, end_body_idx]
    ang_vel_w = robot.data.body_ang_vel_w[:, end_body_idx]

    stick_end_b = torch.tensor([0.0, 0.0, STICK_LENGTH], device=env.device)
    stick_end_b = stick_end_b.unsqueeze(0).expand(quat_link_w.shape[0], -1)

    # position
    stick_dir = quat_apply(quat_link_w, stick_end_b)
    stick_tip_pos = pos_link_w + stick_dir

    # velocity  (v_p = v_c + ω × r)
    stick_to_com_b = stick_end_b - com_pos_b
    stick_to_com_w = quat_apply(quat_link_w, stick_to_com_b)
    stick_tip_vel = lin_vel_w + torch.cross(ang_vel_w, stick_to_com_w, dim=-1)

    return stick_tip_pos, stick_tip_vel


# ==============================================================================
# Observations
# ==============================================================================

def handkerchief_root_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Handkerchief root (centre-of-mass) position in world frame.  (num_envs, 3)"""
    hk: DeformableObject = env.scene[asset_cfg.name]
    return hk.data.root_pos_w


def handkerchief_root_vel_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Handkerchief root velocity in world frame.  (num_envs, 3)"""
    hk: DeformableObject = env.scene[asset_cfg.name]
    return hk.data.root_vel_w


def stick_tip_pos_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stick-tip position in world frame.  (num_envs, 3)"""
    pos, _ = _get_stick_tip_state(env, robot_cfg)
    return pos


def stick_tip_vel_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stick-tip velocity in world frame.  (num_envs, 3)"""
    _, vel = _get_stick_tip_state(env, robot_cfg)
    return vel


def handkerchief_to_stick_tip_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    handkerchief_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Relative position vector from stick-tip to handkerchief root.  (num_envs, 3)"""
    hk: DeformableObject = env.scene[handkerchief_cfg.name]
    tip_pos, _ = _get_stick_tip_state(env, robot_cfg)
    return hk.data.root_pos_w - tip_pos


# ==============================================================================
# Rewards
# ==============================================================================

def handkerchief_spin_angular_momentum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Reward based on the z-component of the handkerchief angular momentum.

    A positive z angular-momentum means the cloth is spinning counter-clockwise
    (viewed from above).  The raw value is returned so that the reward weight
    can control whether to encourage CW or CCW rotation.
    """
    hk: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = hk.data.nodal_pos_w
    nodal_vel = hk.data.nodal_vel_w
    root_pos = hk.data.root_pos_w
    root_vel = hk.data.root_vel_w

    rel_pos = nodal_pos - root_pos.unsqueeze(1)
    rel_vel = nodal_vel - root_vel.unsqueeze(1)
    ang_mom = torch.cross(rel_pos, rel_vel, dim=-1)  # (N, nodes, 3)
    total_ang_mom = torch.sum(ang_mom, dim=1)         # (N, 3)
    return total_ang_mom[:, 2]


def handkerchief_xy_distance_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    handkerchief_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Exponential reward for XY alignment of handkerchief root to stick-tip."""
    hk: DeformableObject = env.scene[handkerchief_cfg.name]
    tip_pos, _ = _get_stick_tip_state(env, robot_cfg)
    xy_dist_sq = (hk.data.root_pos_w[:, 0] - tip_pos[:, 0]) ** 2 + \
                 (hk.data.root_pos_w[:, 1] - tip_pos[:, 1]) ** 2
    return torch.exp(-xy_dist_sq / (2 * std ** 2))


def handkerchief_z_distance_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    handkerchief_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Exponential reward for Z alignment of handkerchief root to stick-tip."""
    hk: DeformableObject = env.scene[handkerchief_cfg.name]
    tip_pos, _ = _get_stick_tip_state(env, robot_cfg)
    z_dist_sq = (hk.data.root_pos_w[:, 2] - tip_pos[:, 2]) ** 2
    return torch.exp(-z_dist_sq / (2 * std ** 2))


def handkerchief_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.57,
    tolerance: float = 0.20,
    alpha: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Reward for keeping the handkerchief at the target height.

    Returns 1.0 if within tolerance, otherwise -exp(alpha * error).
    """
    hk: DeformableObject = env.scene[asset_cfg.name]
    height_error = torch.abs(hk.data.root_pos_w[:, 2] - target_height)
    return torch.where(
        height_error < tolerance,
        torch.ones_like(height_error),
        -torch.exp(alpha * height_error),
    )


def stick_tip_tangential_speed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for the tangential (circular-motion) speed of the stick tip in XY."""
    robot: Articulation = env.scene[robot_cfg.name]
    end_body_idx = robot.find_bodies("link6")[0][0]
    quat_link_w = robot.data.body_quat_w[:, end_body_idx]

    stick_end_b = torch.tensor([0.0, 0.0, STICK_LENGTH], device=env.device).unsqueeze(0).expand(quat_link_w.shape[0], -1)
    stick_dir = quat_apply(quat_link_w, stick_end_b)
    stick_dir_xy = stick_dir[:, :2]

    _, tip_vel = _get_stick_tip_state(env, robot_cfg)
    tip_vel_xy = tip_vel[:, :2]

    unit_radius = stick_dir_xy / (torch.norm(stick_dir_xy, dim=-1, keepdim=True) + 1e-8)
    unit_tangent = torch.stack([-unit_radius[:, 1], unit_radius[:, 0]], dim=-1)
    tangential_speed = torch.sum(tip_vel_xy * unit_tangent, dim=-1)
    return torch.abs(tangential_speed)



# ==============================================================================
# Terminations
# ==============================================================================

def handkerchief_dropped(
    env: ManagerBasedRLEnv,
    min_height: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Terminate if the lowest handkerchief node falls below `min_height`."""
    hk: DeformableObject = env.scene[asset_cfg.name]
    nodal_z = hk.data.nodal_pos_w[..., 2]           # (num_envs, num_nodes)
    min_z = torch.min(nodal_z, dim=-1)[0]            # (num_envs,)
    return min_z < min_height


# ==============================================================================
# Events (reset)
# ==============================================================================

def reset_handkerchief_to_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.05,
) -> None:
    """Reset handkerchief so that its centre is directly above the stick tip.

    The handkerchief is placed at the stick-tip XY position, with an optional
    ``height_offset`` above the tip so that it can fall and drape naturally
    during the settling phase.

    Args:
        height_offset: Extra height (metres) above the stick tip to spawn the
            handkerchief.  A small positive value (e.g. 0.05) lets the cloth
            fall onto the stick under gravity.
    """
    hk: DeformableObject = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # --- compute current stick-tip position in world frame ---
    stick_tip_pos, _ = _get_stick_tip_state(env, robot_cfg)  # (num_envs, 3)

    # --- read default nodal state and compute its current centre ---
    default_state = hk.data.default_nodal_state_w[env_ids].clone()
    # default_state shape: (len(env_ids), num_nodes, 13)  [pos(3) quat(4) vel(3) ang_vel(3)]
    default_nodal_pos = default_state[..., :3]  # (len(env_ids), num_nodes, 3)
    default_centre = default_nodal_pos.mean(dim=1)  # (len(env_ids), 3)

    # --- compute offset to move centre → stick-tip + height_offset ---
    target_pos = stick_tip_pos[env_ids].clone()
    target_pos[:, 2] += height_offset  # lift slightly above tip
    offset = target_pos - default_centre  # (len(env_ids), 3)

    # apply offset to all nodes
    default_state[..., :3] += offset.unsqueeze(1)
    # zero out velocities (nodes at rest before drop)
    default_state[..., 7:10] = 0.0   # linear velocity
    default_state[..., 10:13] = 0.0  # angular velocity

    hk.write_nodal_state_to_sim(default_state, env_ids=env_ids)
    hk.reset(env_ids)

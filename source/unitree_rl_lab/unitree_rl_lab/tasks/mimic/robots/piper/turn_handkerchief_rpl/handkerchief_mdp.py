"""Handkerchief-specific MDP terms for the turn_handkerchief task.

This module contains observation, reward, termination, and event functions
related to the deformable handkerchief object. Kept separate from the shared
mimic MDP module to avoid mixing task-specific logic with general-purpose code.

Stick-tip position/velocity computation is delegated to :class:`MotionCommand`
(configured via ``stick_tip_body_name`` / ``stick_tip_offset`` in the command
cfg), avoiding duplicated geometric calculations.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, DeformableObject
from isaaclab.envs.mdp.actions import ActionTerm
from isaaclab.managers import ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ==============================================================================
# Custom Action Terms (Residual Policy)
# ==============================================================================

class ResidualJointPositionAction(ActionTerm):
    """
    残差关节位置控制 Action Term。
    公式: a_total = a_base (来自 Phase1 Pt) + k * a_res (来自 Phase2 PPO)
    最终目标关节位置 = a_total * scale + offset
    """
    cfg: "ResidualJointPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "ResidualJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # 1. 解析需要控制的关节索引
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        # 2. 初始化 action 缓冲区 (ActionTerm.reset() 需要这两个)
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # 3. 核心：加载 Phase 1 冻结的基础策略网络
        if not self.cfg.phase1_policy_path:
            raise ValueError("Must provide 'phase1_policy_path' in ResidualJointPositionActionCfg.")

        print(f"[INFO] Loading Base Policy from: {self.cfg.phase1_policy_path}")
        self.base_policy = torch.jit.load(self.cfg.phase1_policy_path).to(self.device)
        self.base_policy.eval()

        # 4. 准备 Action 的缩放与偏移 (与标准的 JointPositionAction 逻辑一致)
        self._scale = torch.full((self._num_joints,), self.cfg.scale, device=self.device)
        self._offset = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        if self.cfg.use_default_offset:
            self._offset[:] = self._asset.data.default_joint_pos[:, self._joint_ids]

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        """返回神经网络输出的原始动作（未经缩放或处理）"""
        return self._raw_actions  # 这里的变量名取决于你在 apply_actions 中存的名字

    @property
    def processed_actions(self) -> torch.Tensor:
        """返回实际应用到仿真环境中的处理后的动作"""
        return self._processed_actions  # 这里的变量名取决于你的计算结果

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # 步骤 A: 提取 Base Policy 所需的纯运动学观测
        if self.cfg.base_obs_group not in self._env.obs_buf:
            raise KeyError(f"Observation group '{self.cfg.base_obs_group}' not found in env.obs_buf. "
                           f"You must define this group in Phase 2 ObservationsCfg to feed the Base Policy.")
        base_obs = self._env.obs_buf[self.cfg.base_obs_group]

        # 步骤 B: 让冻结的 Base Policy 计算打底动作
        with torch.no_grad():
            a_base = self.base_policy(base_obs)
        self._last_a_base = a_base.clone()  # cache for last_base_action obs

        # 步骤 C: 残差叠加公式  a_total = a_base + k * a_res
        total_action_normalized = a_base + self.cfg.residual_scale * self.raw_actions

        # 步骤 D: 转换为真实的关节目标位置
        self._processed_actions[:] = total_action_normalized * self._scale + self._offset

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class ResidualJointPositionActionCfg(ActionTermCfg):
    """Configuration for the residual joint position action term."""

    class_type: type[ActionTerm] = ResidualJointPositionAction

    asset_name: str = "robot"
    joint_names: list[str] = [".*"]
    scale: float = 1.0
    use_default_offset: bool = True

    # --- 残差策略专属参数 ---
    phase1_policy_path: str = ""      # Phase 1 导出的 .pt 模型的绝对路径
    residual_scale: float = 0.1       # 残差动作的缩放因子 k
    base_obs_group: str = "phase1_policy"  # Base Policy 观测组名称


# ==============================================================================
# Observations
# ==============================================================================

def last_base_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the cached base policy output (a_base) from the previous step.

    In Phase 2, ``mdp.last_action`` returns the PPO residual output (a_res),
    which has a completely different distribution from the base policy output
    that Phase 1 was trained with.  This function returns the correct value
    so the frozen base policy sees its own previous output as ``last_action``.
    """
    action_term = env.action_manager.get_term("JointPositionAction")
    if hasattr(action_term, "_last_a_base"):
        return action_term._last_a_base.clone()
    # first step: no previous a_base yet, return zeros
    return torch.zeros(env.num_envs, action_term.action_dim, device=env.device)


# ==============================================================================
# PointNet 观测处理 (专门为手绢点云设计的编码器和包装器)
# ==============================================================================

# 文件: handkerchief_mdp.py (添加到 Observations 区域)

def point_cloud_from_top_camera(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
    command_name: str = "motion",
    num_samples: int = 256,
    simulate_camera_noise: bool = True
) -> torch.Tensor:
    """
    模拟顶视相机的点云提取与预处理 (Sim2Real 极速版)。
    它等价于：顶视相机拍摄 -> 颜色过滤拿到手绢 -> 提取 3D 点云 -> 降采样。
    """
    hk: DeformableObject = env.scene[asset_cfg.name]

    # 1. 获取上帝视角的绝对节点坐标 (相当于完美相机过滤出了手绢)
    nodal_pos = hk.data.nodal_pos_w  # [num_envs, num_nodes, 3]
    num_nodes = nodal_pos.shape[1]

    # 2. 模拟真机处理：带放回的随机采样 256 个点
    # 这抹平了仿真网格和真机深度图在点云密度上的差异
    sample_indices = torch.randint(0, num_nodes, (num_samples,), device=env.device)
    sampled_nodes_w = nodal_pos[:, sample_indices, :]  # [num_envs, 256, 3]

    # 3. 关键：注入顶视相机的特有噪声 (Sim2Real Domain Randomization)
    if simulate_camera_noise:
        # RealSense 等深度相机在 XY 平面（像素平移）误差较小，但在 Z 轴（深度探测）误差较大
        noise = torch.randn_like(sampled_nodes_w)
        noise[..., :2] *= 0.005  # X/Y 轴注入 5mm 的标准差噪声
        noise[..., 2] *= 0.015   # Z 轴 (深度) 注入 1.5cm 的标准差噪声
        sampled_nodes_w += noise

    # 4. 视角转换与 Zero-centering (坐标中心化)
    # 无论你的顶视相机挂在天花板多高的地方，网络都不应该去学习相机的绝对坐标。
    # 我们必须把这 256 个点转换到“以机械臂末端(或手绢质心)为原点的局部坐标系”中。
    tip_pos = _get_command(env, command_name).robot_stick_tip_pos_w.unsqueeze(1)

    # 计算相对形态 (这就完美抹除了相机绝对安装位置的差异)
    relative_pc = sampled_nodes_w - tip_pos  # [num_envs, 256, 3]

    # 5. 展平为 1D 数组，喂给 RSL-RL 和 PointNet
    return relative_pc.view(env.num_envs, -1)  # [num_envs, 768]


# ==============================================================================
# Helper
# ==============================================================================


def _get_command(env: ManagerBasedRLEnv, command_name: str = "motion") -> MotionCommand:
    """Get the MotionCommand term (which provides stick tip pos/vel via config)."""
    return env.command_manager.get_term(command_name)


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


def handkerchief_to_stick_tip_pos(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    handkerchief_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Relative position vector from stick-tip to handkerchief root.  (num_envs, 3)"""
    hk: DeformableObject = env.scene[handkerchief_cfg.name]
    return hk.data.root_pos_w - _get_command(env, command_name).robot_stick_tip_pos_w


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

    Note: weight > 0 encourages CCW (逆时针), weight < 0 encourages CW (顺时针).
    Current reference trajectory is CCW, so use positive weight.
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
    command_name: str = "motion",
    handkerchief_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Exponential reward for XY alignment of handkerchief root to stick-tip."""
    hk: DeformableObject = env.scene[handkerchief_cfg.name]
    tip_pos = _get_command(env, command_name).robot_stick_tip_pos_w
    xy_dist_sq = (hk.data.root_pos_w[:, 0] - tip_pos[:, 0]) ** 2 + \
                 (hk.data.root_pos_w[:, 1] - tip_pos[:, 1]) ** 2
    return torch.exp(-xy_dist_sq / (2 * std ** 2))


def handkerchief_z_distance_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.10,
    command_name: str = "motion",
    handkerchief_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Exponential reward for Z alignment of handkerchief root to stick-tip."""
    hk: DeformableObject = env.scene[handkerchief_cfg.name]
    tip_pos = _get_command(env, command_name).robot_stick_tip_pos_w
    z_dist_sq = (hk.data.root_pos_w[:, 2] - tip_pos[:, 2]) ** 2
    return torch.exp(-z_dist_sq / (2 * std ** 2))


def stick_tip_tangential_speed(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Reward for the tangential (circular-motion) speed of the stick tip in XY."""
    command = _get_command(env, command_name)

    # Get link6 orientation and offset from command config
    link_quat_w = command.robot.data.body_quat_w[:, command.stick_tip_body_idx]
    offset = command.stick_tip_offset_t.unsqueeze(0).expand(link_quat_w.shape[0], -1)

    stick_dir = quat_apply(link_quat_w, offset)
    stick_dir_xy = stick_dir[:, :2]

    tip_vel = command.robot_stick_tip_vel_w
    tip_vel_xy = tip_vel[:, :2]

    unit_radius = stick_dir_xy / (torch.norm(stick_dir_xy, dim=-1, keepdim=True) + 1e-8)
    unit_tangent = torch.stack([-unit_radius[:, 1], unit_radius[:, 0]], dim=-1)
    tangential_speed = torch.sum(tip_vel_xy * unit_tangent, dim=-1)
    return torch.abs(tangential_speed)


def handkerchief_spread_area(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
) -> torch.Tensor:
    """Reward proportional to how spread-out the handkerchief is.

    Computed as the sum of XY variance of all nodal positions relative to root.
    Larger variance = more spread out = higher reward.
    Formula: area_proxy = Var(x_nodes) + Var(y_nodes)
    """
    hk: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = hk.data.nodal_pos_w  # (N, nodes, 3)
    root_pos = hk.data.root_pos_w    # (N, 3)
    rel_xy = nodal_pos[..., :2] - root_pos[:, None, :2]  # (N, nodes, 2)
    var_xy = torch.var(rel_xy, dim=1)  # (N, 2)
    return var_xy.sum(dim=-1)  # (N,)


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

def reset_robot_to_trajectory_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "motion",
) -> None:
    """Reset robot joints to reference trajectory frame 0.

    Ensures the robot starts at the exact trajectory start pose,
    avoiding PD transient oscillation from URDF default → trajectory mismatch.
    """
    robot = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    motion_cmd = _get_command(env, command_name)
    joint_pos = motion_cmd.joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_handkerchief_to_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("handkerchief"),
    command_name: str = "motion",
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
    stick_tip_pos = _get_command(env, command_name).robot_stick_tip_pos_w  # (num_envs, 3)

    # --- read default nodal state and compute its current centre ---
    default_state = hk.data.default_nodal_state_w[env_ids].clone()
    # default_state shape: (len(env_ids), num_nodes, 6)  [pos(3) vel(3)]
    default_nodal_pos = default_state[..., :3]  # (len(env_ids), num_nodes, 3)
    default_centre = default_nodal_pos.mean(dim=1)  # (len(env_ids), 3)

    # --- compute offset to move centre → stick-tip + height_offset ---
    target_pos = stick_tip_pos[env_ids].clone()
    target_pos[:, 2] += height_offset  # lift slightly above tip
    offset = target_pos - default_centre  # (len(env_ids), 3)

    # apply offset to all nodes
    default_state[..., :3] += offset.unsqueeze(1)
    # zero out velocities (nodes at rest before drop)
    default_state[..., 3:6] = 0.0   # linear velocity

    hk.write_nodal_state_to_sim(default_state, env_ids=env_ids)
    hk.reset(env_ids)

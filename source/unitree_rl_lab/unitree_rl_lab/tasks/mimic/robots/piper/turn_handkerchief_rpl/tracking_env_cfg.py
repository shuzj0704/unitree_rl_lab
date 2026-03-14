from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.assets.robots.piper import PIPER_CFG, PIPER_GHOST_CFG, PIPER_MIMIC_ACTION_SCALE
from unitree_rl_lab.assets.deformable import HANDKERCHIEF_CFG

# handkerchief-specific MDP terms (observations, rewards, terminations, events)
from . import handkerchief_mdp as hk_mdp

##
# Scene definition
##


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ghost_robot: ArticulationCfg | None = None
    # deformable objects
    handkerchief: DeformableObjectCfg = HANDKERCHIEF_CFG.replace(prim_path="{ENV_REGEX_NS}/handkerchief")
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=f"{os.path.dirname(__file__)}/2026-03-06-00-47-25_tran.npz",
        anchor_body_name="base_link",
        resampling_time_range=(1.0e9, 1.0e9),   # 重采样时间范围，这里设置为极大值，表示不重采样
        debug_vis=True,     # 是否启用调试可视化
        stick_tip_body_name="link6",              # 末端执行器 body 名称
        stick_tip_offset=(0.0, 0.0, 0.17),        # 棍子末端在 link6 本地坐标系下的偏移
        pose_range={
            "x": (0.0, 0.0),    # 底座完全固定
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        velocity_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        joint_position_range=(-0.05, 0.05),
        body_names=[
            "base_link",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            ],
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = hk_mdp.ResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=1.0,
        use_default_offset=True,
        # TODO: 必须在这里填入你 Phase 1 训练好并导出的 .pt 模型绝对路径！
        phase1_policy_path="/home/shu22/locomotion/unitree_rl_lab/logs/rsl_rl/piper_turn_mimic_v0/2026-03-13_00-41-16/exported/policy.pt",
        residual_scale=0.1,  # k=0.1, 限制 RL 每次最多只能微调 0.1 弧度
        base_obs_group="phase1_policy",  # 指定 Base Policy 去哪里拿属于它的观测
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # -------------------------------------------------------------
    # 专供 Phase 1 冻结模型 (Base Policy) 使用的纯运动学观测
    # 必须与 Phase 1 (turn_mimic) PolicyCfg 的 terms 完全一致！
    # -------------------------------------------------------------
    @configclass
    class Phase1PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # Use base policy's own previous output, NOT the PPO residual (a_res)
        last_action = ObsTerm(func=hk_mdp.last_base_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # -------------------------------------------------------------
    # 给 Phase 2 残差 Actor 使用的带噪观测
    # 观测布局: [proprioception(30) | point_cloud_current(768) | point_cloud_prev(768)]
    # PointNetActorWrapper 根据 num_proprioception 切分
    # -------------------------------------------------------------
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. 运动学基础感知 (proprioception, 共 30 维)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})  # 12
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005))  # 6
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.02, n_max=0.02))    # 6
        last_action = ObsTerm(func=mdp.last_action)                                                # 6

        # 2. 点云视觉输入 (history_length=1 → 输出 [current(768) | previous(768)] = 1536 维)
        point_cloud = ObsTerm(func=hk_mdp.point_cloud_from_top_camera, history_length=1)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # -------------------------------------------------------------
    # 给 Phase 2 Critic 使用的上帝视角特权观测
    # -------------------------------------------------------------
    @configclass
    class PrivilegedCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        action = ObsTerm(func=mdp.last_action)

        # -- 上帝视角的特权信息 (无噪声) --
        stick_tip_pos = ObsTerm(func=mdp.stick_tip_pos_w, params={"command_name": "motion"})
        stick_tip_vel = ObsTerm(func=mdp.stick_tip_vel_w, params={"command_name": "motion"})
        hk_root_pos = ObsTerm(func=hk_mdp.handkerchief_root_pos_w)
        hk_root_vel = ObsTerm(func=hk_mdp.handkerchief_root_vel_w)
        hk_to_tip = ObsTerm(func=hk_mdp.handkerchief_to_stick_tip_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # 实例化三个组
    phase1_policy: Phase1PolicyCfg = Phase1PolicyCfg()
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    # reset robot joints to reference trajectory frame 0
    # (avoid URDF default → trajectory mismatch oscillation)
    reset_robot = EventTerm(
        func=hk_mdp.reset_robot_to_trajectory_start,
        mode="reset",
    )

    # reset handkerchief: place it above the stick tip so it falls naturally
    reset_handkerchief = EventTerm(
        func=hk_mdp.reset_handkerchief_to_default,
        mode="reset",
        params={"height_offset": 0.01},  # 1 cm above stick tip
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- 硬件与正则化保护 --
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-9)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-6)

    # 【新增/极度关键】由于 PPO 现在输出的是残差 a_res，惩罚动作大小就是惩罚微调幅度！
    # 逼迫网络：非必要不微调，尽量靠 Base Policy 跑。
    residual_action_penalty = RewTerm(func=mdp.action_l2, weight=-0.5) 
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"])},
    )

    # -- 轨迹跟踪 (权重必须大幅降低，给救手绢留出动作空间) --
    tracking_joint_pos = RewTerm(
        func=mdp.motion_relative_joint_position_tracking_exp,
        weight=0.5,  # 从 3.0 降到 0.5
        params={"command_name": "motion", "k": 1.0, "std": 0.25},
    )
    tracking_joint_vel = RewTerm(
        func=mdp.motion_relative_joint_velocity_tracking_exp,
        weight=0.05, # 从 0.2 降到 0.05
        params={"command_name": "motion", "k": 0.3, "std": 0.8},
    )

    # -- 手绢状态 (核心目标，权重拉高) --
    hk_spin = RewTerm(func=hk_mdp.handkerchief_spin_angular_momentum, weight=8.0) # 核心动能
    hk_xy_dist = RewTerm(func=hk_mdp.handkerchief_xy_distance_exp, weight=1.0, params={"std": 0.05})
    hk_z_dist = RewTerm(func=hk_mdp.handkerchief_z_distance_exp, weight=1.0, params={"std": 0.10})
    hk_height = RewTerm(
        func=hk_mdp.handkerchief_height_reward, weight=2.0,
        params={"target_height": 0.57, "tolerance": 0.20, "alpha": 2.0},
    )
    stick_tangential_speed = RewTerm(func=hk_mdp.stick_tip_tangential_speed, weight=5.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    hk_dropped = DoneTerm(
        func=hk_mdp.handkerchief_dropped,
        params={"min_height": 0.3},
    )


##
# Environment configuration
##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    # -- settling phase --
    # Time (seconds) the robot holds its initial pose while the cloth falls.
    # During this period actions are overridden, rewards are zeroed, and the
    # reference trajectory is frozen.  Set to 0 to disable.
    settling_time_s: float = 2.0

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 40.0
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_max_soft_body_contacts = 2 ** 22  # reduced for GPU memory; increase if contact buffer overflows


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        # enable ghost robot for reference trajectory visualization
        self.scene.ghost_robot = PIPER_GHOST_CFG.replace(prim_path="{ENV_REGEX_NS}/GhostRobot")

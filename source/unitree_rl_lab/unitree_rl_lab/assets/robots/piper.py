# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`PIPER_CFG`: Piper robot with turn end
* :obj:`PIPER__HIGH_PD_CFG`: Piper robot with turn end with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import sys
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Set the Isaac Sim external assets directory path
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_path, "../../../.."))
isaacsim_external_dir = os.path.join(root_path, "model") 
Piper_path = f"{isaacsim_external_dir}/piper_model/piper_stick/piper_stick.usd"



##
# Configuration
##

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=Piper_path,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.261,
            "joint3": -2.09,
            "joint4": 0.0,
            "joint5": 0.34,
            "joint6": 0.0,
        },
    ),
    actuators={
        "piper_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit_sim=8.0,
            velocity_limit_sim=3.0,
            # stiffness=70.0,
            # damping=2.0,
            stiffness=40.0,
            damping=0.1,
        ),
        "piper_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit_sim=8.0,
            velocity_limit_sim=3.0,
            # stiffness=70.0,
            # damping=2.0,
            stiffness=40.0,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# 动作缩放系数：限制策略输出在安全力矩范围内（最大能力的25%）
# 公式：scale = 0.25 * (力矩限制 / 刚度)
# 作用：防止 PD 控制器输出力矩超限，确保训练稳定性
_shoulder_effort_limit = PIPER_CFG.actuators["piper_shoulder"].effort_limit_sim
_shoulder_stiffness = PIPER_CFG.actuators["piper_shoulder"].stiffness
_forearm_effort_limit = PIPER_CFG.actuators["piper_forearm"].effort_limit_sim
_forearm_stiffness = PIPER_CFG.actuators["piper_forearm"].stiffness

PIPER_MIMIC_ACTION_SCALE = {
    # 肩部关节 (joint1-4)
    "joint1": 0.25 * _shoulder_effort_limit / _shoulder_stiffness,
    "joint2": 0.25 * _shoulder_effort_limit / _shoulder_stiffness,
    "joint3": 0.25 * _shoulder_effort_limit / _shoulder_stiffness,
    "joint4": 0.25 * _shoulder_effort_limit / _shoulder_stiffness,
    
    # 前臂关节 (joint5-6)
    "joint5": 0.25 * _forearm_effort_limit / _forearm_stiffness,
    "joint6": 0.25 * _forearm_effort_limit / _forearm_stiffness,
}


PIPER__HIGH_PD_CFG = PIPER_CFG.copy()
PIPER__HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
PIPER__HIGH_PD_CFG.actuators["piper_shoulder"].stiffness = 400.0
PIPER__HIGH_PD_CFG.actuators["piper_shoulder"].damping = 80.0
PIPER__HIGH_PD_CFG.actuators["piper_forearm"].stiffness = 400.0
PIPER__HIGH_PD_CFG.actuators["piper_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

# Ghost robot for reference trajectory visualization (play mode only)
PIPER_GHOST_CFG = PIPER_CFG.copy()
PIPER_GHOST_CFG.spawn.rigid_props.disable_gravity = True
PIPER_GHOST_CFG.spawn.visible = True
# Use very low stiffness so ghost doesn't fight joint writes
PIPER_GHOST_CFG.actuators["piper_shoulder"].stiffness = 0.0
PIPER_GHOST_CFG.actuators["piper_shoulder"].damping = 0.0
PIPER_GHOST_CFG.actuators["piper_forearm"].stiffness = 0.0
PIPER_GHOST_CFG.actuators["piper_forearm"].damping = 0.0
# Set a distinct color for the ghost robot
PIPER_GHOST_CFG.spawn.visual_material = sim_utils.MdlFileCfg(
    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Colors/Blue.mdl",
    project_uvw=True,
)

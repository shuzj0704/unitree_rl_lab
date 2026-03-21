# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug script: play Phase 2 RPL and record internal signals for analysis.

Records per-frame: phase1_policy obs, a_base, a_res, a_total, last_action obs,
actual joint positions. Saves to CSV for comparison with Phase 1 standalone play.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Debug play RPL agent.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--max_steps", type=int, default=2000, help="Max steps to record.")
parser.add_argument("--output", type=str, default=None, help="Output CSV path.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import importlib
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
import unitree_rl_lab
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# rsl-rl resolves policy class_name via eval() in on_policy_runner module scope.
rsl_on_policy_runner.unitree_rl_lab = unitree_rl_lab


def _resolve_policy_class_for_rsl_eval(agent_cfg: RslRlOnPolicyRunnerCfg):
    class_name = agent_cfg.policy.class_name
    if not isinstance(class_name, str) or "." not in class_name:
        return
    module_path, symbol_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    symbol = getattr(module, symbol_name)
    setattr(rsl_on_policy_runner, symbol_name, symbol)
    agent_cfg.policy.class_name = symbol_name


def main():
    """Play with RSL-RL agent and record debug data."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    _resolve_policy_class_for_rsl_eval(agent_cfg)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # get action term reference
    action_term = env.unwrapped.action_manager.get_term("JointPositionAction")

    # prepare CSV output
    # Default: analysis/rl_play_rpl/<run_name>/rpl_debug.csv
    # run_name is extracted from the checkpoint path (e.g. "2026-03-21_01-09-47")
    output_path = args_cli.output
    if output_path is None:
        run_name = os.path.basename(log_dir.rstrip("/"))
        output_path = os.path.join("analysis", "rl_play_rpl", run_name, "rpl_debug.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # build CSV header
    joint_names = [f"j{i+1}" for i in range(6)]
    header = ["step"]
    # phase1_policy obs (36 dims)
    header += [f"p1_obs_{i}" for i in range(36)]
    # a_base (6)
    header += [f"a_base_{n}" for n in joint_names]
    # a_res (6)
    header += [f"a_res_{n}" for n in joint_names]
    # a_total = processed_actions (6)
    header += [f"a_total_{n}" for n in joint_names]
    # last_action from phase1_policy obs (dims 30-35, the last 6 of 36)
    header += [f"last_act_obs_{n}" for n in joint_names]
    # actual joint pos (6)
    header += [f"actual_jpos_{n}" for n in joint_names]

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    print(f"[INFO] Recording {args_cli.max_steps} steps to {output_path}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for step in range(args_cli.max_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            # extract debug data (all from env 0)
            p1_obs = env.unwrapped.obs_buf.get("phase1_policy", None)
            if p1_obs is not None:
                p1_obs_0 = p1_obs[0].cpu().tolist()
            else:
                p1_obs_0 = [0.0] * 36

            a_base_0 = action_term._last_a_base[0].cpu().tolist() if hasattr(action_term, "_last_a_base") else [0.0] * 6
            a_res_0 = action_term._raw_actions[0].cpu().tolist()
            a_total_0 = action_term._processed_actions[0].cpu().tolist()

            # last_action is the last 6 dims of the 36-dim phase1_policy obs
            last_act_obs_0 = p1_obs_0[30:36]

            # actual joint positions
            robot = env.unwrapped.scene["robot"]
            actual_jpos_0 = robot.data.joint_pos[0].cpu().tolist()[:6]

            row = [step] + p1_obs_0 + a_base_0 + a_res_0 + a_total_0 + last_act_obs_0 + actual_jpos_0
            writer.writerow(row)

            if not simulation_app.is_running():
                break

    print(f"[INFO] Saved debug CSV to {output_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

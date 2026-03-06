# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint once and record policy vs reference joint data to CSV."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play once and record tracking data to CSV.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--output", type=str, default=None, help="Output CSV file path (default: auto-generated).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play once through the reference trajectory and record tracking data."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,  # force single env for recording
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # Get the motion command to access reference trajectory
    unwrapped_env = env.unwrapped
    command = unwrapped_env.command_manager.get_term("motion")
    total_motion_frames = command.motion.time_step_total

    # Get joint names
    joint_names = command.robot.joint_names
    num_joints = len(joint_names)
    print(f"[INFO] Joint names ({num_joints}): {joint_names}")
    print(f"[INFO] Total motion frames: {total_motion_frames}")

    # Get the action term to access processed actions (cmd sent to actuator)
    action_term = unwrapped_env.action_manager.get_term("JointPositionAction")

    # Build CSV header
    # Columns: ref_pos, ref_vel, policy_output, cmd_pos, actual_pos, actual_vel, errors
    header = ["timestep", "motion_frame", "time_s"]
    for jn in joint_names:
        header.append(f"ref_pos_{jn}")
    for jn in joint_names:
        header.append(f"ref_vel_{jn}")
    for jn in joint_names:
        header.append(f"policy_output_{jn}")
    for jn in joint_names:
        header.append(f"cmd_pos_{jn}")
    for jn in joint_names:
        header.append(f"actual_pos_{jn}")
    for jn in joint_names:
        header.append(f"actual_vel_{jn}")
    for jn in joint_names:
        header.append(f"err_pos_{jn}")
    for jn in joint_names:
        header.append(f"err_vel_{jn}")
    header.append("total_pos_err_l2")
    header.append("total_vel_err_l2")

    # Determine output path
    if args_cli.output:
        output_path = args_cli.output
    else:
        output_path = os.path.join(os.path.abspath("analysis"), "rl_play", "play_tracking_record.csv")

    # Data collection
    rows = []

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    # Force trajectory to start from frame 0
    # (adaptive sampling may set a random start position)
    command.time_steps[:] = 0
    print(f"[INFO] Forced time_steps to 0 (was set by adaptive sampling)")

    timestep = 0
    trajectory_completed = False

    print(f"[INFO] Starting play... will stop after one full trajectory pass ({total_motion_frames} frames).")
    print(f"[INFO] Output will be saved to: {output_path}")

    # simulate environment - run for one full trajectory
    while simulation_app.is_running() and not trajectory_completed:
        start_time = time.time()

        # Read current motion frame BEFORE step (this is the frame being tracked)
        motion_frame = command.time_steps[0].item()

        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        # Check if trajectory completed (time_steps was reset by _update_command after reaching end)
        new_motion_frame = command.time_steps[0].item()
        if timestep > 0 and new_motion_frame < motion_frame:
            print(f"[INFO] Trajectory completed at timestep {timestep} "
                  f"(reached frame {motion_frame}, resampled to {new_motion_frame})")
            trajectory_completed = True
            # Don't record the resampled frame
            continue

        # Record data
        ref_jp = command.joint_pos[0].cpu().numpy()          # reference pos from npz
        ref_jv = command.joint_vel[0].cpu().numpy()          # reference vel from npz
        policy_out = actions[0].cpu().numpy()                 # raw RL policy output
        cmd_jp = action_term.processed_actions[0].cpu().numpy()  # cmd sent to actuator (actions * scale + offset)
        cur_jp = command.robot_joint_pos[0].cpu().numpy()     # actual feedback pos
        cur_jv = command.robot_joint_vel[0].cpu().numpy()     # actual feedback vel
        err_jp = abs(ref_jp - cur_jp)
        err_jv = abs(ref_jv - cur_jv)

        row = [timestep, motion_frame, timestep * dt]
        row.extend(ref_jp.tolist())
        row.extend(ref_jv.tolist())
        row.extend(policy_out.tolist())
        row.extend(cmd_jp.tolist())
        row.extend(cur_jp.tolist())
        row.extend(cur_jv.tolist())
        row.extend(err_jp.tolist())
        row.extend(err_jv.tolist())
        row.append(float((err_jp ** 2).sum() ** 0.5))  # L2 pos error
        row.append(float((err_jv ** 2).sum() ** 0.5))  # L2 vel error
        rows.append(row)

        timestep += 1

        # Print progress
        if timestep % 200 == 0:
            progress = motion_frame / max(total_motion_frames, 1) * 100
            print(f"  step={timestep}, motion_frame={motion_frame}/{total_motion_frames} ({progress:.1f}%), "
                  f"pos_err_L2={row[-2]:.4f}, vel_err_L2={row[-1]:.4f}")

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"[DONE] Recorded {len(rows)} steps to: {output_path}")
    print(f"  Joints: {joint_names}")
    print(f"  Columns: {len(header)}")

    # Print summary statistics
    if rows:
        import numpy as np
        data = np.array(rows)
        pos_err_col = -2  # total_pos_err_l2
        vel_err_col = -1  # total_vel_err_l2
        print(f"\n  Position Error L2:  mean={data[:, pos_err_col].mean():.4f}, "
              f"max={data[:, pos_err_col].max():.4f}, std={data[:, pos_err_col].std():.4f}")
        print(f"  Velocity Error L2:  mean={data[:, vel_err_col].mean():.4f}, "
              f"max={data[:, vel_err_col].max():.4f}, std={data[:, vel_err_col].std():.4f}")

        # Per-joint position error summary
        print(f"\n  Per-joint mean position error (ref vs actual):")
        err_pos_start = 3 + 6 * num_joints  # after ref_pos, ref_vel, policy_out, cmd_pos, actual_pos, actual_vel
        for j, jn in enumerate(joint_names):
            col = err_pos_start + j
            print(f"    {jn:<12}: {data[:, col].mean():.4f} (max={data[:, col].max():.4f})")
    print(f"{'='*60}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

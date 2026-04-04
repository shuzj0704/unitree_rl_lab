"""Custom ManagerBasedRLEnv subclass for the turn-handkerchief mimic task.

Adds a "settling phase" at the start of each episode: the robot is **physically
frozen** (joint state written directly every sub-step, bypassing PD control) while
the deformable handkerchief falls and drapes over the stick under gravity.

During settling:
* The robot's joint positions are **teleported** every physics sub-step — no PD
  transient, no gravity sag, completely static like pressing pause.
* The MotionCommand time counter is **frozen** so the reference trajectory does not advance.
* All **rewards are zeroed** so the settling period does not affect learning.
* The **episode_length_buf is not incremented** (settling time is "free").
* **Terminations are suppressed** (the cloth is still falling, don't kill the episode).
"""

from __future__ import annotations

import math
import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand


class TurnHandkerchiefMimicEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with a cloth-settling warm-up phase."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # --- settling configuration ---
        # settling_time_s is stored on the cfg (set in tracking_env_cfg.py)
        self.settling_time_s: float = getattr(cfg, "settling_time_s", 2.0)
        self.settling_steps: int = math.ceil(self.settling_time_s / self.step_dt)

        # per-env counter: how many steps of settling remain
        self._settling_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # cache the initial (default) joint positions for holding during settling
        robot = self.scene["robot"]
        self._hold_joint_pos = robot.data.default_joint_pos.clone()  # (num_envs, num_joints)
        self._hold_joint_vel = torch.zeros_like(self._hold_joint_pos)

        print(f"[TurnHandkerchiefMimicEnv] settling_time_s={self.settling_time_s}, "
              f"settling_steps={self.settling_steps}, step_dt={self.step_dt}")

    # ------------------------------------------------------------------
    # Override step
    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        # ---- determine which envs are still settling ----
        is_settling = self._settling_remaining > 0  # (num_envs,) bool

        # ---- 1. process actions (but we will override for settling envs) ----
        self.action_manager.process_action(action.to(self.device))

        # Override processed actions for settling envs → hold default pose
        if torch.any(is_settling):
            for term_name in self.action_manager.active_terms:
                term = self.action_manager.get_term(term_name)
                # JointPositionAction stores processed_actions as target joint pos
                term._processed_actions[is_settling] = self._hold_joint_pos[is_settling]

        # ---- 2. physics stepping (decimation sub-steps) ----
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # pre-compute settling env ids (avoid repeated nonzero inside loop)
        settling_ids = is_settling.nonzero(as_tuple=False).squeeze(-1) if torch.any(is_settling) else None
        robot = self.scene["robot"]

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()

            # Settling: teleport joint state every sub-step, bypassing PD controller.
            # Robot is completely frozen — no gravity sag, no PD transient.
            if settling_ids is not None and len(settling_ids) > 0:
                robot.write_joint_state_to_sim(
                    self._hold_joint_pos[settling_ids],
                    self._hold_joint_vel[settling_ids],
                    env_ids=settling_ids,
                )

            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # ---- 3. post-step bookkeeping ----
        # only increment episode length for non-settling envs
        not_settling = ~is_settling
        self.episode_length_buf[not_settling] += 1
        self.common_step_counter += 1

        # decrement settling counter
        self._settling_remaining[is_settling] -= 1

        # ---- 4. termination check ----
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # suppress terminations during settling (cloth might trigger "dropped")
        self.reset_buf[is_settling] = 0
        self.reset_terminated[is_settling] = False
        self.reset_time_outs[is_settling] = False

        # ---- 5. reward computation ----
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # zero rewards during settling
        self.reward_buf[is_settling] = 0.0

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # ---- 6. reset terminated envs ----
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # ---- 7. update commands ----
        self.command_manager.compute(dt=self.step_dt)
        # freeze motion command time for settling envs (undo the +1 that _update_command did)
        still_settling = self._settling_remaining > 0
        if torch.any(still_settling):
            motion_cmd: MotionCommand = self.command_manager.get_term("motion")
            motion_cmd.time_steps[still_settling] = torch.clamp(
                motion_cmd.time_steps[still_settling] - 1, min=0
            )

        # ---- 8. interval events ----
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # ---- 9. observations ----
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # ------------------------------------------------------------------
    # Override _reset_idx to start settling for newly-reset envs
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # start the settling countdown for these envs
        self._settling_remaining[env_ids] = self.settling_steps
        # Hold the trajectory-frame joint positions that _resample_command just wrote,
        # NOT the URDF default (robot.data.default_joint_pos).
        motion_cmd: MotionCommand = self.command_manager.get_term("motion")
        self._hold_joint_pos[env_ids] = motion_cmd.joint_pos[env_ids].clone()

        # Also set PD target to hold position, so when settling ends the PD
        # controller starts with zero error (smooth transition to active control)
        robot = self.scene["robot"]
        robot.set_joint_position_target(
            self._hold_joint_pos[env_ids], env_ids=env_ids
        )

import gymnasium as gym

gym.register(
    id="Piper-Turn-Handkerchief-Mimic-v0",
    entry_point=f"{__name__}.turn_handkerchief_mimic_env:TurnHandkerchiefMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

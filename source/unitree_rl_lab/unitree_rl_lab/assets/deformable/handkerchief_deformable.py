import os

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg

# 获取handkerchief USD文件路径
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_path, "../../../.."))
HANDKERCHIEF_USD_PATH = os.path.join(root_path, "model", "piper_model", "handkerchief", "new_handkerchief.usd")

HANDKERCHIEF_CFG = DeformableObjectCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=HANDKERCHIEF_USD_PATH),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.016, -0.34, 0.57)),
    debug_vis=False,
)

# Piper 转手绢 — 两阶段训练技术路线

## 概述

采用 **非对称视觉残差策略架构 (Asymmetric Visual Residual Policy)**，将"基础周期性运动"与"基于视觉的动态微调"解耦为两个阶段：

```
Phase 1 (turn_mimic)  →  冻结导出 .pt  →  Phase 2 (turn_handkerchief_rpl)
      盲跑画圆                                 视觉残差微调
```

最终控制方程：**a_total = a_base + 0.1 * a_residual**

---

## Phase 1: 运动学轨迹模仿 (Piper-Turn-Mimic-v0)

### 目标
无手绢环境下，让机械臂完美跟踪预录的画圆参考轨迹。

### 关键文件
```
tasks/mimic/robots/piper/turn_mimic/tracking_env_cfg.py   # 环境配置
tasks/mimic/agents/rsl_rl_ppo_pointnet_cfg.py              # BasePPORunnerCfg
```

### 场景
- 仅 Piper 机械臂，无柔性体
- `num_envs = 4096`，`replicate_physics = True`（刚体可复制）

### 动作空间
- `JointPositionActionCfg`：直接输出 6 个关节目标位置
- `scale = 1.0`，`use_default_offset = True`

### 观测 (36 维)
| 观测项 | 维度 | 说明 |
|--------|------|------|
| motion_command | 12 | 参考轨迹 joint_pos(6) + joint_vel(6) |
| motion_anchor_ori_b | 6 | 基座朝向（占位用） |
| joint_pos_rel | 6 | 当前关节位置（相对默认值） |
| joint_vel_rel | 6 | 当前关节速度（相对默认值） |
| last_action | 6 | 上一步输出的动作 |

**注意：** 末端轨迹 ref_stick_tip_pos/vel 在观测中是注释掉的，不参与训练。

### 奖励
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| tracking_joint_pos | 3.0 | 主目标：关节位置跟踪 |
| tracking_joint_vel | 0.2 | 关节速度跟踪 |
| tracking_stick_tip_pos | 0.5 | 末端位置跟踪（辅助） |
| tracking_stick_tip_vel | 0.2 | 末端速度跟踪（辅助） |
| action_rate_l2 | -0.2 | 动作平滑 |
| joint_limit | -3.0 | 关节限位惩罚 |
| joint_acc | -1e-9 | 加速度正则 |
| joint_torque | -1e-6 | 力矩正则 |

### PPO 超参 (BasePPORunnerCfg)
- Actor/Critic: `[512, 256, 128]`
- `lr = 1e-4`，`entropy = 0.005`，`max_iterations = 30000`

### 仿真参数
- `dt = 0.01`，`decimation = 2`（控制频率 50 Hz）
- `episode_length = 40s`

### 训练命令
```bash
python scripts/rsl_rl/train.py --task=Piper-Turn-Mimic-v0
```

### 导出
训练完成后导出 TorchScript 模型，路径类似：
```
logs/rsl_rl/piper_turn_mimic_v0/<timestamp>/exported/policy.pt
```

**当前使用的 Phase 1 模型：** `2026-03-08_03-00-41`
- 观测无末端轨迹（36 维）
- 奖励包含末端轨迹追踪

---

## Phase 2: 视觉残差策略 (Piper-Turn-Handkerchief-RPL-v0)

### 目标
引入柔性手绢，通过点云视觉输入训练残差策略，在 base policy 基础上做微调以防止手绢掉落。

### 关键文件
```
tasks/mimic/robots/piper/turn_handkerchief_rpl/
├── __init__.py                    # 任务注册
├── tracking_env_cfg.py            # 环境配置
├── handkerchief_mdp.py            # 手绢观测/奖励/动作/事件
├── pointnet_policy.py             # PointNet + Actor Wrapper
└── turn_handkerchief_mimic_env.py # 自定义环境（settling phase）
tasks/mimic/agents/rsl_rl_ppo_pointnet_cfg.py  # PointNetPPORunnerCfg
```

### 场景
- Piper 机械臂 + DeformableObject 柔性手绢
- `num_envs = 4`，`replicate_physics = False`（柔性体不可复制）
- `gpu_max_soft_body_contacts = 2^22`（防溢出）

### 残差动作 (ResidualJointPositionAction)

```
a_total = a_base(frozen_phase1_policy) + 0.1 * a_residual(PPO_output)
target_joint_pos = a_total * scale + offset
```

- Phase 1 模型通过 `torch.jit.load()` 加载并永久冻结 (`torch.no_grad()`)
- `residual_scale = 0.1`：无论网络输出什么，每步微调上限 0.1 弧度（安全约束）
- Base policy 从 `phase1_policy` 观测组取数据

### 三组观测

#### 1. phase1_policy (36 维) — 喂给冻结的 Base Policy
与 Phase 1 训练时完全一致，无噪声：

| 观测项 | 维度 |
|--------|------|
| motion_command | 12 |
| motion_anchor_ori_b | 6 |
| joint_pos_rel | 6 |
| joint_vel_rel | 6 |
| last_action | 6 |

**关键约束：此组必须与 Phase 1 训练配置完全一致，否则冻结模型维度不匹配会 crash。**

#### 2. policy (798 维) — 喂给残差 Actor (带 PointNet)
| 观测项 | 维度 | 说明 |
|--------|------|------|
| motion_command | 12 | 参考轨迹指令 |
| joint_pos_rel | 6 | 带噪声 (±0.005) |
| joint_vel_rel | 6 | 带噪声 (±0.02) |
| last_action | 6 | 上一步动作 |
| point_cloud | 768 | 256 点 × 3 坐标，相对 stick tip 中心化 |
| **合计** | **798** | proprioception(30) + point_cloud(768) |

**点云生成 (`point_cloud_from_top_camera`)：**
1. 获取手绢全部节点坐标（上帝视角）
2. 随机采样 256 个点（带放回）
3. 注入相机噪声（XY ±5mm, Z ±15mm）
4. 坐标中心化到 stick tip（抹除相机绝对位置）
5. 展平为 768 维向量

#### 3. critic (45 维) — 上帝视角特权观测
| 观测项 | 维度 | 说明 |
|--------|------|------|
| motion_command | 12 | 参考轨迹 |
| joint_pos | 6 | 无噪声关节位置 |
| joint_vel | 6 | 无噪声关节速度 |
| action | 6 | 上一步动作 |
| stick_tip_pos | 3 | 棍子末端位置 |
| stick_tip_vel | 3 | 棍子末端速度 |
| hk_root_pos | 3 | 手绢质心位置 |
| hk_root_vel | 3 | 手绢质心速度 |
| hk_to_tip | 3 | 手绢→棍子末端相对位置 |

### PointNet 网络架构

```
观测 [B, 798]
  │
  ├─ proprioception [B, 30] ──────────────────────┐
  │                                                │
  └─ point_cloud [B, 768] → reshape [B, 256, 3]   │
       │                                           │
       ├─ 当前帧 P_t ──→ PointNet → Z_t [B, 64]   │
       └─ 上一帧 P_{t-1} → PointNet → Z_{t-1} [B, 64]
                              │                    │
                         cat [B, 128]              │
                              │                    │
                         cat ←─────────────────────┘
                              │
                         [B, 158]
                              │
                     Actor MLP [256, 128, 64]
                              │
                     a_residual [B, 6]
```

**LightweightPointNetEncoder：**
- Point-wise MLP: 3 → 64 → 128 → 256 (ELU)
- Global Max Pooling: [B, N, 256] → [B, 256]
- Global MLP: 256 → 128 → 64

**PointNetActorWrapper：**
- 从拼接观测中按 `num_proprioception=30` 切分
- 两帧点云分别过同一个 PointNet，拼接得 128 维时空特征
- [proprioception(30) + Z_temporal(128)] → Actor MLP → 6 维残差动作

**PointNetActorCritic：**
- 继承 rsl_rl 的 `ActorCritic`
- 从总观测维度和 `num_proprioception` 自动推算点云维度
- 替换默认 Actor 为 `PointNetActorWrapper`
- Critic 保持标准 MLP（直接吃 45 维特权观测）

### 奖励
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| **hk_spin** | **8.0** | 核心：手绢 z 轴角动量 |
| **stick_tangential_speed** | **5.0** | 末端切向速度 |
| **hk_height** | **2.0** | 手绢保持目标高度 (0.57m) |
| hk_xy_dist | 1.0 | 手绢 XY 对齐 stick tip |
| hk_z_dist | 1.0 | 手绢 Z 对齐 stick tip |
| tracking_joint_pos | 0.5 | 关节跟踪（大幅降权） |
| tracking_joint_vel | 0.05 | 关节速度跟踪（大幅降权） |
| residual_action_penalty | -0.5 | 惩罚残差幅度（鼓励依赖 base policy） |
| action_rate_l2 | -0.1 | 动作平滑 |
| joint_limit | -10.0 | 关节限位 |
| joint_acc | -1e-9 | 加速度正则 |
| joint_torque | -1e-6 | 力矩正则 |

### Settling Phase (静置初始化)

自定义环境 `TurnHandkerchiefMimicEnv` 重写了 `step()` 方法：

每个 Episode 前 **2 秒** (100 步)：
- 机械臂锁定在初始姿态（参考轨迹第一帧）
- 手绢在重力下自然下垂包裹棍子
- 动作被覆写（忽略 policy 输出）
- 奖励清零（不影响学习）
- episode_length_buf 不递增（settling 时间"免费"）
- 终止条件被抑制（手绢还在下落）
- 参考轨迹时间冻结

### PPO 超参 (PointNetPPORunnerCfg)
- Actor: `[256, 128, 64]`（轻量，只算残差）
- Critic: `[512, 256, 128]`（处理特权信息，保持大容量）
- `lr = 1e-3`，`entropy = 0.01`（视觉任务需要更多探索）
- `max_iterations = 5000`
- `num_proprioception = 30`，`num_points = 256`

### 训练命令
```bash
python scripts/rsl_rl/train.py --task=Piper-Turn-Handkerchief-RPL-v0 --num_envs=16
```

---

## 两阶段关键约束

1. **Phase1PolicyCfg 必须与 Phase 1 训练配置完全一致**
   - 观测项、顺序、维度不能有任何差异
   - 冻结模型的第一层 Linear 维度是训练时就定死的

2. **phase1_policy_path 必须指向正确的模型**
   - 当前：`logs/rsl_rl/piper_turn_mimic_v0/2026-03-08_03-00-41/exported/policy.pt`
   - 此模型训练时观测为 36 维（无末端轨迹观测），奖励含末端轨迹追踪

3. **residual_scale = 0.1 是安全隔离**
   - 无论视觉网络在真机上受到什么干扰，动作突变被卡在 ±0.1 弧度以内

4. **replicate_physics = False**
   - 柔性体仿真不支持物理复制，限制了并行环境数量

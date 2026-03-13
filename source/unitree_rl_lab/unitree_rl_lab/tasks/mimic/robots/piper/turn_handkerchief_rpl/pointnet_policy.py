import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import MLP


# ============================================================================
# 4.2.3 PointNet Network Architecture (单帧空间编码器)
# ============================================================================
class LightweightPointNetEncoder(nn.Module):
    """
    轻量级 PointNet 空间特征提取器。
    严格对应方案文档 4.2.3 章节描述的网络结构。
    """
    def __init__(self, point_dim=3, latent_dim=64):
        super(LightweightPointNetEncoder, self).__init__()

        # Step 1: Point-wise MLP (独立升维)
        # 对每一个点独立进行非线性映射，使用 ELU 保证 RL 梯度平滑
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU()
        )

        # Step 3: Global MLP (降维输出)
        # 处理池化后的全局骨架特征，浓缩为 64 维状态
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim)
            # 输出层通常不加激活函数，让网络自由输出任意范围的连续特征
        )

    def forward(self, point_cloud):
        """
        输入: point_cloud 形状为 [B, N, 3] (例如 [Batch_Size, 256, 3])
        输出: latent_feature 形状为 [B, 64]
        """
        # --------------------------------------------------------------------
        # Step 0: Zero-centering (坐标中心化)
        # 减去当前帧点云的几何中心，强制零均值，增强 Sim-to-Real 鲁棒性
        # --------------------------------------------------------------------
        centroid = torch.mean(point_cloud, dim=1, keepdim=True)  # [B, 1, 3]
        point_cloud = point_cloud - centroid                     # [B, N, 3]

        # --------------------------------------------------------------------
        # Step 1: Point-wise MLP
        # --------------------------------------------------------------------
        point_features = self.point_mlp(point_cloud)             # [B, N, 256]

        # --------------------------------------------------------------------
        # Step 2: Global Max Pooling (全局最大池化)
        # --------------------------------------------------------------------
        global_feature, _ = torch.max(point_features, dim=1)     # [B, 256]

        # --------------------------------------------------------------------
        # Step 3: Global MLP -> 输出 Z_t
        # --------------------------------------------------------------------
        latent_feature = self.global_mlp(global_feature)         # [B, 64]

        return latent_feature


# ============================================================================
# 4.2.4 隐空间时空融合 (Latent Temporal Stacking Wrapper)
# ============================================================================
class PointNetActorWrapper(nn.Module):
    """
    将 PointNet 编码器与 Actor MLP 组合的包装器。
    负责从 PPO 传来的一维观测数组中切分出当前帧和上一帧的点云，
    完成时空特征拼接后，送入最终的 Actor 网络。
    """
    def __init__(self, actor_mlp, num_proprioception, num_points=256, num_pc_frames=2):
        super(PointNetActorWrapper, self).__init__()

        # 实例化我们刚才定义的 PointNet 组件
        self.pointnet = LightweightPointNetEncoder(point_dim=3, latent_dim=64)

        # 原生的 Actor MLP (策略网络的主干)
        self.actor_mlp = actor_mlp

        self.num_prop = num_proprioception
        self.num_points = num_points
        self.num_pc_frames = num_pc_frames
        self.pc_flat_dim = num_points * 3  # 单帧点云展平后的维度: 256 * 3 = 768

    def forward(self, obs):
        """
        obs 形状: [Batch, num_prop + 768 (当前帧) + 768 (上一帧)]
        """
        # 1. 切片提取数据
        prop_obs = obs[:, :self.num_prop]  # 机械臂本体感知数据 [B, num_prop]
        pc_t_flat = obs[:, self.num_prop : self.num_prop + self.pc_flat_dim]

        if self.num_pc_frames >= 2 and obs.shape[1] >= self.num_prop + 2 * self.pc_flat_dim:
            pc_t_minus_1_flat = obs[:, self.num_prop + self.pc_flat_dim : self.num_prop + 2 * self.pc_flat_dim]
        else:
            # If only one frame is available, reuse current frame as previous frame.
            pc_t_minus_1_flat = pc_t_flat

        # 2. 重新塑形为 3D 张量
        pc_t = pc_t_flat.view(-1, self.num_points, 3)                   # 当前帧 P_t
        pc_t_minus_1 = pc_t_minus_1_flat.view(-1, self.num_points, 3)   # 上一帧 P_{t-1}

        # 3. 通过 PointNet 提取空间隐特征
        z_t = self.pointnet(pc_t)                   # [B, 64]
        z_t_minus_1 = self.pointnet(pc_t_minus_1)   # [B, 64]

        # --------------------------------------------------------------------
        # 4.2.4: 特征拼接 (Concatenation) -> Z_temporal
        # --------------------------------------------------------------------
        z_temporal = torch.cat([z_t, z_t_minus_1], dim=1)         # [B, 128]

        # --------------------------------------------------------------------
        # 4.2.4: 最终融合输出
        # 将 Z_temporal 与机械臂本体感知数据拼接，共同构成 Actor 完整观测空间
        # --------------------------------------------------------------------
        final_mlp_input = torch.cat([prop_obs, z_temporal], dim=1)  # [B, num_prop + 128]

        # 送入 Actor 的主干网络，输出动作
        return self.actor_mlp(final_mlp_input)


# ============================================================================
# 4.2.5 将 PointNet Wrapper 集成到 RSL-RL 的 ActorCritic 中
# ============================================================================
class PointNetActorCritic(ActorCritic):
    """
    自定义的 PPO 策略类，继承自 RSL-RL 的 ActorCritic。
    在这里我们将原生的 Actor 替换为带有 PointNet 的 Wrapper。
    """
    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        num_points=256,
        num_pc_frames=None,
        num_proprioception=None,
        **kwargs,
    ):

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        # 从拼接后的 policy obs 总维度推算点云维度
        # rsl_rl 在 concatenate_terms=True 时，obs 以 group 名为 key
        policy_keys = obs_groups.get("policy", ["policy"])
        total_policy_dim = sum(obs[k].shape[-1] for k in policy_keys)

        if num_proprioception is None:
            raise ValueError(
                "num_proprioception must be specified explicitly in the policy kwargs "
                "(e.g. kwargs={'num_proprioception': 30}) when using concatenated observations."
            )
        num_proprioception = int(num_proprioception)
        pc_dim = total_policy_dim - num_proprioception

        single_frame_dim = int(num_points) * 3
        if num_pc_frames is None:
            if pc_dim == single_frame_dim:
                num_pc_frames = 1
            elif pc_dim == 2 * single_frame_dim:
                num_pc_frames = 2
            else:
                raise ValueError(
                    f"Unsupported point_cloud dim={pc_dim}. Expected {single_frame_dim} (1 frame) "
                    f"or {2 * single_frame_dim} (2 frames)."
                )
        elif pc_dim != int(num_pc_frames) * single_frame_dim:
            raise ValueError(
                f"point_cloud dim mismatch: got {pc_dim}, expected {int(num_pc_frames) * single_frame_dim} "
                f"for num_pc_frames={num_pc_frames}, num_points={num_points}."
            )

        if num_proprioception <= 0:
            raise ValueError(
                f"Invalid num_proprioception={num_proprioception}. Check policy observation layout and point cloud dims."
            )

        self.num_proprioception = int(num_proprioception)
        self.num_pc_frames = int(num_pc_frames)

        if self.num_pc_frames not in (1, 2):
            raise ValueError(f"num_pc_frames must be 1 or 2, got {self.num_pc_frames}.")

        actor_mlp = MLP(self.num_proprioception + 128, num_actions, actor_hidden_dims, activation)

        self.actor = PointNetActorWrapper(
            actor_mlp=actor_mlp,
            num_proprioception=self.num_proprioception,
            num_points=num_points,
            num_pc_frames=self.num_pc_frames,
        )

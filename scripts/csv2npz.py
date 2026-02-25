# how to use:
# python csv2npz.py

import pandas as pd
import numpy as np
import os

# set file names
csv_name = "trajectory_20260204"  # <--- 修改这里
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, "..", "data")
csv_path = os.path.join(data_path, csv_name+"_proc.csv")
npz_path = os.path.join(data_path, csv_name+"_proc.npz")
df = pd.read_csv(csv_path)

# === 1. 提取关节位置和速度 (6维) ===
joint_pos_cols = ["q1_pos", "q2_pos", "q3_pos", "q4_pos", "q5_pos", "q6_pos"]
joint_vel_cols = ["q1_vel", "q2_vel", "q3_vel", "q4_vel", "q5_vel", "q6_vel"]

joint_pos = df[joint_pos_cols].values.astype(np.float32)  # Shape: [T, 6]
joint_vel = df[joint_vel_cols].values.astype(np.float32)  # Shape: [T, 6]

# === 2. 提取末端执行器位置和速度 (3维) ===
end_pos_w = df[["pos_x", "pos_y", "pos_z"]].values.astype(np.float32)  # Shape: [T, 3]
end_vel_w = df[["vel_x", "vel_y", "vel_z"]].values.astype(np.float32)  # Shape: [T, 3]

# === 3. 计算 FPS (从 timestamp 推算) ===
timestamps = df["timestamp"].values
dt_mean = np.diff(timestamps).mean()
fps = 1.0 / dt_mean if dt_mean > 0 else 60.0

# === 3.5. 基座位置和旋转（固定值） ===
T = len(df)
num_bodies = 7  # base_link + link1~6
time_step_total = T  # 总时间步数
body_pos_w = np.zeros((T, num_bodies, 3))  # 所有帧都是 [0, 0, 0]
body_quat_w = np.tile([1, 0, 0, 0], (T, num_bodies, 1))  # 所有帧都是单位四元数
body_lin_vel_w = np.zeros((T, num_bodies, 3))
body_ang_vel_w = np.zeros((T, num_bodies, 3))


# === 4. 保存为 .npz ===
np.savez_compressed(
    npz_path,
    fps=fps,                      # float - 采样频率
    joint_pos=joint_pos,          # [T, 6] - 关节位置
    joint_vel=joint_vel,          # [T, 6] - 关节速度
    end_pos_w=end_pos_w,          # [T, 3] - 末端执行器位置
    end_vel_w=end_vel_w,          # [T, 3] - 末端执行器速度
    time_step_total=time_step_total,  # int - 总时间步数
    body_pos_w=body_pos_w,        # [T, 7, 3] - 基座位置（所有帧）
    body_quat_w=body_quat_w,      # [T, 7, 4] - 基座旋转（所有帧）
    body_lin_vel_w=body_lin_vel_w,# [T, 7, 3] - 基座线速度（所有帧）
    body_ang_vel_w=body_ang_vel_w # [T, 7, 3] - 基座角速度（所有帧）
)

# === 5. 打印转换信息 ===
num_frames = len(df)
print("✅ 转换完成！")
print(f"  FPS: {fps:.2f} Hz")
print(f"  总帧数: {num_frames}")
print(f"  时长: {num_frames / fps:.2f} 秒")
print("\n数据结构:")
print(f"  joint_pos:  {joint_pos.shape}  # [T, 6] - 关节位置")
print(f"  joint_vel:  {joint_vel.shape}  # [T, 6] - 关节速度")
print(f"  end_pos_w:  {end_pos_w.shape}  # [T, 3] - 末端位置")
print(f"  end_vel_w:  {end_vel_w.shape}  # [T, 3] - 末端速度")

# === 6. 验证数据质量 ===
print("\n数据质量检查:")
print(f"  joint_pos 范围: [{joint_pos.min():.3f}, {joint_pos.max():.3f}]")
print(f"  joint_vel 范围: [{joint_vel.min():.3f}, {joint_vel.max():.3f}]")
print(f"  end_pos_w 范围: [{end_pos_w.min():.3f}, {end_pos_w.max():.3f}]")
print(f"  end_vel_w 范围: [{end_vel_w.min():.3f}, {end_vel_w.max():.3f}]")
print(f"  NaN检查: joint_pos={np.isnan(joint_pos).any()}, end_pos_w={np.isnan(end_pos_w).any()}")
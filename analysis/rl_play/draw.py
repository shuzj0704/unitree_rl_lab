# *** 说明 ***
# 这个脚本用于分析 play 模式下记录的 CSV 文件，生成每个关节的对比图和速度图。
# 生成的图表将保存在与 CSV 文件相同的目录下，每个关节一个子文件夹，包含 5 张图：
# 1. Ref Pos vs Policy Output
# 2. Policy Output vs Cmd Pos
# 3. Cmd Pos vs Actual Pos
# 4. Ref Pos vs Actual Pos
# 5. Policy Output Velocity

import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 配置区 =====================
# 修改这个变量即可切换不同实验
EXPERIMENT = "2026-03-06_02-18-58-test01"
# ==================================================

# 自动计算路径（基于脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, EXPERIMENT)
CSV_FILE = os.path.join(DATA_DIR, "play_tracking_record.csv")

# 5 组对比图定义: (标题, 列前缀A, 列前缀B, 标签A, 标签B, 文件名后缀)
COMPARE_PAIRS = [
    ("Ref Pos vs Policy Output", "ref_pos", "policy_output", "Ref Pos", "Policy Output", "1_ref_vs_policy"),
    ("Policy Output vs Cmd Pos", "policy_output", "cmd_pos", "Policy Output", "Cmd Pos", "2_policy_vs_cmd"),
    ("Cmd Pos vs Actual Pos",    "cmd_pos",        "actual_pos",  "Cmd Pos",       "Actual Pos",    "3_cmd_vs_actual"),
    ("Ref Pos vs Actual Pos",    "ref_pos",        "actual_pos",  "Ref Pos",       "Actual Pos",    "4_ref_vs_actual"),
    ("Policy Output Velocity", "policy_output", None, "Policy Output", "Velocity", "5_policy_velocity"),
]


def get_joint_names(df):
    """从 CSV 列名中提取关节名列表"""
    prefix = "ref_pos_"
    return [col[len(prefix):] for col in df.columns if col.startswith(prefix)]


def plot_comparison(df, joint, col_a, col_b, label_a, label_b, title, save_path):
    """画一张对比图：两条曲线 + 差值"""
    time_s = df["time_s"]
    a = df[col_a]
    b = df[col_b]
    diff = (a - b).abs()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # 上图：两条曲线叠加
    ax1.plot(time_s, a, label=label_a, color="#1f77b4", linewidth=1)
    ax1.plot(time_s, b, label=label_b, color="#ff7f0e", linewidth=1)
    ax1.set_title(f"{title}  —  {joint}", fontsize=12)
    ax1.set_ylabel("Position (rad)", fontsize=10)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 下图：绝对误差
    ax2.fill_between(time_s, 0, diff, color="#d62728", alpha=0.35)
    ax2.plot(time_s, diff, color="#d62728", linewidth=0.8)
    ax2.set_ylabel("|Error| (rad)", fontsize=10)
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_velocity(df, joint, col, label, title, save_path):
    """画一张速度图：关节角速度"""
    time_s = df["time_s"]
    values = df[col]
    velocity = values.diff() / time_s.diff()

    fig, ax = plt.subplots(figsize=(20, 4))

    # 速度图
    ax.plot(time_s, velocity, label=f"{label} Velocity", color="#2ca02c", linewidth=1)
    ax.set_title(f"{title}  —  {joint}", fontsize=12)
    ax.set_ylabel("Velocity (rad/s)", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.exists(CSV_FILE):
        print(f"错误：找不到文件 '{CSV_FILE}'，请检查 EXPERIMENT 变量。")
        return

    print(f"读取 CSV：{CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    joint_names = get_joint_names(df)
    print(f"关节列表 ({len(joint_names)}): {joint_names}")

    for joint in joint_names:
        # 每个关节一个子文件夹
        joint_dir = os.path.join(DATA_DIR, joint)
        os.makedirs(joint_dir, exist_ok=True)

        for title, prefix_a, prefix_b, label_a, label_b, suffix in COMPARE_PAIRS:
            if suffix == "5_policy_velocity":
                col = f"{prefix_a}_{joint}"
                if col not in df.columns:
                    print(f"  跳过 {joint}/{suffix}：缺少列 {col}")
                    continue

                save_path = os.path.join(joint_dir, f"{suffix}.png")
                plot_velocity(df, joint, col, label_a, title, save_path)
            else:
                col_a = f"{prefix_a}_{joint}"
                col_b = f"{prefix_b}_{joint}" if prefix_b else None

                if col_a not in df.columns or (col_b and col_b not in df.columns):
                    print(f"  跳过 {joint}/{suffix}：缺少列 {col_a} 或 {col_b}")
                    continue

                save_path = os.path.join(joint_dir, f"{suffix}.png")
                plot_comparison(df, joint, col_a, col_b, label_a, label_b, title, save_path)

        print(f"  {joint}/ -> 5 张图已保存")

    print(f"\n完成！所有图片保存在：{DATA_DIR}/")


if __name__ == "__main__":
    main()
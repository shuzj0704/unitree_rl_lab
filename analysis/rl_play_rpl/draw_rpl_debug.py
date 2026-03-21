# *** 说明 ***
# 分析 play_rpl_debug.py 记录的 CSV，画出：
# Fig 1: 各关节 a_base vs 0.1*a_res 幅度对比（确认残差占比）
# Fig 2: 各关节 a_base, 0.1*a_res, a_total 时序曲线
# Fig 3: 残差占比 |0.1*a_res| / (|a_base| + |0.1*a_res|) 随时间变化
# Fig 4: 各关节实际关节位置
# Fig 5: last_action obs vs a_base 验证（确认 bug 已修复）

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# ===================== 配置区 =====================
# 修改这个变量即可切换不同实验
EXPERIMENT = "2026-03-21_01-09-47"
RESIDUAL_SCALE = 0.1
# ==================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, EXPERIMENT)
CSV_FILE = os.path.join(DATA_DIR, "rpl_debug.csv")

JOINT_NAMES = [f"j{i+1}" for i in range(6)]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def fig1_action_magnitude_comparison(df, save_dir):
    """Bar chart: mean |a_base| vs mean |k*a_res| per joint."""
    fig, ax = plt.subplots(figsize=(10, 5))

    a_base_means = []
    a_res_scaled_means = []
    for jn in JOINT_NAMES:
        a_base_means.append(df[f"a_base_{jn}"].abs().mean())
        a_res_scaled_means.append((df[f"a_res_{jn}"] * RESIDUAL_SCALE).abs().mean())

    x = np.arange(len(JOINT_NAMES))
    w = 0.35
    ax.bar(x - w/2, a_base_means, w, label="|a_base|", color="#1f77b4")
    ax.bar(x + w/2, a_res_scaled_means, w, label=f"|{RESIDUAL_SCALE}×a_res|", color="#ff7f0e")

    ax.set_xlabel("Joint")
    ax.set_ylabel("Mean Absolute Value (rad)")
    ax.set_title("Action Magnitude: a_base vs scaled a_res")
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 在柱上标数值
    for i, (b, r) in enumerate(zip(a_base_means, a_res_scaled_means)):
        ax.text(i - w/2, b + 0.002, f"{b:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, r + 0.002, f"{r:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig1_action_magnitude.png"), dpi=200)
    plt.close(fig)
    print("  Fig 1: action magnitude comparison saved")


def fig2_action_timeseries(df, save_dir):
    """Per-joint time series: a_base, k*a_res, a_total."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)

    steps = df["step"]
    for i, jn in enumerate(JOINT_NAMES):
        ax = axes[i]
        a_base = df[f"a_base_{jn}"]
        a_res_scaled = df[f"a_res_{jn}"] * RESIDUAL_SCALE
        a_total = df[f"a_total_{jn}"]

        ax.plot(steps, a_base, label="a_base", color="#1f77b4", linewidth=1)
        ax.plot(steps, a_res_scaled, label=f"{RESIDUAL_SCALE}×a_res", color="#ff7f0e", linewidth=1)
        ax.plot(steps, a_total, label="a_total", color="#2ca02c", linewidth=1, linestyle="--")
        ax.set_ylabel(f"{jn} (rad)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_title("Action Time Series: a_base, scaled a_res, a_total")
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig2_action_timeseries.png"), dpi=200)
    plt.close(fig)
    print("  Fig 2: action time series saved")


def fig3_residual_ratio(df, save_dir):
    """Residual ratio: |k*a_res| / (|a_base| + |k*a_res|) per step, averaged over joints."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    steps = df["step"]

    # Per-joint ratio
    for i, jn in enumerate(JOINT_NAMES):
        a_base_abs = df[f"a_base_{jn}"].abs()
        a_res_abs = (df[f"a_res_{jn}"] * RESIDUAL_SCALE).abs()
        ratio = a_res_abs / (a_base_abs + a_res_abs + 1e-8)
        ax1.plot(steps, ratio, label=jn, color=COLORS[i], linewidth=0.8, alpha=0.7)

    ax1.set_ylabel("Residual Ratio")
    ax1.set_title("Residual Ratio: |k×a_res| / (|a_base| + |k×a_res|) per joint")
    ax1.legend(fontsize=8, ncol=6)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_ylim(0, 1)

    # Mean across joints
    all_base = np.stack([df[f"a_base_{jn}"].abs().values for jn in JOINT_NAMES], axis=1)
    all_res = np.stack([(df[f"a_res_{jn}"] * RESIDUAL_SCALE).abs().values for jn in JOINT_NAMES], axis=1)
    mean_ratio = all_res.mean(axis=1) / (all_base.mean(axis=1) + all_res.mean(axis=1) + 1e-8)

    ax2.plot(steps, mean_ratio, color="#d62728", linewidth=1)
    ax2.axhline(y=np.mean(mean_ratio), color="gray", linestyle="--", linewidth=0.8,
                label=f"mean={np.mean(mean_ratio):.4f}")
    ax2.set_ylabel("Mean Residual Ratio")
    ax2.set_xlabel("Step")
    ax2.set_title("Mean Residual Ratio (averaged over joints)")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig3_residual_ratio.png"), dpi=200)
    plt.close(fig)
    print("  Fig 3: residual ratio saved")


def fig4_actual_joint_pos(df, save_dir):
    """Actual joint positions over time."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)

    steps = df["step"]
    for i, jn in enumerate(JOINT_NAMES):
        ax = axes[i]
        ax.plot(steps, df[f"actual_jpos_{jn}"], color=COLORS[i], linewidth=1)
        ax.set_ylabel(f"{jn} (rad)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_title("Actual Joint Positions")
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig4_actual_jpos.png"), dpi=200)
    plt.close(fig)
    print("  Fig 4: actual joint positions saved")


def fig5_last_action_check(df, save_dir):
    """Verify last_action obs matches a_base (same frame)."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)

    steps = df["step"].values
    for i, jn in enumerate(JOINT_NAMES):
        ax = axes[i]
        last_act = df[f"last_act_obs_{jn}"].values
        a_base = df[f"a_base_{jn}"].values

        # last_base_action returns current frame's a_base (same frame, not lagged)
        diff = np.abs(last_act - a_base)

        ax.plot(steps, diff, color="#d62728", linewidth=0.8)
        ax.set_ylabel(f"{jn} |diff|", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        mean_diff = np.mean(diff)
        ax.set_title(f"{jn}: mean|last_act_obs - a_base| = {mean_diff:.6f}", fontsize=9)

    axes[0].set_title("last_action obs vs a_base (same frame) — should be 0 if bug fixed")
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig5_last_action_check.png"), dpi=200)
    plt.close(fig)
    print("  Fig 5: last_action check saved")


def main():
    if not os.path.exists(CSV_FILE):
        print(f"错误：找不到 '{CSV_FILE}'，请检查 EXPERIMENT 变量。")
        sys.exit(1)

    df = load_data(CSV_FILE)

    save_dir = DATA_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig1_action_magnitude_comparison(df, save_dir)
    fig2_action_timeseries(df, save_dir)
    fig3_residual_ratio(df, save_dir)
    fig4_actual_joint_pos(df, save_dir)
    fig5_last_action_check(df, save_dir)

    print(f"\n完成！所有图片保存在：{save_dir}/")


if __name__ == "__main__":
    main()

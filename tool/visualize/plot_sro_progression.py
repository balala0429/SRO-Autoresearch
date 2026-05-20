import matplotlib.pyplot as plt
import numpy as np


def plot_sro_progression(x_test, y_obs, history, save_path="sro_progression.png"):
    """
    绘制 SRO 推理全过程的动态折线图
    history: list of dicts, 包含每一步的拟合状态
    """
    n_steps = len(history)
    if n_steps == 0:
        print("没有历史记录，无法绘图！")
        return

    # 设置学术风画布大小
    fig, axes = plt.subplots(nrows=2, ncols=n_steps, figsize=(5 * n_steps, 8))

    # 防止只有 1 轮迭代时 axes 变成一维数组报错
    if n_steps == 1:
        axes = np.array([axes]).T

    # 全局 Y 轴统一范围，方便肉眼对比误差缩减
    y_min, y_max = np.min(y_obs) - 1, np.max(y_obs) + 1
    max_res_abs = max([np.max(np.abs(step_data['res'])) for step_data in history])

    for i, data in enumerate(history):
        ax_top = axes[0, i]
        ax_bot = axes[1, i]

        # ==========================================
        # 上半部分：真实观测数据 vs 当前拟合曲线
        # ==========================================
        ax_top.scatter(x_test, y_obs, color='lightgray', s=20, label='Ground Truth', zorder=1)
        ax_top.plot(x_test, data['y_pred'], color='#E63946', linewidth=2.5, label='Fitted Curve', zorder=2)

        ax_top.set_ylim(y_min, y_max)
        ax_top.set_title(f"Iter {data['step']}: + {data['patch']}", fontsize=12, pad=10)
        ax_top.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax_top.set_ylabel("Value (Y)", fontsize=12, fontweight='bold')
            ax_top.legend(loc="upper left")

        # ==========================================
        # 下半部分：残差被“吃掉”的过程
        # ==========================================
        ax_bot.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)
        ax_bot.plot(x_test, data['res'], color='#457B9D', linewidth=2.5, label='Residual Wave', zorder=2)
        # 给残差加个面积填充，视觉冲击力极强
        ax_bot.fill_between(x_test, 0, data['res'], color='#457B9D', alpha=0.15)

        # 固定残差的 Y 轴，直观感受残差越来越平
        ax_bot.set_ylim(-max_res_abs * 1.1, max_res_abs * 1.1)
        ax_bot.set_title(f"Residual MSE: {data['mse']:.6f}", fontsize=11, color='#1D3557')
        ax_bot.grid(True, linestyle='--', alpha=0.6)
        ax_bot.set_xlabel("Input (X)", fontsize=11)
        if i == 0:
            ax_bot.set_ylabel("Error (Residual)", fontsize=12, fontweight='bold')
            ax_bot.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 SRO 拟合过程图已生成并保存至: {save_path}")
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# 设置专业论文风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'
sns.set_style("whitegrid")

# 专业配色
COLORS = {
    'train': '#2C3E50',
    'val': '#E74C3C',
    'ci': '#27AE60',
    'rm2': '#3498DB',
    'lr': '#9B59B6',
    'best': '#F39C12'
}


def symlog_transform(y, threshold1=1, threshold2=5, linear_scale=1.5, log_scale=0.3):
    """
    两段对称对数变换（强化0-1区间）
    - 0到threshold1 (0-1): 完全线性，放大1.5倍
    - >threshold1 (>1): 统一对数压缩（系数0.3）
    """
    y = np.asarray(y, dtype=float)
    result = np.zeros_like(y)
    
    # 第一段：0-1 线性区域，放大显示
    mask_linear = y <= threshold1
    result[mask_linear] = y[mask_linear] * linear_scale
    
    # 第二段：>1 对数压缩（包括1-5和>5）
    mask_log = y > threshold1
    if np.any(mask_log):
        linear_height = threshold1 * linear_scale
        # 使用对数缩放
        result[mask_log] = linear_height + log_scale * np.log1p(y[mask_log] - threshold1)
    
    return result


def load_all_training_data(stage1_path, stage2_path):
    """加载并合并所有训练数据"""
    try:
        df_stage1 = pd.read_csv(stage1_path)
        df_stage2 = pd.read_csv(stage2_path)
        
        df_stage2['epoch'] = df_stage2['epoch'] + df_stage1['epoch'].max()
        df_combined = pd.concat([df_stage1, df_stage2], ignore_index=True)
        
        print(f"✓ 成功加载训练数据")
        print(f"  Stage 1: {len(df_stage1)} epochs")
        print(f"  Stage 2: {len(df_stage2)} epochs")
        print(f"  Total: {len(df_combined)} epochs")
        
        return df_combined
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None


def plot_loss_curves_symlog(df, save_path):
    """绘制训练损失和验证MSE曲线（对称对数缩放）"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 应用对称对数变换（0-1放大1.5倍，>1压缩0.3倍）
    train_loss_transformed = symlog_transform(df['train_loss'].values, 
                                              threshold1=1, 
                                              threshold2=5,
                                              linear_scale=1.5, 
                                              log_scale=0.3)
    val_mse_transformed = symlog_transform(df['val_mse'].values,
                                          threshold1=1,
                                          threshold2=5,
                                          linear_scale=1.5,
                                          log_scale=0.3)
    
    # 绘制曲线
    ax.plot(df['epoch'], train_loss_transformed, 
            color=COLORS['train'], linewidth=2.5, alpha=0.9, 
            label='Training Loss', linestyle='-')
    ax.plot(df['epoch'], val_mse_transformed, 
            color=COLORS['val'], linewidth=2.5, alpha=0.9,
            label='Validation MSE', linestyle='-')
    
    # 设置Y轴刻度（0-1密集，跳过2,3,4）
    tick_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 5, 10, 20, 40, 80, 120]
    tick_positions = symlog_transform(tick_values, 1, 5, 1.5, 0.3)
    
    # 只显示有效范围内的刻度
    y_max = max(train_loss_transformed.max(), val_mse_transformed.max())
    valid_indices = [i for i, pos in enumerate(tick_positions) if pos <= y_max * 1.1]
    
    ax.set_yticks([tick_positions[i] for i in valid_indices])
    ax.set_yticklabels([str(tick_values[i]) for i in valid_indices])
    
    # 添加网格线（对应刻度位置，突出0-1区间）
    for i in valid_indices:
        if tick_values[i] <= 1:
            alpha = 0.5  # 0-1区间网格线更明显
            linewidth = 1.0
        elif tick_values[i] in [2, 5, 10]:
            alpha = 0.3
            linewidth = 0.8
        else:
            alpha = 0.2
            linewidth = 0.6
        ax.axhline(y=tick_positions[i], color='gray', linestyle=':', 
                   linewidth=linewidth, alpha=alpha, zorder=1)
    
    # 设置坐标轴
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss / MSE', fontsize=14, fontweight='bold')
    ax.set_title('Training and Validation Loss', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # 图例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
             edgecolor='black', fancybox=True)
    
    # 设置y轴范围
    ax.set_ylim(-0.2, y_max * 1.05)
    
    # 刻度
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    
    # 移除默认网格，使用自定义的
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 对称对数缩放训练曲线已保存: {save_path}")


def plot_validation_mse_symlog(df, save_path):
    """绘制验证MSE曲线（对称对数缩放）"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 应用对称对数变换（0-1放大1.5倍，>1压缩0.3倍）
    val_mse_transformed = symlog_transform(df['val_mse'].values,
                                          threshold1=1,
                                          threshold2=5,
                                          linear_scale=1.5,
                                          log_scale=0.3)
    
    # 绘制曲线
    ax.plot(df['epoch'], val_mse_transformed, 
            color=COLORS['val'], linewidth=2.8, alpha=0.9, 
            label='Validation MSE')
    
    # 设置Y轴刻度（0-1密集，跳过2,3,4）
    tick_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 5, 10, 20, 40, 80, 120]
    tick_positions = symlog_transform(tick_values, 1, 5, 1.5, 0.3)
    
    y_max = val_mse_transformed.max()
    valid_indices = [i for i, pos in enumerate(tick_positions) if pos <= y_max * 1.1]
    
    ax.set_yticks([tick_positions[i] for i in valid_indices])
    ax.set_yticklabels([str(tick_values[i]) for i in valid_indices])
    
    # 添加网格线（突出0-1区间）
    for i in valid_indices:
        if tick_values[i] <= 1:
            alpha = 0.5  # 0-1区间网格线更明显
            linewidth = 1.0
        elif tick_values[i] in [2, 5, 10]:
            alpha = 0.3
            linewidth = 0.8
        else:
            alpha = 0.2
            linewidth = 0.6
        ax.axhline(y=tick_positions[i], color='gray', linestyle=':', 
                   linewidth=linewidth, alpha=alpha, zorder=1)
    
    # 设置坐标轴
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
    ax.set_title('Validation MSE', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # 图例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
             edgecolor='black', fancybox=True)
    
    # 设置y轴范围
    ax.set_ylim(-0.2, y_max * 1.05)
    
    # 刻度
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 对称对数缩放验证MSE曲线已保存: {save_path}")


def plot_concordance_index(df, save_path):
    """绘制CI曲线（标准线性）"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(df['epoch'], df['val_ci'], 
            color=COLORS['ci'], linewidth=2.8, alpha=0.9, 
            label='Concordance Index')
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concordance Index', fontsize=14, fontweight='bold')
    ax.set_title('Concordance Index', fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95,
             edgecolor='black', fancybox=True)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    y_margin = (df['val_ci'].max() - df['val_ci'].min()) * 0.1
    ax.set_ylim(df['val_ci'].min() - y_margin, df['val_ci'].max() + y_margin)
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ CI曲线已保存: {save_path}")


def plot_modified_r2(df, save_path):
    """绘制Rm²曲线（标准线性）"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(df['epoch'], df['val_rm2'], 
            color=COLORS['rm2'], linewidth=2.8, alpha=0.9, 
            label='Modified Rm²')
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Modified Rm²', fontsize=14, fontweight='bold')
    ax.set_title('Modified Rm²', fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95,
             edgecolor='black', fancybox=True)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    y_margin = (df['val_rm2'].max() - df['val_rm2'].min()) * 0.1
    ax.set_ylim(df['val_rm2'].min() - y_margin, df['val_rm2'].max() + y_margin)
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Rm²曲线已保存: {save_path}")


def plot_learning_rate(df, save_path):
    """绘制学习率曲线"""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(df['epoch'], df['lr'], 
            color=COLORS['lr'], linewidth=2.8, alpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold', pad=15)
    
    ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='both')
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 学习率曲线已保存: {save_path}")


def plot_combined_metrics(df, save_path):
    """绘制三个指标的合并图（MSE、CI、Rm²）"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12), sharex=True)
    
    # 图1: 验证MSE（使用对数缩放）
    val_mse_transformed = symlog_transform(df['val_mse'].values,
                                          threshold1=1,
                                          threshold2=5,
                                          linear_scale=1.5,
                                          log_scale=0.3)
    
    ax1.plot(df['epoch'], val_mse_transformed, 
            color=COLORS['val'], linewidth=2.5, alpha=0.9, 
            label='Validation MSE')
    
    # 设置Y轴刻度
    tick_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 5, 10, 20, 40, 80, 120]
    tick_positions = symlog_transform(tick_values, 1, 5, 1.5, 0.3)
    
    y_max = val_mse_transformed.max()
    valid_indices = [i for i, pos in enumerate(tick_positions) if pos <= y_max * 1.1]
    
    ax1.set_yticks([tick_positions[i] for i in valid_indices])
    ax1.set_yticklabels([f'{tick_values[i]:.1f}' if tick_values[i] < 1 else str(int(tick_values[i])) for i in valid_indices])
    
    # 添加网格线
    for i in valid_indices:
        if tick_values[i] <= 1:
            alpha = 0.5
            linewidth = 1.0
        elif tick_values[i] in [2, 5, 10]:
            alpha = 0.3
            linewidth = 0.8
        else:
            alpha = 0.2
            linewidth = 0.6
        ax1.axhline(y=tick_positions[i], color='gray', linestyle=':', 
                   linewidth=linewidth, alpha=alpha, zorder=1)
    
    ax1.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.2, y_max * 1.05)
    ax1.tick_params(axis='both', which='major', labelsize=10, width=1.2, length=5)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black')
    ax1.grid(False)
    
    # 图2: CI（标准线性）
    ax2.plot(df['epoch'], df['val_ci'], 
            color=COLORS['ci'], linewidth=2.5, alpha=0.9, 
            label='Concordance Index')
    
    ax2.set_ylabel('CI', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='black')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    y_margin = (df['val_ci'].max() - df['val_ci'].min()) * 0.1
    ax2.set_ylim(df['val_ci'].min() - y_margin, df['val_ci'].max() + y_margin)
    ax2.tick_params(axis='both', which='major', labelsize=10, width=1.2, length=5)
    
    # 图3: Rm²（标准线性）
    ax3.plot(df['epoch'], df['val_rm2'], 
            color=COLORS['rm2'], linewidth=2.5, alpha=0.9, 
            label='Modified Rm²')
    
    ax3.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Rm²', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.set_axisbelow(True)
    
    y_margin = (df['val_rm2'].max() - df['val_rm2'].min()) * 0.1
    ax3.set_ylim(df['val_rm2'].min() - y_margin, df['val_rm2'].max() + y_margin)
    ax3.tick_params(axis='both', which='major', labelsize=10, width=1.2, length=5)
    
    # 总标题
    fig.suptitle('Validation Metrics', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 三指标合并图已保存: {save_path}")


def plot_figure1_combined(df, save_path):
    """生成Figure 1：2×2布局（Loss、MSE、CI、Rm²）"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 子图1：训练和验证损失
    ax1 = fig.add_subplot(gs[0, 0])
    train_loss_transformed = symlog_transform(df['train_loss'].values, 1, 5, 1.5, 0.3)
    val_mse_transformed = symlog_transform(df['val_mse'].values, 1, 5, 1.5, 0.3)
    
    ax1.plot(df['epoch'], train_loss_transformed, 
            color=COLORS['train'], linewidth=2.2, alpha=0.9, label='Training Loss')
    ax1.plot(df['epoch'], val_mse_transformed, 
            color=COLORS['val'], linewidth=2.2, alpha=0.9, label='Validation MSE')
    
    tick_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 5, 10, 20, 40, 80, 120]
    tick_positions = symlog_transform(tick_values, 1, 5, 1.5, 0.3)
    y_max = max(train_loss_transformed.max(), val_mse_transformed.max())
    valid_indices = [i for i, pos in enumerate(tick_positions) if pos <= y_max * 1.1]
    
    ax1.set_yticks([tick_positions[i] for i in valid_indices])
    ax1.set_yticklabels([f'{tick_values[i]:.1f}' if tick_values[i] < 1 else str(int(tick_values[i])) for i in valid_indices])
    
    for i in valid_indices:
        alpha = 0.5 if tick_values[i] <= 1 else 0.3 if tick_values[i] in [2, 5, 10] else 0.2
        linewidth = 1.0 if tick_values[i] <= 1 else 0.8 if tick_values[i] in [2, 5, 10] else 0.6
        ax1.axhline(y=tick_positions[i], color='gray', linestyle=':', linewidth=linewidth, alpha=alpha, zorder=1)
    
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss / MSE', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black')
    ax1.set_ylim(-0.2, y_max * 1.05)
    ax1.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=5)
    ax1.grid(False)
    
    # 子图2：验证MSE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['epoch'], val_mse_transformed, 
            color=COLORS['val'], linewidth=2.5, alpha=0.9, label='Validation MSE')
    
    y_max_mse = val_mse_transformed.max()
    valid_indices_mse = [i for i, pos in enumerate(tick_positions) if pos <= y_max_mse * 1.1]
    
    ax2.set_yticks([tick_positions[i] for i in valid_indices_mse])
    ax2.set_yticklabels([f'{tick_values[i]:.1f}' if tick_values[i] < 1 else str(int(tick_values[i])) for i in valid_indices_mse])
    
    for i in valid_indices_mse:
        alpha = 0.5 if tick_values[i] <= 1 else 0.3 if tick_values[i] in [2, 5, 10] else 0.2
        linewidth = 1.0 if tick_values[i] <= 1 else 0.8 if tick_values[i] in [2, 5, 10] else 0.6
        ax2.axhline(y=tick_positions[i], color='gray', linestyle=':', linewidth=linewidth, alpha=alpha, zorder=1)
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Validation MSE', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black')
    ax2.set_ylim(-0.2, y_max_mse * 1.05)
    ax2.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=5)
    ax2.grid(False)
    
    # 子图3：CI
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['epoch'], df['val_ci'], 
            color=COLORS['ci'], linewidth=2.5, alpha=0.9, label='Concordance Index')
    
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('CI', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Concordance Index', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.set_axisbelow(True)
    
    y_margin = (df['val_ci'].max() - df['val_ci'].min()) * 0.1
    ax3.set_ylim(df['val_ci'].min() - y_margin, df['val_ci'].max() + y_margin)
    ax3.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=5)
    
    # 子图4：Rm²
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['epoch'], df['val_rm2'], 
            color=COLORS['rm2'], linewidth=2.5, alpha=0.9, label='Modified Rm²')
    
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Rm²', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Modified Rm²', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='black')
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax4.set_axisbelow(True)
    
    y_margin = (df['val_rm2'].max() - df['val_rm2'].min()) * 0.1
    ax4.set_ylim(df['val_rm2'].min() - y_margin, df['val_rm2'].max() + y_margin)
    ax4.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=5)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figure 1 (2×2) 已保存: {save_path}")


def plot_figure5_combined(df, save_path):
    """生成Figure 5：2×2布局（Training Loss、Val MSE、CI、Rm²）"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 子图1：训练损失（单独）
    ax1 = fig.add_subplot(gs[0, 0])
    train_loss_transformed = symlog_transform(df['train_loss'].values, 1, 5, 1.5, 0.3)
    
    ax1.plot(df['epoch'], train_loss_transformed, 
            color=COLORS['train'], linewidth=2.5, alpha=0.9, label='Training Loss')
    
    tick_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 5, 10, 20, 40, 80, 120]
    tick_positions = symlog_transform(tick_values, 1, 5, 1.5, 0.3)
    y_max = train_loss_transformed.max()
    valid_indices = [i for i, pos in enumerate(tick_positions) if pos <= y_max * 1.1]
    
    ax1.set_yticks([tick_positions[i] for i in valid_indices])
    ax1.set_yticklabels([f'{tick_values[i]:.1f}' if tick_values[i] < 1 else str(int(tick_values[i])) for i in valid_indices])
    
    for i in valid_indices:
        alpha = 0.5 if tick_values[i] <= 1 else 0.3 if tick_values[i] in [2, 5, 10] else 0.2
        linewidth = 1.0 if tick_values[i] <= 1 else 0.8 if tick_values[i] in [2, 5, 10] else 0.6
        ax1.axhline(y=tick_positions[i], color='gray', linestyle=':', linewidth=linewidth, alpha=alpha, zorder=1)
    
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Training Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black')
    ax1.set_ylim(-0.2, y_max * 1.05)
    ax1.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=5)
    ax1.grid(False)
    
    # 子图2：验证MSE
    ax2 = fig.add_subplot(gs[0, 1])
    val_mse_transformed = symlog_transform(df['val_mse'].values, 1, 5, 1.5, 0.3)
    
    ax2.plot(df['epoch'], val_mse_transformed, 
            color=COLORS['val'], linewidth=2.5, alpha=0.9, label='Validation MSE')
    
    y_max_mse = val_mse_transformed.max()
    valid_indices_mse = [i for i, pos in enumerate(tick_positions) if pos <= y_max_mse * 1.1]
    
    ax2.set_yticks([tick_positions[i] for i in valid_indices_mse])
    ax2.set_yticklabels([f'{tick_values[i]:.1f}' if tick_values[i] < 1 else str(int(tick_values[i])) for i in valid_indices_mse])
    
    for i in valid_indices_mse:
        alpha = 0.5 if tick_values[i] <= 1 else 0.3 if tick_values[i] in [2, 5, 10] else 0.2
        linewidth = 1.0 if tick_values[i] <= 1 else 0.8 if tick_values[i] in [2, 5, 10] else 0.6
        ax2.axhline(y=tick_positions[i], color='gray', linestyle=':', linewidth=linewidth, alpha=alpha, zorder=1)
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Validation MSE', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black')
    ax2.set_ylim(-0.2, y_max_mse * 1.05)
    ax2.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=5)
    ax2.grid(False)
    
    # 子图3：CI
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['epoch'], df['val_ci'], 
            color=COLORS['ci'], linewidth=2.5, alpha=0.9, label='Concordance Index')
    
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('CI', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Concordance Index', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.set_axisbelow(True)
    
    y_margin = (df['val_ci'].max() - df['val_ci'].min()) * 0.1
    ax3.set_ylim(df['val_ci'].min() - y_margin, df['val_ci'].max() + y_margin)
    ax3.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=5)
    
    # 子图4：Rm²
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['epoch'], df['val_rm2'], 
            color=COLORS['rm2'], linewidth=2.5, alpha=0.9, label='Modified Rm²')
    
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Rm²', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Modified Rm²', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='black')
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax4.set_axisbelow(True)
    
    y_margin = (df['val_rm2'].max() - df['val_rm2'].min()) * 0.1
    ax4.set_ylim(df['val_rm2'].min() - y_margin, df['val_rm2'].max() + y_margin)
    ax4.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=5)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figure 5 (2×2) 已保存: {save_path}")


def plot_five_panels_combined(df, save_path):
    """生成五张图的完整合并图"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, 
                          height_ratios=[1, 1, 1])
    
    tick_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 5, 10, 20, 40, 80, 120]
    tick_positions = symlog_transform(tick_values, 1, 5, 1.5, 0.3)
    
    # 子图1：训练和验证损失 (左上)
    ax1 = fig.add_subplot(gs[0, 0])
    train_loss_transformed = symlog_transform(df['train_loss'].values, 1, 5, 1.5, 0.3)
    val_mse_transformed = symlog_transform(df['val_mse'].values, 1, 5, 1.5, 0.3)
    
    ax1.plot(df['epoch'], train_loss_transformed, 
            color=COLORS['train'], linewidth=2.2, alpha=0.9, label='Training Loss')
    ax1.plot(df['epoch'], val_mse_transformed, 
            color=COLORS['val'], linewidth=2.2, alpha=0.9, label='Validation MSE')
    
    y_max = max(train_loss_transformed.max(), val_mse_transformed.max())
    valid_indices = [i for i, pos in enumerate(tick_positions) if pos <= y_max * 1.1]
    
    ax1.set_yticks([tick_positions[i] for i in valid_indices])
    ax1.set_yticklabels([f'{tick_values[i]:.1f}' if tick_values[i] < 1 else str(int(tick_values[i])) for i in valid_indices])
    
    for i in valid_indices:
        alpha = 0.5 if tick_values[i] <= 1 else 0.3
        linewidth = 1.0 if tick_values[i] <= 1 else 0.6
        ax1.axhline(y=tick_positions[i], color='gray', linestyle=':', linewidth=linewidth, alpha=alpha, zorder=1)
    
    ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Loss / MSE', fontsize=10, fontweight='bold')
    ax1.set_title('(A) Training and Validation Loss', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.95, edgecolor='black')
    ax1.set_ylim(-0.2, y_max * 1.05)
    ax1.tick_params(axis='both', which='major', labelsize=8, width=1.2, length=4)
    ax1.grid(False)
    
    # 子图2：验证MSE (右上)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['epoch'], val_mse_transformed, 
            color=COLORS['val'], linewidth=2.5, alpha=0.9, label='Validation MSE')
    
    y_max_mse = val_mse_transformed.max()
    valid_indices_mse = [i for i, pos in enumerate(tick_positions) if pos <= y_max_mse * 1.1]
    
    ax2.set_yticks([tick_positions[i] for i in valid_indices_mse])
    ax2.set_yticklabels([f'{tick_values[i]:.1f}' if tick_values[i] < 1 else str(int(tick_values[i])) for i in valid_indices_mse])
    
    for i in valid_indices_mse:
        alpha = 0.5 if tick_values[i] <= 1 else 0.3
        linewidth = 1.0 if tick_values[i] <= 1 else 0.6
        ax2.axhline(y=tick_positions[i], color='gray', linestyle=':', linewidth=linewidth, alpha=alpha, zorder=1)
    
    ax2.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=10, fontweight='bold')
    ax2.set_title('(B) Validation MSE', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.95, edgecolor='black')
    ax2.set_ylim(-0.2, y_max_mse * 1.05)
    ax2.tick_params(axis='both', which='major', labelsize=8, width=1.2, length=4)
    ax2.grid(False)
    
    # 子图3：CI (左中)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['epoch'], df['val_ci'], 
            color=COLORS['ci'], linewidth=2.5, alpha=0.9, label='Concordance Index')
    
    ax3.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax3.set_ylabel('CI', fontsize=10, fontweight='bold')
    ax3.set_title('(C) Concordance Index', fontsize=11, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8, framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.set_axisbelow(True)
    
    y_margin = (df['val_ci'].max() - df['val_ci'].min()) * 0.1
    ax3.set_ylim(df['val_ci'].min() - y_margin, df['val_ci'].max() + y_margin)
    ax3.tick_params(axis='both', which='major', labelsize=8, width=1.2, length=4)
    
    # 子图4：Rm² (右中)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['epoch'], df['val_rm2'], 
            color=COLORS['rm2'], linewidth=2.5, alpha=0.9, label='Modified Rm²')
    
    ax4.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Rm²', fontsize=10, fontweight='bold')
    ax4.set_title('(D) Modified Rm²', fontsize=11, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=8, framealpha=0.95, edgecolor='black')
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax4.set_axisbelow(True)
    
    y_margin = (df['val_rm2'].max() - df['val_rm2'].min()) * 0.1
    ax4.set_ylim(df['val_rm2'].min() - y_margin, df['val_rm2'].max() + y_margin)
    ax4.tick_params(axis='both', which='major', labelsize=8, width=1.2, length=4)
    
    # 子图5：学习率 (下方跨两列)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(df['epoch'], df['lr'], 
            color=COLORS['lr'], linewidth=2.5, alpha=0.9, label='Learning Rate')
    
    ax5.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Learning Rate', fontsize=10, fontweight='bold')
    ax5.set_title('(E) Learning Rate Schedule', fontsize=11, fontweight='bold')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='both')
    ax5.set_axisbelow(True)
    ax5.tick_params(axis='both', which='major', labelsize=8, width=1.2, length=4)
    ax5.legend(loc='upper right', fontsize=8, framealpha=0.95, edgecolor='black')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 五图合并 已保存: {save_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" "*12 + "DTA模型可视化 - 平滑自然版")
    print("="*70 + "\n")
    
    stage1_path = '/root/1/result/training_stage1.csv'
    stage2_path = '/root/1/result/training_stage2.csv'
    output_dir = '/root/1/result/paper_figures_smooth'
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("📂 加载训练数据...")
    df = load_all_training_data(stage1_path, stage2_path)
    
    if df is None:
        print("✗ 无法加载训练数据，退出")
        return
    
    print("\n🎨 开始生成平滑自然版图表...\n")
    
    # 1. 训练和验证损失
    plot_loss_curves_symlog(df, f'{output_dir}/fig1_loss_curves.png')
    
    # 2. 验证MSE
    plot_validation_mse_symlog(df, f'{output_dir}/fig2_validation_mse.png')
    
    # 3. CI
    plot_concordance_index(df, f'{output_dir}/fig3_concordance_index.png')
    
    # 4. Rm²
    plot_modified_r2(df, f'{output_dir}/fig4_modified_r2.png')
    
    # 5. 学习率
    plot_learning_rate(df, f'{output_dir}/fig5_learning_rate.png')
    
    # 6. 三指标合并图
    plot_combined_metrics(df, f'{output_dir}/fig6_combined_metrics.png')
    
    # 7. Figure 1 (2×2)
    plot_figure1_combined(df, f'{output_dir}/fig7_figure1_2x2.png')
    
    # 8. Figure 5 (2×2)
    plot_figure5_combined(df, f'{output_dir}/fig8_figure5_2x2.png')
    
    # 9. 五图合并
    plot_five_panels_combined(df, f'{output_dir}/fig9_five_panels.png')
    
    print("\n" + "="*70)
    print("✅ 所有平滑自然版图表已生成完成!")
    print("="*70)
    print(f"\n📁 输出目录: {output_dir}")
    print("\n生成的图表:")
    print("  1. fig1_loss_curves.png - 训练和验证损失")
    print("  2. fig2_validation_mse.png - 验证MSE")
    print("  3. fig3_concordance_index.png - CI曲线")
    print("  4. fig4_modified_r2.png - Rm²曲线")
    print("  5. fig5_learning_rate.png - 学习率曲线")
    print("  6. fig6_combined_metrics.png - 三指标合并图（MSE+CI+Rm²）")
    print("  7. fig7_figure1_2x2.png - Figure 1 (2×2布局)")
    print("  8. fig8_figure5_2x2.png - Figure 5 (2×2布局)")
    print("  9. fig9_five_panels.png - 五图合并 (Loss+MSE+CI+Rm²+LR)")
    print("\n💡 特点:")
    print("  ✓ 使用两段对称对数缩放（强化0-1区间）")
    print("  ✓ 0-1区间：放大1.5倍显示（占据更多空间）")
    print("  ✓ >1区间：压缩到0.3倍（大幅缩短）")
    print("  ✓ 视觉效果：重点突出低值区域")
    print("  ✓ Y轴刻度：0, 0.2, 0.4, 0.6, 0.8, 1, 5, 10, 20, 40, 80, 120")
    print("  ✓ 图表尺寸：7×5英寸（紧凑）")
    print("  ✓ 不显示Best参考线，更简洁")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
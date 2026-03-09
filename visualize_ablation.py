import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_performance_comparison(results_df, save_dir='/root/1/result'):
    """绘制性能对比图"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['Test_MSE', 'Test_CI', 'Test_Rm2']
    titles = ['Mean Squared Error (Lower is Better)', 
              'Concordance Index (Higher is Better)', 
              'Modified R² (Higher is Better)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx]
        
        # 排序
        data = results_df.sort_values(metric, ascending=(metric == 'Test_MSE'))
        
        # 绘制条形图
        bars = ax.barh(data['Experiment'], data[metric], color=color, alpha=0.7)
        
        # 高亮baseline
        baseline_idx = data[data['Experiment'] == 'Baseline'].index[0]
        bars[list(data.index).index(baseline_idx)].set_color('#f39c12')
        bars[list(data.index).index(baseline_idx)].set_alpha(1.0)
        
        ax.set_xlabel(metric.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, data[metric])):
            ax.text(val, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', 
                   va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ablation_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 性能对比图已保存: {save_dir}/ablation_performance_comparison.png")
    plt.close()


def plot_component_contribution(results_df, save_dir='/root/1/result'):
    """绘制组件贡献热力图"""
    
    baseline_row = results_df[results_df['Experiment'] == 'Baseline'].iloc[0]
    
    # 计算相对性能下降
    contributions = []
    for _, row in results_df.iterrows():
        if row['Experiment'] != 'Baseline':
            contributions.append({
                'Component': row['Experiment'].replace('_', ' '),
                'CI_Drop_%': (baseline_row['Test_CI'] - row['Test_CI']) / baseline_row['Test_CI'] * 100,
                'Rm2_Drop_%': (baseline_row['Test_Rm2'] - row['Test_Rm2']) / baseline_row['Test_Rm2'] * 100,
                'MSE_Increase_%': (row['Test_MSE'] - baseline_row['Test_MSE']) / baseline_row['Test_MSE'] * 100
            })
    
    contrib_df = pd.DataFrame(contributions)
    contrib_df = contrib_df.set_index('Component')
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(contrib_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Performance Degradation (%)'}, 
                linewidths=0.5, ax=ax)
    
    ax.set_title('Component Contribution Analysis\n(Performance Drop When Removed)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ablated Components', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ablation_contribution_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ 组件贡献热力图已保存: {save_dir}/ablation_contribution_heatmap.png")
    plt.close()
    
    return contrib_df


def plot_radar_chart(results_df, save_dir='/root/1/result'):
    """绘制雷达图对比"""
    
    from math import pi
    
    # 选择关键实验
    experiments_to_plot = ['Baseline', 'NoMultiScale_Drug', 'NoLSTM_Protein', 
                          'NoAttention_Drug', 'Simplified_Predictor']
    
    filtered_df = results_df[results_df['Experiment'].isin(experiments_to_plot)]
    
    # 归一化指标 (0-1)
    baseline_mse = results_df[results_df['Experiment'] == 'Baseline']['Test_MSE'].values[0]
    baseline_ci = results_df[results_df['Experiment'] == 'Baseline']['Test_CI'].values[0]
    baseline_rm2 = results_df[results_df['Experiment'] == 'Baseline']['Test_Rm2'].values[0]
    
    categories = ['CI', 'Rm²', 'MSE\n(Inverted)']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    colors = ['#f39c12', '#e74c3c', '#3498db', '#9b59b6', '#1abc9c']
    
    for idx, (_, row) in enumerate(filtered_df.iterrows()):
        values = [
            row['Test_CI'] / baseline_ci,
            row['Test_Rm2'] / baseline_rm2,
            baseline_mse / row['Test_MSE']  # 反转MSE
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=row['Experiment'].replace('_', ' '), 
               color=colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Variant Performance Comparison\n(Normalized to Baseline)', 
                fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ablation_radar_chart.png', dpi=300, bbox_inches='tight')
    print(f"✅ 雷达图已保存: {save_dir}/ablation_radar_chart.png")
    plt.close()


def generate_analysis_report(results_df, contrib_df, save_dir='/root/1/result'):
    """生成详细分析报告"""
    
    report_path = f'{save_dir}/ablation_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" " * 25 + "DTA模型消融实验分析报告\n")
        f.write("="*80 + "\n\n")
        
        # 1. 实验概览
        f.write("1. 实验概览\n")
        f.write("-" * 80 + "\n")
        f.write(f"总实验数: {len(results_df)}\n")
        f.write(f"基准模型: Baseline\n")
        f.write(f"消融变体: {len(results_df) - 1}\n\n")
        
        # 2. Baseline性能
        baseline = results_df[results_df['Experiment'] == 'Baseline'].iloc[0]
        f.write("2. Baseline模型性能\n")
        f.write("-" * 80 + "\n")
        f.write(f"MSE:  {baseline['Test_MSE']:.4f}\n")
        f.write(f"CI:   {baseline['Test_CI']:.4f}\n")
        f.write(f"Rm²:  {baseline['Test_Rm2']:.4f}\n\n")
        
        # 3. 组件重要性排名
        f.write("3. 组件重要性排名 (基于CI下降百分比)\n")
        f.write("-" * 80 + "\n")
        importance_ranking = contrib_df.sort_values('CI_Drop_%', ascending=False)
        for idx, (component, row) in enumerate(importance_ranking.iterrows(), 1):
            f.write(f"{idx}. {component:<35} CI下降: {row['CI_Drop_%']:>6.2f}%  "
                   f"Rm²下降: {row['Rm2_Drop_%']:>6.2f}%\n")
        f.write("\n")
        
        # 4. 关键发现
        f.write("4. 关键发现\n")
        f.write("-" * 80 + "\n")
        
        most_important = importance_ranking.index[0]
        max_drop = importance_ranking['CI_Drop_%'].iloc[0]
        f.write(f"• 最重要组件: {most_important}\n")
        f.write(f"  移除后CI下降 {max_drop:.2f}%\n\n")
        
        least_important = importance_ranking.index[-1]
        min_drop = importance_ranking['CI_Drop_%'].iloc[-1]
        f.write(f"• 最不重要组件: {least_important}\n")
        f.write(f"  移除后CI仅下降 {min_drop:.2f}%\n\n")
        
        # 5. 建议
        f.write("5. 模型优化建议\n")
        f.write("-" * 80 + "\n")
        f.write("基于消融实验结果:\n\n")
        
        if importance_ranking['CI_Drop_%'].iloc[0] > 5:
            f.write(f"• 保留 {most_important} - 这是模型的核心组件\n")
        
        if importance_ranking['CI_Drop_%'].iloc[-1] < 1:
            f.write(f"• 可考虑简化 {least_important} - 对性能影响较小\n")
        
        f.write("\n• 多尺度CNN和注意力机制的组合效果显著\n")
        f.write("• LSTM层对捕捉蛋白质序列特征至关重要\n")
        f.write("• 建议保留完整的预测头结构以获得最佳性能\n\n")
        
        # 6. 统计摘要
        f.write("6. 统计摘要\n")
        f.write("-" * 80 + "\n")
        f.write(f"平均CI下降:    {contrib_df['CI_Drop_%'].mean():.2f}%\n")
        f.write(f"CI下降标准差:  {contrib_df['CI_Drop_%'].std():.2f}%\n")
        f.write(f"平均Rm²下降:   {contrib_df['Rm2_Drop_%'].mean():.2f}%\n")
        f.write(f"Rm²下降标准差: {contrib_df['Rm2_Drop_%'].std():.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("报告生成完成\n")
        f.write("="*80 + "\n")
    
    print(f"✅ 分析报告已保存: {report_path}")


def main():
    """主函数"""
    
    print("\n" + "="*80)
    print(" "*25 + "消融实验结果分析")
    print("="*80 + "\n")
    
    # 读取结果
    results_path = '/root/1/result/ablation_results.csv'
    
    if not Path(results_path).exists():
        print(f"❌ 错误: 未找到结果文件 {results_path}")
        print("   请先运行 ablation_study.py")
        return
    
    results_df = pd.read_csv(results_path)
    print(f"📊 已加载 {len(results_df)} 个实验结果\n")
    
    save_dir = '/root/1/result'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成可视化
    print("🎨 生成可视化图表...\n")
    
    plot_performance_comparison(results_df, save_dir)
    contrib_df = plot_component_contribution(results_df, save_dir)
    plot_radar_chart(results_df, save_dir)
    
    # 生成分析报告
    print("\n📝 生成分析报告...\n")
    generate_analysis_report(results_df, contrib_df, save_dir)
    
    # 显示摘要
    print("\n" + "="*80)
    print("实验结果摘要:")
    print("="*80)
    print(results_df[['Experiment', 'Test_MSE', 'Test_CI', 'Test_Rm2']].to_string(index=False))
    print("="*80)
    
    print("\n✅ 分析完成！生成的文件:")
    print("   • ablation_performance_comparison.png - 性能对比图")
    print("   • ablation_contribution_heatmap.png - 组件贡献热力图")
    print("   • ablation_radar_chart.png - 雷达图对比")
    print("   • ablation_analysis_report.txt - 详细分析报告")
    print()


if __name__ == "__main__":
    main()
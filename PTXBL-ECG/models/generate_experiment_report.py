# -*- coding: utf-8 -*-
"""
ECG分类实验报告生成脚本

生成包含所有模型性能对比的完整实验报告
"""

import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import pandas as pd

def load_results(results_path):
    """加载结果文件"""
    try:
        if os.path.exists(results_path):
            return np.load(results_path, allow_pickle=True).item()
    except Exception as e:
        print(f"加载结果文件失败 {results_path}: {e}")
    return None

def load_all_results():
    """加载所有模型的结果"""
    results = {}
    
    # 基线模型结果
    baseline_models = {
        'CNN_Baseline': './results/cnn_baseline_results.npy',
        'RNN_LSTM_Baseline': './results/lstm_baseline_results.npy',
        'RNN_GRU_Baseline': './results/gru_baseline_results.npy',
        'ResNet_Baseline': './results/resnet_baseline_results.npy',
        'DenseNet_Baseline': './results/densenet_baseline_results.npy',
        'Inception_Baseline': './results/inception_baseline_results.npy',
        'EfficientNet_Baseline': './results/efficientnet_baseline_results.npy',
        'VisionTransformer_Baseline': './results/vit_baseline_results.npy'
    }
    
    # 自定义模型结果
    custom_models = {
        'Original_Model': './results/ptbxl_advanced_optimized_results.npy',
        'V2_Improved': './results/ptbxl_improved_v2_results.npy',
        'V3_Improved': './results/ptbxl_improved_v3_results.npy',
        'Final_Optimized': './results/ptbxl_final_optimized_results.npy'
    }
    
    # 加载基线模型结果
    for model_name, result_path in baseline_models.items():
        result = load_results(result_path)
        if result:
            results[model_name] = result
            print(f"✅ 加载 {model_name} 结果")
        else:
            print(f"❌ 未找到 {model_name} 结果")
    
    # 加载自定义模型结果
    for model_name, result_path in custom_models.items():
        result = load_results(result_path)
        if result:
            results[model_name] = result
            print(f"✅ 加载 {model_name} 结果")
        else:
            print(f"❌ 未找到 {model_name} 结果")
    
    return results

def create_performance_summary(results):
    """创建性能总结"""
    summary = []
    
    for model_name, result in results.items():
        if 'macro_f1' in result:
            summary.append({
                'Model': model_name,
                'Macro_F1': result['macro_f1'],
                'Micro_F1': result.get('micro_f1', 0),
                'Weighted_F1': result.get('weighted_f1', 0),
                'Best_Val_F1': result.get('best_val_f1', 0)
            })
    
    # 按Macro F1排序
    summary.sort(key=lambda x: x['Macro_F1'], reverse=True)
    return summary

def create_class_performance_analysis(results):
    """创建各类别性能分析"""
    # 选择性能最好的模型进行详细分析
    best_model_name = None
    best_f1 = 0
    
    for model_name, result in results.items():
        if 'macro_f1' in result and result['macro_f1'] > best_f1:
            best_f1 = result['macro_f1']
            best_model_name = model_name
    
    if not best_model_name:
        return None
    
    best_result = results[best_model_name]
    
    if 'test_predictions' in best_result and 'test_true' in best_result:
        test_preds = best_result['test_predictions']
        test_true = best_result['test_true']
        
        classes = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
        class_metrics = []
        
        for i, class_name in enumerate(classes):
            precision = precision_score(test_true[:, i], test_preds[:, i], zero_division=0)
            recall = recall_score(test_true[:, i], test_preds[:, i], zero_division=0)
            f1 = f1_score(test_true[:, i], test_preds[:, i], zero_division=0)
            
            class_metrics.append({
                'Class': class_name,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Sample_Count': int(test_true[:, i].sum())
            })
        
        return {
            'best_model': best_model_name,
            'best_f1': best_f1,
            'class_metrics': class_metrics,
            'total_samples': len(test_true)
        }
    
    return None

def create_visualizations(summary, class_analysis):
    """创建可视化图表"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 模型性能排名
    model_names = [item['Model'] for item in summary]
    macro_f1_scores = [item['Macro_F1'] for item in summary]
    
    bars = ax1.barh(range(len(model_names)), macro_f1_scores, color='skyblue', alpha=0.8)
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names, fontsize=10)
    ax1.set_xlabel('Macro F1 Score', fontsize=12)
    ax1.set_title('模型性能排名 (按Macro F1排序)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, macro_f1_scores)):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}', 
                va='center', ha='left', fontsize=9)
    
    # 2. 各类别F1分数对比
    if class_analysis:
        classes = [item['Class'] for item in class_analysis['class_metrics']]
        class_f1_scores = [item['F1_Score'] for item in class_analysis['class_metrics']]
        sample_counts = [item['Sample_Count'] for item in class_analysis['class_metrics']]
        
        bars = ax2.bar(classes, class_f1_scores, color='lightcoral', alpha=0.8)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title(f'各类别F1分数 ({class_analysis["best_model"]})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加样本数量标签
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # 3. 性能指标热力图
    metrics_data = []
    for item in summary:
        metrics_data.append([
            item['Macro_F1'],
            item['Micro_F1'],
            item['Weighted_F1']
        ])
    
    metrics_df = pd.DataFrame(metrics_data, 
                            index=[item['Model'] for item in summary],
                            columns=['Macro F1', 'Micro F1', 'Weighted F1'])
    
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3)
    ax3.set_title('模型性能指标热力图', fontsize=14, fontweight='bold')
    
    # 4. 类别分布饼图
    if class_analysis:
        classes = [item['Class'] for item in class_analysis['class_metrics']]
        sample_counts = [item['Sample_Count'] for item in class_analysis['class_metrics']]
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        wedges, texts, autotexts = ax4.pie(sample_counts, labels=classes, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax4.set_title('测试集类别分布', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def generate_report(summary, class_analysis, results):
    """生成实验报告"""
    report = []
    
    # 报告头部
    report.append("# ECG分类任务实验报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report.append("")
    report.append("## 实验概述")
    report.append("")
    report.append("本实验旨在比较不同深度学习架构在ECG分类任务上的性能表现。")
    report.append("数据集：PTB-XL ECG数据集")
    report.append("任务：5类ECG异常分类 (CD, HYP, MI, NORM, STTC)")
    report.append("")
    
    # 模型性能总结
    report.append("## 模型性能总结")
    report.append("")
    report.append("| 排名 | 模型名称 | Macro F1 | Micro F1 | Weighted F1 | 最佳验证F1 |")
    report.append("|------|----------|----------|----------|-------------|------------|")
    
    for i, item in enumerate(summary, 1):
        report.append(f"| {i} | {item['Model']} | {item['Macro_F1']:.4f} | "
                     f"{item['Micro_F1']:.4f} | {item['Weighted_F1']:.4f} | "
                     f"{item['Best_Val_F1']:.4f} |")
    
    report.append("")
    
    # 关键发现
    report.append("## 关键发现")
    report.append("")
    
    if len(summary) > 1:
        best_model = summary[0]
        worst_model = summary[-1]
        
        improvement = ((best_model['Macro_F1'] - worst_model['Macro_F1']) / 
                     worst_model['Macro_F1']) * 100
        
        report.append(f"- **最佳模型**: {best_model['Model']} (Macro F1: {best_model['Macro_F1']:.4f})")
        report.append(f"- **最差模型**: {worst_model['Model']} (Macro F1: {worst_model['Macro_F1']:.4f})")
        report.append(f"- **性能提升**: 相对提升 {improvement:.2f}%")
        report.append("")
    
    # 各类别性能分析
    if class_analysis:
        report.append("## 各类别性能分析")
        report.append("")
        report.append(f"基于最佳模型: **{class_analysis['best_model']}**")
        report.append("")
        report.append("| 类别 | 精确率 | 召回率 | F1分数 | 样本数量 |")
        report.append("|------|--------|--------|--------|----------|")
        
        for item in class_analysis['class_metrics']:
            report.append(f"| {item['Class']} | {item['Precision']:.4f} | "
                         f"{item['Recall']:.4f} | {item['F1_Score']:.4f} | "
                         f"{item['Sample_Count']} |")
        
        report.append("")
        
        # 找出表现最好和最差的类别
        best_class = max(class_analysis['class_metrics'], key=lambda x: x['F1_Score'])
        worst_class = min(class_analysis['class_metrics'], key=lambda x: x['F1_Score'])
        
        report.append(f"- **表现最好的类别**: {best_class['Class']} (F1: {best_class['F1_Score']:.4f})")
        report.append(f"- **表现最差的类别**: {worst_class['Class']} (F1: {worst_class['F1_Score']:.4f})")
        report.append("")
    
    # 模型架构分析
    report.append("## 模型架构分析")
    report.append("")
    report.append("### CNN系列模型")
    report.append("- **CNN_Baseline**: 经典1D卷积神经网络")
    report.append("- **ResNet_Baseline**: 使用残差连接的深度CNN")
    report.append("- **DenseNet_Baseline**: 使用密集连接的CNN")
    report.append("- **Inception_Baseline**: 多尺度卷积的CNN")
    report.append("- **EfficientNet_Baseline**: 深度可分离卷积的CNN")
    report.append("")
    
    report.append("### 序列模型")
    report.append("- **RNN_LSTM_Baseline**: 长短期记忆网络")
    report.append("- **RNN_GRU_Baseline**: 门控循环单元")
    report.append("")
    
    report.append("### 注意力机制模型")
    report.append("- **VisionTransformer_Baseline**: 基于自注意力的Transformer")
    report.append("")
    
    report.append("### 自定义混合模型")
    report.append("- **Original_Model**: 原始CNN-CC-BiMamba模型")
    report.append("- **V2_Improved**: 改进版本2 (Focal Loss + Label Smoothing)")
    report.append("- **V3_Improved**: 改进版本3 (Asymmetric Loss + CutMix)")
    report.append("- **Final_Optimized**: 最终优化版本")
    report.append("")
    
    # 结论和建议
    report.append("## 结论和建议")
    report.append("")
    report.append("### 主要结论")
    report.append("1. **架构影响**: 不同神经网络架构在ECG分类任务上表现差异显著")
    report.append("2. **类别不平衡**: HYP类别仍然是分类难点，需要进一步优化")
    report.append("3. **模型复杂度**: 更复杂的模型不一定带来更好的性能")
    report.append("")
    
    report.append("### 改进建议")
    report.append("1. **数据增强**: 针对少数类别使用更多数据增强技术")
    report.append("2. **损失函数**: 尝试更多针对类别不平衡的损失函数")
    report.append("3. **集成学习**: 考虑多个模型的集成来提高整体性能")
    report.append("4. **特征工程**: 结合传统特征和深度学习特征")
    report.append("")
    
    # 技术细节
    report.append("## 技术细节")
    report.append("")
    report.append("### 训练设置")
    report.append("- **优化器**: AdamW")
    report.append("- **学习率调度**: Cosine Annealing")
    report.append("- **早停策略**: 验证集F1分数无改善时停止")
    report.append("- **阈值优化**: 为每个类别寻找最优F1阈值")
    report.append("")
    
    report.append("### 评估指标")
    report.append("- **Macro F1**: 各类别F1分数的平均值")
    report.append("- **Micro F1**: 所有样本的总体F1分数")
    report.append("- **Weighted F1**: 按样本数量加权的F1分数")
    report.append("")
    
    return "\n".join(report)

def main():
    """主函数"""
    print("🚀 开始生成ECG分类实验报告")
    print("=" * 60)
    
    # 加载所有结果
    print("📊 正在加载所有模型结果...")
    results = load_all_results()
    
    if not results:
        print("❌ 没有找到任何模型结果，无法生成报告")
        return
    
    print(f"✅ 成功加载 {len(results)} 个模型的结果")
    
    # 创建性能总结
    print("📈 正在分析模型性能...")
    summary = create_performance_summary(results)
    
    # 创建类别性能分析
    print("🔍 正在分析各类别性能...")
    class_analysis = create_class_performance_analysis(results)
    
    # 创建可视化图表
    print("📊 正在生成可视化图表...")
    try:
        fig = create_visualizations(summary, class_analysis)
        viz_path = f"./results/ecg_performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化图表已保存到: {viz_path}")
    except Exception as e:
        print(f"⚠️ 可视化图表生成失败: {e}")
    
    # 生成实验报告
    print("📝 正在生成实验报告...")
    report = generate_report(summary, class_analysis, results)
    
    # 保存报告
    report_path = f"./results/ecg_experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 实验报告已保存到: {report_path}")
    
    # 输出简要总结
    print(f"\n📋 实验报告生成完成!")
    print(f"📊 分析了 {len(summary)} 个模型")
    print(f"🏆 最佳模型: {summary[0]['Model']} (Macro F1: {summary[0]['Macro_F1']:.4f})")
    
    if class_analysis:
        print(f"🔍 类别分析基于: {class_analysis['best_model']}")
    
    print(f"\n📁 输出文件:")
    print(f"   - 实验报告: {report_path}")
    if 'viz_path' in locals():
        print(f"   - 可视化图表: {viz_path}")
    
    print(f"\n✅ 实验报告生成任务完成!")

if __name__ == "__main__":
    main()


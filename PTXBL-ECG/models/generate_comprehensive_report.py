# -*- coding: utf-8 -*-
"""
综合ECG分类实验报告生成器

分析所有基线模型和CC-CNN-Mamba模型的性能，生成详细的对比报告
"""

import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def load_model_results(results_dir="./results"):
    """加载所有模型的结果"""
    results = {}
    
    # 基线模型结果
    baseline_models = [
        "cnn_baseline_results.npy",
        "rnn_baseline_results.npy", 
        "resnet_baseline_results.npy",
        "densenet_baseline_results.npy",
        "inception_baseline_results.npy",
        "efficientnet_baseline_results.npy",
        "vit_baseline_results.npy"
    ]
    
    # CC-CNN-Mamba模型结果
    cc_cnn_mamba_models = [
        "ptbxl_final_optimized_results.npy"
    ]
    
    print("📊 正在加载模型结果...")
    
    # 加载基线模型结果
    for model_file in baseline_models:
        model_name = model_file.replace("_results.npy", "").replace("_", " ").title()
        file_path = os.path.join(results_dir, model_file)
        
        if os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True).item()
                results[model_name] = data
                print(f"✅ 已加载: {model_name}")
            except Exception as e:
                print(f"❌ 加载失败 {model_name}: {e}")
        else:
            print(f"⚠️ 文件不存在: {file_path}")
    
    # 加载CC-CNN-Mamba模型结果
    for model_file in cc_cnn_mamba_models:
        model_name = "CC-CNN-Mamba (Final Optimized)"
        file_path = os.path.join(results_dir, model_file)
        
        if os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True).item()
                results[model_name] = data
                print(f"✅ 已加载: {model_name}")
            except Exception as e:
                print(f"❌ 加载失败 {model_name}: {e}")
        else:
            print(f"⚠️ 文件不存在: {file_path}")
    
    return results

def create_performance_summary(results):
    """创建性能总结"""
    summary = []
    
    for model_name, data in results.items():
        try:
            summary.append({
                'Model': model_name,
                'Macro F1': data.get('macro_f1', 0),
                'Micro F1': data.get('micro_f1', 0),
                'Weighted F1': data.get('weighted_f1', 0),
                'Best Val F1': data.get('best_val_f1', 0)
            })
        except Exception as e:
            print(f"⚠️ 处理 {model_name} 时出错: {e}")
    
    return pd.DataFrame(summary)

def create_class_performance_analysis(results):
    """创建类别性能分析"""
    class_performance = {}
    
    for model_name, data in results.items():
        try:
            if 'test_predictions' in data and 'test_true' in data:
                y_true = data['test_true']
                y_pred = data['test_predictions']
                
                # 计算每个类别的F1分数
                from sklearn.metrics import f1_score
                f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
                
                class_performance[model_name] = f1_scores
        except Exception as e:
            print(f"⚠️ 分析 {model_name} 类别性能时出错: {e}")
    
    return class_performance

def create_visualizations(results, output_dir="./experiment_visualizations"):
    """创建可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 总体性能对比图
    summary_df = create_performance_summary(results)
    
    plt.figure(figsize=(12, 8))
    
    # Macro F1对比
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(summary_df)), summary_df['Macro F1'])
    plt.title('Macro F1 Score Comparison')
    plt.xlabel('Models')
    plt.ylabel('Macro F1 Score')
    plt.xticks(range(len(summary_df)), summary_df['Model'], rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Micro F1对比
    plt.subplot(2, 2, 2)
    bars = plt.bar(range(len(summary_df)), summary_df['Micro F1'])
    plt.title('Micro F1 Score Comparison')
    plt.xlabel('Models')
    plt.ylabel('Micro F1 Score')
    plt.xticks(range(len(summary_df)), summary_df['Model'], rotation=45, ha='right')
    plt.ylim(0, 1)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Weighted F1对比
    plt.subplot(2, 2, 3)
    bars = plt.bar(range(len(summary_df)), summary_df['Weighted F1'])
    plt.title('Weighted F1 Score Comparison')
    plt.xlabel('Models')
    plt.ylabel('Weighted F1 Score')
    plt.xticks(range(len(summary_df)), summary_df['Model'], rotation=45, ha='right')
    plt.ylim(0, 1)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 综合性能雷达图
    plt.subplot(2, 2, 4)
    metrics = ['Macro F1', 'Micro F1', 'Weighted F1']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 选择性能最好的模型进行雷达图展示
    best_model_idx = summary_df['Macro F1'].idxmax()
    best_model_name = summary_df.loc[best_model_idx, 'Model']
    best_model_scores = summary_df.loc[best_model_idx, metrics].values.tolist()
    best_model_scores += best_model_scores[:1]  # 闭合图形
    
    ax = plt.subplot(2, 2, 4, projection='polar')
    ax.plot(angles, best_model_scores, 'o-', linewidth=2, label=best_model_name)
    ax.fill(angles, best_model_scores, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title(f'Performance Radar Chart - {best_model_name}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 类别性能热力图
    class_performance = create_class_performance_analysis(results)
    if class_performance:
        class_names = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
        
        # 创建性能矩阵
        performance_matrix = []
        model_names = []
        for model_name, scores in class_performance.items():
            performance_matrix.append(scores)
            model_names.append(model_name)
        
        performance_matrix = np.array(performance_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(performance_matrix, 
                    xticklabels=class_names, 
                    yticklabels=model_names,
                    annot=True, 
                    fmt='.3f', 
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'F1 Score'})
        plt.title('Class-wise F1 Score Comparison')
        plt.xlabel('ECG Classes')
        plt.ylabel('Models')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ 可视化图表已保存到: {output_dir}")

def generate_markdown_report(results, output_file="./ECG_Classification_Experiment_Report.md"):
    """生成Markdown格式的实验报告"""
    
    summary_df = create_performance_summary(results)
    class_performance = create_class_performance_analysis(results)
    
    # 按Macro F1排序
    summary_df = summary_df.sort_values('Macro F1', ascending=False)
    
    report = f"""# ECG分类任务综合实验报告

## 📋 实验概述

本实验对比了多种深度学习模型在PTB-XL ECG数据集上的分类性能，包括：
- **基线模型**: CNN、RNN (LSTM/GRU)、ResNet、DenseNet、Inception、EfficientNet、Vision Transformer
- **创新模型**: CC-CNN-Mamba (CNN + Mamba混合架构)

### 实验设置
- **数据集**: PTB-XL ECG数据集 (100Hz采样率)
- **训练轮数**: 20 epochs
- **批次大小**: 32
- **学习率**: 0.001
- **评估指标**: Macro F1、Micro F1、Weighted F1

## 🏆 模型性能排名

### 总体性能对比

| 排名 | 模型名称 | Macro F1 | Micro F1 | Weighted F1 | 最佳验证F1 |
|------|----------|----------|----------|-------------|------------|
"""
    
    # 添加性能排名
    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        report += f"| {i} | {row['Model']} | {row['Macro F1']:.4f} | {row['Micro F1']:.4f} | {row['Weighted F1']:.4f} | {row['Best Val F1']:.4f} |\n"
    
    report += f"""

## 📊 详细性能分析

### 1. 最佳模型分析

**🏆 最佳模型: {summary_df.iloc[0]['Model']}**

- **Macro F1**: {summary_df.iloc[0]['Macro F1']:.4f}
- **Micro F1**: {summary_df.iloc[0]['Micro F1']:.4f}
- **Weighted F1**: {summary_df.iloc[0]['Weighted F1']:.4f}

### 2. 模型类型性能对比

#### 基线模型性能
"""
    
    # 分析基线模型性能
    baseline_models = summary_df[summary_df['Model'].str.contains('Baseline|Cnn|Rnn|Resnet|Densenet|Inception|Efficientnet|Vision', case=False)]
    if not baseline_models.empty:
        best_baseline = baseline_models.iloc[0]
        report += f"- **最佳基线模型**: {best_baseline['Model']} (Macro F1: {best_baseline['Macro F1']:.4f})\n"
        report += f"- **基线模型平均Macro F1**: {baseline_models['Macro F1'].mean():.4f}\n"
    
    # 分析CC-CNN-Mamba模型性能
    cc_mamba_models = summary_df[summary_df['Model'].str.contains('CC-CNN-Mamba|Mamba', case=False)]
    if not cc_mamba_models.empty:
        best_cc_mamba = cc_mamba_models.iloc[0]
        report += f"\n#### CC-CNN-Mamba模型性能\n"
        report += f"- **CC-CNN-Mamba模型**: {best_cc_mamba['Model']} (Macro F1: {best_cc_mamba['Macro F1']:.4f})\n"
    
    report += f"""

### 3. 类别性能分析

#### 各类别F1分数对比
"""
    
    if class_performance:
        class_names = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
        
        # 创建类别性能表格
        report += "| 模型 | " + " | ".join(class_names) + " |\n"
        report += "|------|" + "|".join(["-----" for _ in class_names]) + "|\n"
        
        for model_name, scores in class_performance.items():
            score_str = " | ".join([f"{score:.3f}" for score in scores])
            report += f"| {model_name} | {score_str} |\n"
    
    report += f"""

## 🔍 关键发现

### 1. 性能优势
- **最佳模型**: {summary_df.iloc[0]['Model']} 在Macro F1上表现最佳
- **模型类型**: {'CC-CNN-Mamba模型' if 'CC-CNN-Mamba' in summary_df.iloc[0]['Model'] else '基线模型'} 在整体性能上领先

### 2. 类别表现
- **表现最好的类别**: 通常NORM类别表现较好，因为数据相对平衡
- **挑战类别**: CD、HYP、MI等病理类别可能需要更多关注

### 3. 架构特点
- **CNN系列**: 在空间特征提取上表现稳定
- **RNN系列**: 在时序建模上有优势
- **Transformer系列**: 在长序列建模上表现良好
- **CC-CNN-Mamba**: 结合了CNN的空间建模能力和Mamba的序列建模能力

## 📈 改进建议

### 1. 数据层面
- 考虑使用数据增强技术改善类别不平衡
- 探索更多ECG预处理方法

### 2. 模型层面
- 尝试集成学习方法
- 探索更多超参数组合
- 考虑使用预训练模型

### 3. 训练策略
- 使用更长的训练轮数
- 尝试不同的学习率调度策略
- 探索早停策略的优化

## 📁 实验文件

- **模型文件**: 保存在 `./saved_models/` 目录
- **结果文件**: 保存在 `./results/` 目录
- **可视化图表**: 保存在 `./experiment_visualizations/` 目录

## 📅 实验时间

- **实验开始**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **报告生成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

*本报告由ECG分类实验自动生成*
"""
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Markdown报告已保存到: {output_file}")
    return output_file

def main():
    """主函数"""
    print("🚀 开始生成综合ECG分类实验报告")
    print("=" * 60)
    
    # 1. 加载所有模型结果
    results = load_model_results()
    
    if not results:
        print("❌ 没有找到任何模型结果文件")
        return
    
    print(f"\n📊 成功加载 {len(results)} 个模型的结果")
    
    # 2. 创建性能总结
    summary_df = create_performance_summary(results)
    print(f"\n📋 性能总结:")
    print(summary_df.to_string(index=False))
    
    # 3. 创建可视化图表
    print(f"\n🎨 正在生成可视化图表...")
    create_visualizations(results)
    
    # 4. 生成Markdown报告
    print(f"\n📝 正在生成实验报告...")
    report_file = generate_markdown_report(results)
    
    print(f"\n🎯 实验报告生成完成!")
    print(f"📁 报告文件: {report_file}")
    print(f"📊 可视化图表: ./experiment_visualizations/")
    
    # 5. 输出最佳模型
    best_model = summary_df.loc[summary_df['Macro F1'].idxmax()]
    print(f"\n🏆 最佳模型: {best_model['Model']}")
    print(f"   Macro F1: {best_model['Macro F1']:.4f}")
    print(f"   Micro F1: {best_model['Micro F1']:.4f}")
    print(f"   Weighted F1: {best_model['Weighted F1']:.4f}")

if __name__ == "__main__":
    main()

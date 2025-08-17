# -*- coding: utf-8 -*-
"""
综合ECG分类实验脚本

运行所有基线模型和CC-CNN-Mamba模型，设置相同的epochs进行公平对比
"""

import os
import subprocess
import time
from datetime import datetime

def run_model_training(model_name, script_path, args):
    """运行单个模型训练"""
    print(f"\n{'='*80}")
    print(f"🚀 开始训练 {model_name}")
    print(f"{'='*80}")
    
    # 构建命令
    cmd = ["python3", script_path] + args
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行训练
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} 训练成功完成!")
            print(f"训练耗时: {duration/60:.1f} 分钟")
            return True, duration
        else:
            print(f"❌ {model_name} 训练失败!")
            print(f"错误输出: {result.stderr}")
            return False, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} 训练超时 (2小时)")
        return False, 7200
    except Exception as e:
        print(f"❌ {model_name} 训练异常: {e}")
        return False, 0

def main():
    """主函数"""
    print("🚀 综合ECG分类实验 - 基线模型 vs CC-CNN-Mamba模型")
    print("=" * 80)
    
    # 设置环境变量
    os.environ['TMPDIR'] = os.path.expanduser('~/tmp')
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)
    
    # 创建必要的目录
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # 统一的训练参数
    EPOCHS = 20  # 设置相同的epochs
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    
    # 所有模型配置
    models_config = {
        # 基线模型
        "CNN_Baseline": {
            "script": "models/baselinemodels/cnn_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE)]
        },
        "RNN_LSTM_Baseline": {
            "script": "models/baselinemodels/rnn_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE), "--rnn_type", "lstm"]
        },
        "RNN_GRU_Baseline": {
            "script": "models/baselinemodels/rnn_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE), "--rnn_type", "gru"]
        },
        "ResNet_Baseline": {
            "script": "models/baselinemodels/resnet_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE)]
        },
        "DenseNet_Baseline": {
            "script": "models/baselinemodels/densenet_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE)]
        },
        "Inception_Baseline": {
            "script": "models/baselinemodels/inception_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE)]
        },
        "EfficientNet_Baseline": {
            "script": "models/baselinemodels/efficientnet_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE)]
        },
        "VisionTransformer_Baseline": {
            "script": "models/baselinemodels/vit_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE), "--d_model", "128"]
        },
        
        # CC-CNN-Mamba模型系列
        "Original_CC_CNN_Mamba": {
            "script": "models/ptbxl_cnn_cc_biomamba.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE)]
        },
        "V2_Improved_CC_CNN_Mamba": {
            "script": "models/ptbxl_cnn_cc_biomamba_v2.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE)]
        },
        "V3_Improved_CC_CNN_Mamba": {
            "script": "models/ptbxl_cnn_cc_biomamba_v3.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE)]
        },
        "Final_Optimized_CC_CNN_Mamba": {
            "script": "models/ptbxl_cnn_cc_biomamba_final_optimized.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE)]
        }
    }
    
    # 训练结果记录
    training_results = {}
    total_start_time = time.time()
    
    print(f"📋 计划训练 {len(models_config)} 个模型")
    print(f"统一训练参数: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, Learning Rate={LEARNING_RATE}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 依次训练每个模型
    for model_name, config in models_config.items():
        print(f"\n📝 准备训练: {model_name}")
        print(f"脚本路径: {config['script']}")
        print(f"参数: {' '.join(config['args'])}")
        
        # 检查脚本是否存在
        if not os.path.exists(config['script']):
            print(f"❌ 脚本文件不存在: {config['script']}")
            training_results[model_name] = {
                "status": "failed",
                "error": "Script file not found",
                "duration": 0
            }
            continue
        
        # 运行训练
        success, duration = run_model_training(
            model_name, 
            config['script'], 
            config['args']
        )
        
        # 记录结果
        training_results[model_name] = {
            "status": "success" if success else "failed",
            "duration": duration,
            "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 等待一下再开始下一个模型
        if success:
            print(f"⏳ 等待30秒后开始下一个模型...")
            time.sleep(30)
    
    # 计算总耗时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 输出训练总结
    print(f"\n{'='*80}")
    print(f"🎯 所有模型训练完成!")
    print(f"{'='*80}")
    
    print(f"总耗时: {total_duration/3600:.1f} 小时")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 统计成功和失败的模型
    successful_models = [name for name, result in training_results.items() if result["status"] == "success"]
    failed_models = [name for name, result in training_results.items() if result["status"] == "failed"]
    
    print(f"\n📊 训练结果统计:")
    print(f"   ✅ 成功: {len(successful_models)}/{len(models_config)}")
    print(f"   ❌ 失败: {len(failed_models)}/{len(models_config)}")
    
    if successful_models:
        print(f"\n✅ 成功训练的模型:")
        for model_name in successful_models:
            result = training_results[model_name]
            print(f"   - {model_name}: {result['duration']/60:.1f} 分钟")
    
    if failed_models:
        print(f"\n❌ 训练失败的模型:")
        for model_name in failed_models:
            result = training_results[model_name]
            print(f"   - {model_name}: {result.get('error', 'Unknown error')}")
    
    # 下一步建议
    print(f"\n🎯 下一步建议:")
    print(f"   1. 检查所有模型的结果文件")
    print(f"   2. 运行性能对比分析脚本")
    print(f"   3. 生成综合实验报告")
    print(f"   4. 创建README文档记录实验结果")
    
    print(f"\n✅ 综合实验训练任务完成!")

if __name__ == "__main__":
    main()


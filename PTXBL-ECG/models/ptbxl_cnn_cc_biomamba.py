# -*- coding: utf-8 -*-
"""
使用 CNN-CC-BiMamba 混合模型训练 PTB-XL 数据集 (高级优化版)

功能:
1. 加载预处理好的 PTB-XL 数据集文件。
2. 🔥【优化V2】使用 Focal Loss 处理类别不平衡和困难样本问题。
3. 🔥【优化V2】在验证集上自动寻找最优预测阈值，并用于最终测试。
4. 包含了完整的训练、验证和评估流程。

依赖库:
pip install pandas numpy scikit-learn torch tqdm mamba-ssm --no-build-isolation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from tqdm import tqdm
import time

# 尝试导入优化的Mamba库
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
    print("✅ 成功导入 mamba_ssm 库")
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("❌ 未安装 mamba_ssm 库")
    print("💡 运行: pip install mamba-ssm --no-build-isolation")

# ==============================================================================
# 🔥【优化V2】Focal Loss 损失函数定义
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # 使用 pos_weight 来处理类别不平衡, 这是在多标签场景下更直接和标准的方法
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        p_t = torch.exp(-bce_loss)
        focal_loss = (1 - p_t)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==============================================================================
# 数据加载模块
# ==============================================================================
def load_ptbxl_data(data_file_path):
    print(f"正在加载PTB-XL数据: {data_file_path}")
    data = np.load(data_file_path, allow_pickle=True)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    
    print(f"✅ 原始数据形状:")
    print(f"   - 训练集: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"   - 验证集: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"   - 测试集: X_test {X_test.shape}, y_test {y_test.shape}")

    # 🔥【修复】添加安全检查，确保X和y的样本数量一致
    # 这可以防止因预处理脚本跳过损坏文件而导致的数据不匹配问题
    min_train = min(X_train.shape[0], y_train.shape[0])
    if X_train.shape[0] != y_train.shape[0]:
        print(f"⚠️ 警告: 训练集样本数不匹配！将截断至 {min_train}。")
        X_train, y_train = X_train[:min_train], y_train[:min_train]

    min_val = min(X_val.shape[0], y_val.shape[0])
    if X_val.shape[0] != y_val.shape[0]:
        print(f"⚠️ 警告: 验证集样本数不匹配！将截断至 {min_val}。")
        X_val, y_val = X_val[:min_val], y_val[:min_val]

    min_test = min(X_test.shape[0], y_test.shape[0])
    if X_test.shape[0] != y_test.shape[0]:
        print(f"⚠️ 警告: 测试集样本数不匹配！将截断至 {min_test}。")
        X_test, y_test = X_test[:min_test], y_test[:min_test]

    print(f"✅ 修正后数据形状:")
    print(f"   - 训练集: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"   - 验证集: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"   - 测试集: X_test {X_test.shape}, y_test {y_test.shape}")
    
    print(f"✅ 数据加载完成。")
    return X_train, y_train, X_val, y_val, X_test, y_test, classes

# ==============================================================================
# 核心模型定义 (与之前版本一致)
# ==============================================================================
class BidirectionalMambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        if not MAMBA_SSM_AVAILABLE: raise ImportError("mamba-ssm is not available.")
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.fusion = nn.Sequential(nn.Linear(d_model * 2, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model), nn.Dropout(0.1))
    def forward(self, x):
        forward_out = self.forward_mamba(x)
        x_reversed = torch.flip(x, dims=[1])
        backward_out = self.backward_mamba(x_reversed)
        backward_out = torch.flip(backward_out, dims=[1])
        return self.fusion(torch.cat([forward_out, backward_out], dim=-1))

class ContextClustering1DLayer(nn.Module):
    def __init__(self, d_model, window_size=15, n_clusters=4, dropout=0.1):
        super().__init__()
        self.d_model, self.window_size, self.n_clusters = d_model, window_size, n_clusters
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, d_model))
        self.similarity_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, L, D = x.shape
        x_permuted = x.permute(0, 2, 1)
        pad_left, pad_right = (self.window_size - 1) // 2, self.window_size // 2
        x_padded = F.pad(x_permuted, (pad_left, pad_right))
        patches = x_padded.unfold(2, self.window_size, 1).permute(0, 2, 3, 1)
        patches_proj = self.similarity_projection(patches)
        centers_proj = self.similarity_projection(self.cluster_centers)
        
        # 计算 patches 和聚类中心的相似度
        # 维度: (B, L, W, 1, D) vs (1, 1, 1, C, D) -> (B, L, W, C)
        similarity = F.cosine_similarity(patches_proj.unsqueeze(3), centers_proj.view(1, 1, 1, self.n_clusters, D), dim=-1)
        
        # 将相似度转换为权重
        assignment_weights = F.softmax(similarity * 10, dim=-1) # (B, L, W, C)
        
        value_patches = self.value_projection(patches)
        
        # 根据分配权重聚合 patch 的值
        # einsum: (B, L, W, C), (B, L, W, D) -> (B, L, C, D)
        aggregated_features = torch.einsum('blwc,blwd->blcd', assignment_weights, value_patches)
        
        # 归一化聚合后的特征
        weight_sums = assignment_weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
        aggregated_features = aggregated_features / weight_sums.transpose(-1, -2)
        
        # 提取每个窗口中心点的权重
        center_weights = assignment_weights[:, :, self.window_size // 2, :] # (B, L, C)
        
        # 计算激活门
        activated_weights = torch.sigmoid(self.alpha * center_weights + self.beta)
        
        # 将聚合的特征与激活门结合
        # einsum: (B, L, C, D), (B, L, C) -> (B, L, D)
        output = torch.einsum('blcd,blc->bld', aggregated_features, activated_weights)
        
        return self.dropout(self.norm(x + output))

class CCBiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dropout=0.1, window_size=15, n_clusters=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mamba_path = BidirectionalMambaLayer(d_model, d_state, expand)
        self.cc_path = ContextClustering1DLayer(d_model, window_size, n_clusters, dropout)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, 2), nn.Softmax(dim=-1))
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        self.gate_ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
    def forward(self, x):
        x_norm = self.norm1(x)
        mamba_out, cc_out = self.mamba_path(x_norm), self.cc_path(x_norm)
        gate_weights = self.gate(torch.cat([mamba_out, cc_out], dim=-1))
        fused_out = gate_weights[..., 0].unsqueeze(-1) * mamba_out + gate_weights[..., 1].unsqueeze(-1) * cc_out
        x = x + fused_out
        ffn_out = self.ffn(self.norm2(x))
        gate_weights_ffn = self.gate_ffn(x)
        return x + gate_weights_ffn * ffn_out

class ECGCNNExtractor(nn.Module):
    def __init__(self, input_channels=12, d_model=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 15, 1, 7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(3, 2, 1)
        self.conv2 = nn.Conv1d(32, 64, 11, 1, 5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(3, 2, 1)
        self.conv3 = nn.Conv1d(64, d_model, 7, 1, 3)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.feature_adapter = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Dropout(0.1), nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.activation(self.bn3(self.conv3(x)))
        return self.feature_adapter(x.transpose(1, 2))

class CNNCCBiMambaClassifier(nn.Module):
    def __init__(self, input_leads=12, num_classes=5, d_model=64, n_mamba_layers=4, d_state=16, expand=2, dropout=0.1, window_size=15, n_clusters=4):
        super().__init__()
        self.cnn_extractor = ECGCNNExtractor(input_channels=input_leads, d_model=d_model)
        self.cc_mamba_layers = nn.ModuleList([CCBiMambaBlock(d_model, d_state, expand, dropout, window_size=window_size if i % 2 == 0 else window_size // 2 + 1, n_clusters=n_clusters) for i in range(n_mamba_layers)])
        self.cross_layer_fusion = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_att_pool = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.SiLU(), nn.Linear(d_model // 4, 1), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, d_model * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes))
    def forward(self, x):
        cnn_features = self.cnn_extractor(x)
        cc_mamba_features = cnn_features
        for layer in self.cc_mamba_layers:
            cc_mamba_features = layer(cc_mamba_features)
        cnn_pooled = self.global_avg_pool(cnn_features.transpose(1, 2)).squeeze(2)
        cc_mamba_pooled = self.global_avg_pool(cc_mamba_features.transpose(1, 2)).squeeze(2)
        fused_features = self.cross_layer_fusion(torch.cat([cnn_pooled, cc_mamba_pooled], dim=1))
        final_features = fused_features.unsqueeze(1).expand(-1, cc_mamba_features.size(1), -1) + cc_mamba_features
        avg_pooled = self.global_avg_pool(final_features.transpose(1, 2)).squeeze(2)
        max_pooled = self.global_max_pool(final_features.transpose(1, 2)).squeeze(2)
        att_weights = self.global_att_pool(final_features)
        att_pooled = (final_features * att_weights).sum(dim=1)
        global_features = torch.cat([avg_pooled, max_pooled, att_pooled], dim=1)
        return self.classifier(global_features)

# ==============================================================================
# 🔥【优化V2】寻找最优阈值函数
# ==============================================================================
def find_optimal_thresholds(model, val_loader, device):
    model.eval()
    val_probs = []
    val_true = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output)
            val_probs.append(probs.cpu().numpy())
            val_true.append(target.cpu().numpy())
    
    val_probs = np.concatenate(val_probs)
    val_true = np.concatenate(val_true)
    
    num_classes = val_true.shape[1]
    optimal_thresholds = np.zeros(num_classes)
    
    print("\n🔥 正在为每个类别寻找最优F1阈值...")
    for i in range(num_classes):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.01):
            preds = (val_probs[:, i] > thresh).astype(int)
            f1 = f1_score(val_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds[i] = best_thresh
    
    print(f"✅ 最优阈值查找完成: {optimal_thresholds}")
    return torch.tensor(optimal_thresholds).to(device)

# ==============================================================================
# 训练与评估主流程
# ==============================================================================
def main(args):
    # 1. 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_ptbxl_data(args.data_file)
    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

    # 2. 创建 DataLoader
    train_ds, val_ds, test_ds = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val), TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. 初始化模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNCCBiMambaClassifier(num_classes=len(classes), d_model=args.d_model).to(device)

    # 计算 pos_weight 用于 Focal Loss, 有效处理类别不平衡
    pos_counts = y_train.sum(dim=0)
    pos_weight = (y_train.shape[0] - pos_counts) / (pos_counts + 1e-8)
    
    criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. 训练循环
    print("\nStep 4: 开始训练高级优化版模型")
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        # 验证
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = (torch.sigmoid(output) > 0.5).float() # 使用0.5作为验证阶段的临时阈值
                val_preds.append(preds.cpu().numpy())
                val_true.append(target.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}, Val F1-macro (at 0.5 thresh): {val_f1:.4f}")
        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"🚀 新的最佳模型已保存，F1-macro: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"早停: {args.patience} 个epoch无改善")
            break

    # 5. 测试
    print("\nStep 5: 在测试集上评估最佳模型")
    model.load_state_dict(torch.load(args.save_path))
    
    # 🔥【优化V2】使用验证集寻找最优阈值
    optimal_thresholds = find_optimal_thresholds(model, val_loader, device)
    
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="测试中"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 🔥【优化V2】使用找到的最优阈值进行预测
            preds = (torch.sigmoid(output) > optimal_thresholds).float()
            test_preds.append(preds.cpu().numpy())
            test_true.append(target.cpu().numpy())
            
    test_preds = np.concatenate(test_preds)
    test_true = np.concatenate(test_true)
    print("\n--- 详细分类报告 (测试集 @ 最优阈值) ---")
    print(classification_report(test_true, test_preds, target_names=classes, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用高级优化版CNN-CC-BiMamba训练PTB-XL数据集')
    parser.add_argument('--data_file', type=str, default='/home/elonge/WorkSpace/ECG_Project/processed_data/ptbxl_processed_100hz.npz', help='预处理好的PTB-XL数据文件路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练周期数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--patience', type=int, default=30, help='早停的耐心值')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss的gamma参数')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--n_mamba_layers', type=int, default=6, help='Mamba层数')
    parser.add_argument('--save_path', type=str, default='./saved_models/ptbxl_advanced_optimized_model.pth', help='模型保存路径')
    args = parser.parse_args()
    main(args)

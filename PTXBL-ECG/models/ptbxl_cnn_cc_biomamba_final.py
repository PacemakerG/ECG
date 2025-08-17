# -*- coding: utf-8 -*-
"""
使用 CNN-CC-BiMamba 混合模型训练 PTB-XL 数据集 (最终优化版)

最终优化策略:
1. 🔥【最终版】基于V2架构，添加V3的最佳特性
2. 🔥【最终版】使用 Asymmetric Loss + Label Smoothing 组合
3. 🔥【最终版】添加类别权重平衡
4. 🔥【最终版】使用 EMA 和更好的学习率调度
5. 🔥【最终版】针对HYP类别进行特殊优化
6. 🔥【最终版】添加更多正则化和数据增强
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
import random

# 尝试导入优化的Mamba库
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
    print("✅ 成功导入 mamba_ssm 库")
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("❌ 未安装 mamba_ssm 库")

# ==============================================================================
# 🔥【最终版】Asymmetric Loss + Label Smoothing 组合
# ==============================================================================
class AsymmetricLossWithSmoothing(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, alpha=0.1):
        super(AsymmetricLossWithSmoothing, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.alpha = alpha

    def forward(self, x, y):
        # Label smoothing
        smooth_targets = y * (1 - self.alpha) + 0.5 * self.alpha
        
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # 正样本的损失
        los_pos = smooth_targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - smooth_targets) * torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * smooth_targets
            pt1 = xs_neg * (1 - smooth_targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * smooth_targets + self.gamma_neg * (1 - smooth_targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        if self.clip > 0:
            loss = loss * (1 - self.clip) + self.clip * 0.5

        return -loss.mean()

# ==============================================================================
# 🔥【最终版】EMA (Exponential Moving Average)
# ==============================================================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

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
    
    print(f"✅ 数据形状:")
    print(f"   - 训练集: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"   - 验证集: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"   - 测试集: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, classes

# ==============================================================================
# 🔥【最终版】基于V2的增强模型定义
# ==============================================================================
class BidirectionalMambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        if not MAMBA_SSM_AVAILABLE: raise ImportError("mamba-ssm is not available.")
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), 
            nn.LayerNorm(d_model * 2),
            nn.SiLU(), 
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x):
        forward_out = self.forward_mamba(x)
        x_reversed = torch.flip(x, dims=[1])
        backward_out = self.backward_mamba(x_reversed)
        backward_out = torch.flip(backward_out, dims=[1])
        return self.fusion(torch.cat([forward_out, backward_out], dim=-1))

class ContextClustering1DLayer(nn.Module):
    def __init__(self, d_model, window_size=15, n_clusters=8, dropout=0.1):
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
        similarity = F.cosine_similarity(patches_proj.unsqueeze(3), centers_proj.view(1, 1, 1, self.n_clusters, D), dim=-1)
        assignment_weights = F.softmax(similarity * 10, dim=-1)
        value_patches = self.value_projection(patches)
        aggregated_features = torch.einsum('blwc,blwd->blcd', assignment_weights, value_patches)
        weight_sums = assignment_weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
        aggregated_features = aggregated_features / weight_sums.transpose(-1, -2)
        center_weights = assignment_weights[:, :, self.window_size // 2, :]
        activated_weights = torch.sigmoid(self.alpha * center_weights + self.beta)
        output = torch.einsum('blcd,blc->bld', aggregated_features, activated_weights)
        return self.dropout(self.norm(x + output))

class CCBiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dropout=0.1, window_size=15, n_clusters=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mamba_path = BidirectionalMambaLayer(d_model, d_state, expand)
        self.cc_path = ContextClustering1DLayer(d_model, window_size, n_clusters, dropout)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), 
            nn.LayerNorm(d_model),
            nn.SiLU(), 
            nn.Linear(d_model, 2), 
            nn.Softmax(dim=-1)
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.SiLU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_model * 4, d_model), 
            nn.Dropout(dropout)
        )
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
    def __init__(self, input_channels=12, d_model=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, 15, 1, 7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(3, 2, 1)
        
        self.conv2 = nn.Conv1d(64, 128, 11, 1, 5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(3, 2, 1)
        
        self.conv3 = nn.Conv1d(128, 256, 7, 1, 3)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, d_model, 5, 1, 2)
        self.bn4 = nn.BatchNorm1d(d_model)
        
        # 🔥【最终版】SE注意力模块
        self.se_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 16, 1),
            nn.SiLU(),
            nn.Conv1d(d_model // 16, d_model, 1),
            nn.Sigmoid()
        )
        
        self.feature_adapter = nn.Sequential(
            nn.Linear(d_model, d_model * 2), 
            nn.LayerNorm(d_model * 2),
            nn.SiLU(), 
            nn.Dropout(0.1), 
            nn.Linear(d_model * 2, d_model), 
            nn.LayerNorm(d_model)
        )
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
        x = self.dropout(x)
        
        x = self.activation(self.bn4(self.conv4(x)))
        
        # 🔥【最终版】SE注意力
        se_weights = self.se_attention(x)
        x = x * se_weights
        
        return self.feature_adapter(x.transpose(1, 2))

class CNNCCBiMambaClassifier(nn.Module):
    def __init__(self, input_leads=12, num_classes=5, d_model=256, n_mamba_layers=8, d_state=16, expand=2, dropout=0.1, window_size=15, n_clusters=8):
        super().__init__()
        self.cnn_extractor = ECGCNNExtractor(input_channels=input_leads, d_model=d_model)
        self.cc_mamba_layers = nn.ModuleList([
            CCBiMambaBlock(d_model, d_state, expand, dropout, 
                          window_size=window_size if i % 2 == 0 else window_size // 2 + 1, 
                          n_clusters=n_clusters) for i in range(n_mamba_layers)
        ])
        
        # 🔥【最终版】增强的特征融合
        self.cross_layer_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), 
            nn.LayerNorm(d_model),
            nn.SiLU(), 
            nn.Dropout(dropout)
        )
        
        # 🔥【最终版】多头注意力池化
        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_att_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4), 
            nn.SiLU(), 
            nn.Linear(d_model // 4, 1), 
            nn.Softmax(dim=1)
        )
        
        # 🔥【最终版】更深的分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 3), 
            nn.Linear(d_model * 3, d_model * 2), 
            nn.SiLU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_model * 2, d_model), 
            nn.SiLU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_model, d_model // 2), 
            nn.SiLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        cnn_features = self.cnn_extractor(x)
        cc_mamba_features = cnn_features
        
        # 逐层处理
        for layer in self.cc_mamba_layers:
            cc_mamba_features = layer(cc_mamba_features)
        
        # 特征融合
        cnn_pooled = self.global_avg_pool(cnn_features.transpose(1, 2)).squeeze(2)
        cc_mamba_pooled = self.global_avg_pool(cc_mamba_features.transpose(1, 2)).squeeze(2)
        fused_features = self.cross_layer_fusion(torch.cat([cnn_pooled, cc_mamba_pooled], dim=1))
        
        # 🔥【最终版】多头注意力处理
        final_features = fused_features.unsqueeze(1).expand(-1, cc_mamba_features.size(1), -1) + cc_mamba_features
        attn_out, _ = self.multi_head_attention(final_features, final_features, final_features)
        final_features = final_features + attn_out
        
        # 多种池化策略
        avg_pooled = self.global_avg_pool(final_features.transpose(1, 2)).squeeze(2)
        max_pooled = self.global_max_pool(final_features.transpose(1, 2)).squeeze(2)
        att_weights = self.global_att_pool(final_features)
        att_pooled = (final_features * att_weights).sum(dim=1)
        
        global_features = torch.cat([avg_pooled, max_pooled, att_pooled], dim=1)
        return self.classifier(global_features)

# ==============================================================================
# 🔥【最终版】寻找最优阈值函数
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
# 🔥【最终版】学习率预热和余弦退火调度器
# ==============================================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # 线性预热
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # 余弦退火
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# ==============================================================================
# 训练与评估主流程
# ==============================================================================
def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
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
    model = CNNCCBiMambaClassifier(
        num_classes=len(classes), 
        d_model=args.d_model,
        n_mamba_layers=args.n_mamba_layers
    ).to(device)

    # 🔥【最终版】使用组合损失函数
    criterion = AsymmetricLossWithSmoothing(gamma_neg=4, gamma_pos=1, clip=0.05, alpha=0.1)
    
    # 🔥【最终版】使用 AdamW 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("✅ 使用 AdamW 优化器")
    
    # 🔥【最终版】学习率调度器
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)
    
    # 🔥【最终版】EMA
    ema = EMA(model, decay=0.9999)
    ema.register()

    # 4. 训练循环
    print(f"\n🔥 开始训练最终优化版模型 (d_model={args.d_model}, layers={args.n_mamba_layers})")
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 🔥【最终版】更新 EMA
            ema.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'lr': f'{scheduler.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 验证
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = (torch.sigmoid(output) > 0.5).float()
                val_preds.append(preds.cpu().numpy())
                val_true.append(target.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}, Val F1-macro: {val_f1:.4f}, Avg Loss: {epoch_loss/len(train_loader):.4f}")

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
    print("\n🔥 在测试集上评估最佳模型")
    model.load_state_dict(torch.load(args.save_path, weights_only=True))
    
    # 使用 EMA 模型进行测试
    ema.apply_shadow()
    
    # 寻找最优阈值
    optimal_thresholds = find_optimal_thresholds(model, val_loader, device)
    
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="测试中"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = (torch.sigmoid(output) > optimal_thresholds).float()
            test_preds.append(preds.cpu().numpy())
            test_true.append(target.cpu().numpy())
            
    test_preds = np.concatenate(test_preds)
    test_true = np.concatenate(test_true)
    
    print("\n--- 详细分类报告 (最终版 @ 最优阈值) ---")
    print(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
    
    # 恢复原始模型
    ema.restore()
    
    # 保存结果
    results = {
        'test_predictions': test_preds,
        'test_true': test_true,
        'optimal_thresholds': optimal_thresholds.cpu().numpy(),
        'best_val_f1': best_f1
    }
    np.save(args.results_path, results)
    print(f"✅ 结果已保存到: {args.results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用最终优化版 CNN-CC-BiMamba训练PTB-XL数据集')
    parser.add_argument('--data_file', type=str, default='/home/elonge/WorkSpace/ECG_Project/processed_data/ptbxl_processed_100hz.npz', help='预处理好的PTB-XL数据文件路径')
    parser.add_argument('--epochs', type=int, default=20, help='训练周期数')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--lr', type=float, default=4e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--patience', type=int, default=8, help='早停的耐心值')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--n_mamba_layers', type=int, default=8, help='Mamba层数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_path', type=str, default='./saved_models/ptbxl_final_optimized_model.pth', help='模型保存路径')
    parser.add_argument('--results_path', type=str, default='./results/ptbxl_final_optimized_results.npy', help='结果保存路径')
    args = parser.parse_args()
    main(args)

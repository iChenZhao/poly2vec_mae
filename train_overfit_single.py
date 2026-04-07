import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import matplotlib.path as mpltPath

# 导入你项目中的模块 (请确保路径正确)
from src.loader_ocf import OCFDataset
from src.decoder import AnalyticallyGuidedDecoder

def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def calculate_iou(pred_bin, gt_bin):
    """计算二值化掩码之间的 IoU"""
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return intersection / union

def visualize_prediction(model, sample_data, epoch, device, sample_idx=0):
    """
    按照三联图布局进行可视化输出：
    1. Prob Field (热力图) | 2. Model Result (IoU 二值图) | 3. GT Label (真值图)
    """
    model.eval()
    res = 256
    
    # 1. 生成评估网格坐标 [-1, 1]
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    grid_x, grid_y = np.meshgrid(x, y, indexing='xy')
    # 平铺成 [1, 256*256, 2] 的张量送入模型
    eval_coords = torch.from_numpy(np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)).float().unsqueeze(0).to(device)
    
    # 2. 提取特征并推理预测
    embedding = sample_data['embedding'].unsqueeze(0).to(device)
    freq_real = sample_data['freq_real'].unsqueeze(0).to(device)
    freq_imag = sample_data['freq_imag'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(eval_coords, freq_real, freq_imag, embedding)
        probs = torch.sigmoid(logits).view(res, res).cpu().numpy()
    
    # 3. 生成 Ground Truth 的二值掩码图像 (用于计算 IoU 和右侧展示)
    # 利用三角形数据判断网格点是否在内部
    gt_mask = np.zeros((res, res), dtype=bool)
    eval_points_flat = eval_coords.squeeze(0).cpu().numpy()
    
    # 注意：这里我们使用 sample_data 原始的多边形信息
    # 假设 sample_data 包含原始 triangles 或者可以直接从 dataset cache 获取
    # 为了演示，我们遍历三角形
    tris = sample_data['triangles'] # 确保 dataset[idx] 返回时带上了这个字段
    for tri in tris:
        path = mpltPath.Path(tri)
        mask = path.contains_points(eval_points_flat).reshape(res, res)
        gt_mask = np.logical_or(gt_mask, mask)
    
    # 4. 计算 IoU
    pred_binary = (probs > 0.5)
    iou_val = calculate_iou(pred_binary, gt_mask)
    
    # 5. 绘图布局 (15x5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- 左图：Prob Field ---
    im0 = axes[0].imshow(probs, origin='lower', extent=[-1, 1, -1, 1], cmap='magma')
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"Prob Field (Idx: {sample_idx})")
    axes[0].set_xticks([-1, 0, 1])
    axes[0].set_yticks([-1, 0, 1])

    # --- 中图：Model Result ---
    axes[1].imshow(pred_binary, origin='lower', extent=[-1, 1, -1, 1], cmap='gray')
    axes[1].set_title(f"Model Result (IoU: {iou_val:.4f})")
    axes[1].set_xticks([-1, 0, 1])
    axes[1].set_yticks([-1, 0, 1])

    # --- 右图：GT Label ---
    axes[2].imshow(gt_mask, origin='lower', extent=[-1, 1, -1, 1], cmap='gray')
    axes[2].set_title("GT Label (Force Norm)")
    axes[2].set_xticks([-1, 0, 1])
    axes[2].set_yticks([-1, 0, 1])

    plt.tight_layout()
    save_path = f"overfit_vis_epoch_{epoch}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"📊 已保存对比图至: {save_path} | 当前 IoU: {iou_val:.4f}")

def overfit_single_sample():
    print("🚀 启动单样本过拟合 (Sanity Check) 训练...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")

    # 1. 加载配置和数据 (使用清洗后的数据集)
    cfg = load_config("configs/recons.yaml") # 请确保你的 config 路径正确
    
    # 强制修改配置，确保不出错
    cfg['paths']['data_path'] = "data/data_100.pt" # 指向清洗后的纯净数据
    
    dataset = OCFDataset(cfg)
    
    # 核心：我们只取第 0 个样本进行死磕！
    # 因为 Dataset 中有动态采样，每次 getitem 取出的 query 点都是随机变动的，
    # 这能完美测试模型是否真正学到了解析场，而不是死记硬背网格。
    TARGET_SAMPLE_IDX = 0 
    
    # 2. 初始化修正后的模型
    model = AnalyticallyGuidedDecoder(
        embedding_dim=384, 
        H=63, 
        W=32, 
        use_embedding=True
    ).to(device)
    
    # 3. 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 不使用 reduction，因为我们需要乘以 Dataloader 传来的动态权重
    criterion = nn.BCEWithLogitsLoss(reduction='none') 
    
    epochs = 2000
    print(f"\n⚡ 开始训练，目标 Epoch: {epochs}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 每次从 Dataset 获取第 0 个样本 (附带全新的随机采样点)
        sample = dataset[TARGET_SAMPLE_IDX]
        
        # 升维增加 Batch 维度 (B=1)，并移动到 GPU
        coords = sample['coords'].unsqueeze(0).to(device)       # [1, N, 2]
        labels = sample['labels'].unsqueeze(0).to(device)       # [1, N, 1]
        weights = sample['weights'].unsqueeze(0).to(device)     # [1, N, 1]
        embedding = sample['embedding'].unsqueeze(0).to(device) # [1, 384]
        freq_real = sample['freq_real'].unsqueeze(0).to(device) # [1, H, W]
        freq_imag = sample['freq_imag'].unsqueeze(0).to(device) # [1, H, W]
        
        # 前向传播
        logits = model(coords, freq_real, freq_imag, embedding)
        
        # 计算加权 BCE Loss
        bce_loss = criterion(logits, labels)
        loss = (bce_loss * weights).mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 打印日志
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch [{epoch:4d}/{epochs}] | Loss: {loss.item():.6f}")
            
        # 阶段性可视化渲染结果 (例如在 1, 200, 500, 1000 轮)
        if epoch in [1, 200, 500, 1000, 1300, 1600, 2000]:
            visualize_prediction(model, sample, epoch, device)
            
    print("\n🎉 单样本过拟合测试完成！请查看生成的图片 (overfit_vis_epoch_*.png)。")
    print("👉 重点观察：红色轮廓线是否完美贴合黄色区域，且没有任何模糊/圆滑？如果完美，即可开启全量训练！")

if __name__ == "__main__":
    overfit_single_sample()
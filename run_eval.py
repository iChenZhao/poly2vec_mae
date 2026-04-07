import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from tqdm import tqdm

# 环境补丁
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "./"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.decoder_v1 import AnalyticallyGuidedDecoder
from src.loader_ocf import OCFDataset

def calculate_iou(pred_bin, gt_bin):
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return intersection / (union + 1e-7)

def visualize_triple(probs, pred_binary, gt_mask, iou, sample_idx, save_path):
    """ 按照要求的 1x3 模式进行可视化 """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Prob Field (Magma 色图)
    im0 = axes[0].imshow(probs, origin='lower', extent=[-1, 1, -1, 1], cmap='magma')
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"Prob Field (Idx: {sample_idx})")
    
    # 2. Model Result (二值化结果)
    axes[1].imshow(pred_binary, origin='lower', extent=[-1, 1, -1, 1], cmap='gray')
    axes[1].set_title(f"Model Result (IoU: {iou:.4f})")
    
    # 3. GT Label (真值)
    axes[2].imshow(gt_mask, origin='lower', extent=[-1, 1, -1, 1], cmap='gray')
    axes[2].set_title("GT Label (Force Norm)")
    
    for ax in axes:
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_testing():
    print("🚀 启动 Freq-OCF 模型全面评估...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载配置
    config_path = "configs/recons.yaml" # 确保路径正确
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 2. 加载数据集与测试集索引
    dataset = OCFDataset(cfg)
    indices_path = os.path.join(cfg['paths']['save_dir'], "dataset_split_indices.pt")
    if not os.path.exists(indices_path):
        print(f"❌ 找不到索引文件 {indices_path}，请确保已经运行过训练脚本。")
        return
    
    split_info = torch.load(indices_path)
    test_indices = split_info.get('test', [])
    print(f"✅ 成功加载测试集，包含 {len(test_indices)} 个样本。")

    # 3. 初始化并加载模型
    model = AnalyticallyGuidedDecoder(
        embedding_dim=cfg['model']['embedding_dim'],
        H=cfg['model']['freq_H'],
        W=cfg['model']['freq_W'],
        hidden_dims=cfg['model']['hidden_dims']
    ).to(device)
    
    ckpt_path = os.path.join(cfg['paths']['save_dir'], "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 找不到 best_model.pth，尝试加载最后一个保存的权重...")
        ckpt_path = os.path.join(cfg['paths']['save_dir'], f"model_ep{cfg['training']['num_epochs']:03d}.pth")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✅ 模型加载成功: {ckpt_path}")

    # 4. 创建结果保存目录
    test_res_dir = os.path.join(cfg['paths']['save_dir'], "test_results")
    os.makedirs(test_res_dir, exist_ok=True)

    # 5. 开始批量测试与可视化
    all_ious = []
    res = 256 # 可视化分辨率
    
    # 生成密集的评估网格
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    grid_x, grid_y = np.meshgrid(x, y, indexing='xy')
    eval_points_flat = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
    eval_coords_torch = torch.from_numpy(eval_points_flat).float().unsqueeze(0).to(device)

    # 仅对测试集的前 20 个样本进行三联图输出，其余只计分 (防止生成数千张图占硬盘)
    VIS_LIMIT = 3 

    for i, idx in enumerate(tqdm(test_indices, desc="Evaluating")):
        # 从 Dataset 获取数据张量
        sample = dataset[idx]
        
        # 准备推理特征
        embedding = sample['embedding'].unsqueeze(0).to(device)
        freq_real = sample['freq_real'].unsqueeze(0).to(device)
        freq_imag = sample['freq_imag'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(eval_coords_torch, freq_real, freq_imag, embedding)
            probs = torch.sigmoid(logits).view(res, res).cpu().numpy()
        
        # 获取 GT 掩码 (直接访问 dataset.data 绕过 triangles 注释问题)
        gt_mask = np.zeros((res, res), dtype=bool)
        raw_sample = dataset.data[idx] # 访问底层原始列表
        tris = raw_sample['triangles']
        for tri in tris:
            path = mpltPath.Path(tri)
            mask = path.contains_points(eval_points_flat).reshape(res, res)
            gt_mask = np.logical_or(gt_mask, mask)
        
        # 计算 IoU
        pred_binary = (probs > 0.5)
        iou_val = calculate_iou(pred_binary, gt_mask)
        all_ious.append(iou_val)

        # 阶段性保存可视化图像
        if i < VIS_LIMIT:
            save_name = os.path.join(test_res_dir, f"test_sample_{idx:05d}.png")
            visualize_triple(probs, pred_binary, gt_mask, iou_val, idx, save_name)

    # 6. 输出最终统计报告
    mean_iou = np.mean(all_ious)
    median_iou = np.median(all_ious)
    print("\n" + "="*50)
    print("📊 测试集最终评估报告")
    print("="*50)
    print(f"📈 Mean IoU:   {mean_iou:.4f}")
    print(f"📈 Median IoU: {median_iou:.4f}")
    print(f"🖼️ 可视化图已存至: {test_res_dir}")
    print("="*50)

if __name__ == "__main__":
    run_testing()
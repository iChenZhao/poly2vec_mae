import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.loader_ocf import OCFDataset

def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def debug_dataloader():
    print("🚀 开始 DataLoader 专项 Debug 测试...\n")

    # 1. 加载配置
    config_path = "configs/recons.yaml"
    try:
        cfg = load_config(config_path)
        print(f"✅ 成功加载配置: {config_path}")
    except Exception as e:
        print(f"❌ 读取配置失败: {e}")
        return

    # 2. 实例化 Dataset
    try:
        dataset = OCFDataset(cfg)
        print(f"✅ 数据集实例化成功，共有 {len(dataset)} 个样本。")
    except Exception as e:
        print(f"❌ 数据集实例化失败: {e}")
        return

    # 3. 实例化 DataLoader (模拟训练时的设定，batch_size 设小一点用于测试)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # 4. 获取一个 Batch 的数据
    print("\n⏳ 正在抓取第一个 Batch 的数据...")
    try:
        batch = next(iter(dataloader))
    except Exception as e:
        print(f"❌ 迭代 DataLoader 失败，可能是 __getitem__ 内部报错: {e}")
        return

    # 5. 打印体检报告 (核对 Shape 极其重要！)
    print("\n" + "="*40)
    print("📊 Batch 张量形状体检报告")
    print("="*40)
    for key, tensor in batch.items():
        print(f"👉 {key.ljust(15)}: {tensor.shape}  |  dtype: {tensor.dtype}")
    print("="*40)

    # 6. 可视化采样点分布 (验证 Global/Edge/Corner 策略是否符合预期)
    print("\n🎨 正在生成采样点分布可视化 (取 Batch 中的第 0 个样本)...")
    
    # 取出第一个样本的坐标、标签和权重
    coords = batch['coords'][0].numpy()   # [N, 2]
    labels = batch['labels'][0].numpy().squeeze() # [N]
    weights = batch['weights'][0].numpy().squeeze() # [N]
    
    plt.figure(figsize=(15, 5))
    
    # 子图 1：根据内外标签着色
    plt.subplot(1, 2, 1)
    plt.scatter(coords[labels==0, 0], coords[labels==0, 1], c='black', s=1, alpha=0.5, label='Outside (0)')
    plt.scatter(coords[labels==1, 0], coords[labels==1, 1], c='blue', s=1, alpha=0.8, label='Inside (1)')
    plt.title("Sample Points: Inside vs Outside")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.gca().set_aspect('equal')

    # 子图 2：根据采样权重着色 (验证 Edge/Corner 的高权重是否落在边界上)
    plt.subplot(1, 2, 2)
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=weights, cmap='viridis', s=2, alpha=0.8)
    plt.colorbar(sc, label='Loss Weight')
    plt.title("Sample Points: Loss Weights (Hotter = Higher Weight)")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')

    # 保存图片
    save_path = "debug_sampling_points.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 采样分布图已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    debug_dataloader()
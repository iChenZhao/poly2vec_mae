import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_samples(file_path, num_samples=3):
    # 1. 加载数据
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        print(f"成功加载文件: {file_path}, 总样本数: {len(data)}")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 随机选几个样本
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)

    # 创建画布
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
    if len(indices) == 1: axes = [axes] 

    for i, idx in enumerate(indices):
        sample = data[idx]
        
        # 提取数据
        tris = sample['triangles']   
        real = sample['freq_real']   
        imag = sample['freq_imag']   
        mag = torch.sqrt(real**2 + imag**2)

        # --- 1. 几何图 ---
        ax_tri = axes[i][0]
        for t in range(tris.shape[0]):
            points = tris[t].numpy()
            polygon = np.concatenate([points, points[:1]], axis=0)
            ax_tri.plot(polygon[:, 0], polygon[:, 1], 'b-', alpha=0.6, linewidth=0.8)
        
        ax_tri.set_title(f"Sample {idx}: Geometry")
        
        # --- 关键修改点：强制锁定轴范围 ---
        ax_tri.set_xlim([-1.05, 1.05]) # 留一点点边距，防止贴边
        ax_tri.set_ylim([-1.05, 1.05])
        
        # 保持物理上的正方形，防止拉伸变形
        ax_tri.set_aspect('equal', adjustable='box')

        # --- 2. 实部图 ---
        ax_real = axes[i][1]
        im1 = ax_real.imshow(real.numpy(), cmap='RdBu_r')
        ax_real.set_title("Freq Real (Phase Info)")
        plt.colorbar(im1, ax=ax_real)

        # --- 3. 幅值图 ---
        ax_mag = axes[i][2]
        im2 = ax_mag.imshow(mag.numpy(), cmap='magma')
        ax_mag.set_title("Freq Magnitude")
        plt.colorbar(im2, ax=ax_mag)

    plt.tight_layout()
    
    # 保存图片而不是直接显示
    output_name = "debug_visualization.png"
    plt.savefig(output_name, dpi=150)
    print(f"可视化结果已保存至: {os.path.abspath(output_name)}")
    plt.close()

if __name__ == "__main__":
    visualize_samples('phase.pt')
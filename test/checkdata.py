import torch
import numpy as np

# 新增这行：强制 matplotlib 使用非交互式后端（必须在 import pyplot 之前！）
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotPolygon

# ==========================================
# 1. 前置可视化函数
# ==========================================
def inspect_polygon_freq_data(sample, sample_idx=0):
    # 解析数据
    triangles = sample.get('triangles', None)
    real = sample.get('freq_real', None)
    imag = sample.get('freq_imag', None)
    
    # 兼容性检查：确保核心频域数据存在
    if real is None or imag is None:
        print(f"❌ 样本 {sample_idx} 缺少 'freq_real' 或 'freq_imag' 键！现有键: {list(sample.keys())}")
        return

    # 转为 numpy
    if triangles is not None:
        triangles = triangles.cpu().numpy() if torch.is_tensor(triangles) else triangles
    real = real.cpu().numpy() if torch.is_tensor(real) else real
    imag = imag.cpu().numpy() if torch.is_tensor(imag) else imag
    
    # 计算振幅和相位
    amplitude = np.sqrt(real**2 + imag**2 + 1e-8)
    phase = np.arctan2(imag, real)
    
    # --- 打印体检报告 ---
    print(f"\n=== Sample {sample_idx} 数据体检报告 ===")
    print(f"[频域形状] Real: {real.shape}, Imag: {imag.shape}")
    print(f"[Real 极值] Min: {real.min():.4f}, Max: {real.max():.4f}, Mean: {real.mean():.4f}")
    print(f"[Imag 极值] Min: {imag.min():.4f}, Max: {imag.max():.4f}, Mean: {imag.mean():.4f}")
    print(f"[振幅极值] Min: {amplitude.min():.4f}, Max: {amplitude.max():.4f}")
    print("=======================================\n")

    # --- 绘图 ---
    fig = plt.figure(figsize=(24, 5))
    plt.suptitle(f"Polygon Frequency Asset Analysis (Sample {sample_idx})", fontsize=16, fontweight='bold')

    # (A) 空间几何
    ax1 = fig.add_subplot(1, 5, 1)
    if triangles is not None:
        for tri in triangles:
            polygon = MatplotPolygon(tri, closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.5)
            ax1.add_patch(polygon)
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_title("1. Spatial Geometry")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # (B) Real
    ax2 = fig.add_subplot(1, 5, 2)
    im2 = ax2.imshow(real, cmap='RdBu', aspect='auto')
    ax2.set_title("2. Frequency: Real")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # (C) Imag
    ax3 = fig.add_subplot(1, 5, 3)
    im3 = ax3.imshow(imag, cmap='RdBu', aspect='auto')
    ax3.set_title("3. Frequency: Imaginary")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # (D) Log Amplitude (找零频)
    ax4 = fig.add_subplot(1, 5, 4)
    log_amp = np.log1p(amplitude) 
    im4 = ax4.imshow(log_amp, cmap='magma', aspect='auto')
    ax4.set_title("4. Derived: Log Amplitude")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    max_idx = np.unravel_index(np.argmax(amplitude, axis=None), amplitude.shape)
    ax4.plot(max_idx[1], max_idx[0], 'r+', markersize=15, markeredgewidth=2)
    ax4.text(max_idx[1]+2, max_idx[0], 'DC', color='red', fontsize=12)

    # (E) Phase
    ax5 = fig.add_subplot(1, 5, 5)
    im5 = ax5.imshow(phase, cmap='hsv', aspect='auto')
    ax5.set_title("5. Derived: Phase")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = f"freq_analysis_sample_0.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 图像已成功保存至: {save_path}")
    
    plt.close(fig) # 释放内存，防止批量处理时内存泄漏


# ==========================================
# 2. 核心：加载与解剖 phase.pt
# ==========================================
if __name__ == "__main__":
    file_path = 'data/ocf_augmented_data_final_100.pt' # 确保文件名和路径正确
    print(f"正在加载 {file_path} ...")
    
    try:
        # map_location='cpu' 防止你在没有GPU的机器上读取报错
        data = torch.load(file_path, map_location='cpu') 
    except Exception as e:
        print(f"读取失败: {e}")
        exit()

    print(f"✅ 读取成功！数据的顶层类型是: {type(data)}")

    # 情景 1: data 是一个列表 (通常是整个数据集 [sample1, sample2, ...])
    if isinstance(data, list):
        print(f"数据是一个列表，共包含 {len(data)} 个样本。")
        sample_to_view = data[0] # 取出第一个样本
        print(f"样本类型: {type(sample_to_view)}")
        if isinstance(sample_to_view, dict):
            print(f"样本包含的 Keys: {list(sample_to_view.keys())}")
            inspect_polygon_freq_data(sample_to_view, sample_idx=0)
        else:
            print("列表里的元素不是字典，无法自动解析，请手动检查。")

    # 情景 2: data 是一个字典 (可能包含 'freq_real': [N, H, W], 'triangles': [N, T, 3, 2] 等)
    elif isinstance(data, dict):
        print(f"数据是一个字典，包含的 Keys: {list(data.keys())}")
        
        # 我们需要从这个大字典里，提取出第 0 个样本构成一个小字典
        sample_to_view = {}
        for key in data.keys():
            # 假设字典里的 value 是带有 Batch 维度的 Tensor
            if torch.is_tensor(data[key]) and len(data[key]) > 0:
                sample_to_view[key] = data[key][0] # 取 Batch 的第 0 个
            else:
                sample_to_view[key] = data[key]
        
        inspect_polygon_freq_data(sample_to_view, sample_idx=0)

    # 情景 3: 未知结构
    else:
        print("数据的结构无法自动识别，请查看以下内容：")
        print(data)
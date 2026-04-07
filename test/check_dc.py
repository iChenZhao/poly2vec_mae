import torch
import numpy as np
from collections import Counter

def check_dataset_dc(data_path):
    print(f"🚀 正在加载数据集进行 DC 能量中心检测: {data_path}")
    try:
        data = torch.load(data_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return

    samples = data.get('samples', [])
    if not samples:
        print("❌ 未在数据集中找到 'samples' 列表，请检查数据格式。")
        return
        
    print(f"✅ 成功加载数据，共包含 {len(samples)} 个样本。")
    print("⏳ 正在计算所有样本的频域能量峰值坐标...")

    dc_locations = []
    
    for i, sample in enumerate(samples):
        # 获取实部和虚部
        freq_real = torch.as_tensor(sample['freq_real']).float()
        freq_imag = torch.as_tensor(sample['freq_imag']).float()
        
        # 计算振幅图 (能量)
        amplitude = torch.sqrt(freq_real**2 + freq_imag**2)
        
        # 找到能量最大的元素的平铺索引
        max_idx_flat = torch.argmax(amplitude).item()
        
        # 将平铺索引转换回 [H, W] 的 2D 坐标
        H, W = amplitude.shape
        h_idx = max_idx_flat // W
        w_idx = max_idx_flat % W
        
        dc_locations.append((h_idx, w_idx))
        
    # 统计不同能量中心坐标出现的频次
    counter = Counter(dc_locations)
    
    print("\n" + "="*50)
    print("📊 训练集 DC (零频/最高能量) 位置统计报告")
    print("="*50)
    
    total_samples = len(samples)
    for loc, count in counter.most_common():
        percentage = (count / total_samples) * 100
        print(f"📍 坐标 [H={loc[0]}, W={loc[1]}]: 命中 {count} 次 ({percentage:.2f}%)")
        
    print("="*50)
    
    # 专家诊断结论
    most_common_loc = counter.most_common(1)[0][0]
    if most_common_loc == (31, 0):
        if counter[most_common_loc] == total_samples:
            print("\n🎉 【诊断结果】: 完美！100% 的样本 DC 中心都严格锚定在 [31, 0]！")
            print("👉 你的数据预处理完全正确，请放心使用我刚才给出的 decoder.py 修正方案进行训练。")
        else:
            print(f"\n⚠️ 【诊断结果】: 大部分 DC 在 [31, 0]，但有 {total_samples - counter[most_common_loc]} 个样本发生了漂移！")
            print("👉 建议排查那些漂移的样本，可能是在 rFFT 之前的 Padding 或 Crop 尺寸不一致导致的。")
    else:
        print(f"\n🚨 【致命警告】: 能量中心主要位于 {most_common_loc}，而不是预期的 [31, 0]！")
        print("👉 请立刻停止训练！如果数据 DC 在这里，你需要依据这个实际坐标去修改 decoder.py 的 freq_grid 范围。")

if __name__ == "__main__":
    # 替换为你真实的 .pt 数据集路径
    DATASET_PATH = "data/data_clean_full.pt" 
    check_dataset_dc(DATASET_PATH)
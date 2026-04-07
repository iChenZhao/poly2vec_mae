import torch
import numpy as np
from tqdm import tqdm # 如果没有 tqdm，可以使用 pip install tqdm 安装，用于显示进度条

def clean_ocf_dataset(input_path, output_path):
    print(f"🚀 开始加载原始数据集: {input_path}")
    try:
        data = torch.load(input_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return

    metadata = data.get('metadata', {})
    samples = data.get('samples', [])
    
    if not samples:
        print("❌ 未找到 'samples' 列表，请检查数据格式。")
        return
        
    total_samples = len(samples)
    print(f"📦 成功加载数据，共包含 {total_samples} 个样本。准备进行清洗...")

    clean_samples = []
    discarded_indices = []

    # 遍历所有样本进行严格过滤
    for idx, sample in enumerate(tqdm(samples, desc="Filtering DC anomalies")):
        # 提取频域特征
        freq_real = torch.as_tensor(sample['freq_real']).float()
        freq_imag = torch.as_tensor(sample['freq_imag']).float()
        
        # 计算振幅
        amplitude = torch.sqrt(freq_real**2 + freq_imag**2)
        
        # 找到能量最高点的 2D 坐标
        max_idx_flat = torch.argmax(amplitude).item()
        H, W = amplitude.shape
        h_idx = max_idx_flat // W
        w_idx = max_idx_flat % W
        
        # 核心判定逻辑：只有 DC 严格在 [31, 0] 的才保留
        if h_idx == 31 and w_idx == 0:
            clean_samples.append(sample)
        else:
            # 记录被抛弃的样本原索引及其错误的 DC 位置
            discarded_indices.append((idx, h_idx, w_idx))

    # 打印清洗报告
    clean_count = len(clean_samples)
    discard_count = len(discarded_indices)
    
    print("\n" + "="*50)
    print("🧹 数据清洗报告")
    print("="*50)
    print(f"✅ 原始样本数: {total_samples}")
    print(f"✅ 保留样本数: {clean_count}")
    print(f"🗑️ 剔除样本数: {discard_count}")
    print("="*50)
    
    if discard_count > 0:
        print("🚨 被剔除的异常样本信息 (原索引, 错误H, 错误W):")
        # 最多打印前 20 个异常样本信息，防止刷屏
        for info in discarded_indices[:20]:
            print(f"   - 样本索引: {info[0]:<5} | 漂移位置: [H={info[1]}, W={info[2]}]")
        if discard_count > 20:
            print(f"   ... (及其他 {discard_count - 20} 个样本)")

    # 重新打包保存
    print(f"\n💾 正在将纯净数据保存至: {output_path}")
    clean_data = {
        "metadata": metadata,
        "samples": clean_samples
    }
    torch.save(clean_data, output_path)
    print("🎉 保存完成！你可以使用新的纯净数据集开始训练了！")

if __name__ == "__main__":
    # 请修改为你的实际路径
    INPUT_FILE = "data/data_full.pt"
    # 建议输出文件名加上数量，避免混淆
    OUTPUT_FILE = "data/data_clean_full.pt" 
    
    clean_ocf_dataset(INPUT_FILE, OUTPUT_FILE)
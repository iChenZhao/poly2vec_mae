import os
import sys
import torch
import random
import numpy as np

# 兼容性补丁
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core

def create_mini_dataset(input_path, output_path, num_samples=100, random_sample=True):
    print(f"📦 正在载入原始数据: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ 找不到原始数据文件: {input_path}")
        return

    # 1. 加载原始数据 (通常是字典结构)
    data = torch.load(input_path, weights_only=False, map_location='cpu')
    
    # 获取真正的样本列表
    if isinstance(data, dict) and "samples" in data:
        full_samples_list = data["samples"]
        metadata = data.get("metadata", {})
    else:
        # 如果数据直接就是列表，则按原逻辑处理
        full_samples_list = data
        metadata = None

    total_samples = len(full_samples_list)
    print(f"✅ 成功加载！原始数据集包含 {total_samples} 个样本。")

    # 2. 检查请求的数量
    if num_samples > total_samples:
        print(f"⚠️ 警告：请求 {num_samples} 个，实际只有 {total_samples} 个。")
        num_samples = total_samples

    # 3. 提取子集
    if random_sample:
        print(f"🎲 正在随机抽取 {num_samples} 个样本...")
        random.seed(42) 
        mini_samples = random.sample(full_samples_list, num_samples)
    else:
        print(f"✂️ 正在按顺序截取...")
        mini_samples = full_samples_list[:num_samples]

    # 4. 重新封装成原始格式并保存
    print(f"💾 正在保存微型数据集至: {output_path}")
    
    if metadata is not None:
        # 保持与图示一致的字典格式
        output_data = {
            "metadata": metadata,
            "samples": mini_samples
        }
    else:
        output_data = mini_samples

    torch.save(output_data, output_path)
    print(f"🎉 提取完成！新文件包含 {len(mini_samples)} 个样本。")

if __name__ == "__main__":
    # 配置区
    INPUT_FILE = "data/data_clean_full.pt" 
    OUTPUT_FILE = "data/data_100.pt" 
    NUM_SAMPLES = 100
    RANDOM_SAMPLE = True 
    
    create_mini_dataset(INPUT_FILE, OUTPUT_FILE, NUM_SAMPLES, RANDOM_SAMPLE)
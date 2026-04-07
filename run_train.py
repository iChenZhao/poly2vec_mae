# ==============================================================================
# 单卡启动: python scripts/train.py --config configs/recons.yaml
# 多卡启动: torchrun --nproc_per_node=4 scripts/train.py --config configs/recons.yaml
# ==============================================================================

import os
import sys
import yaml
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------
# 1. 路径与环境补丁
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "./")) # 假设在根目录运行
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 导入全新的架构组件
from src.decoder import AnalyticallyGuidedDecoder
from src.loader_ocf import OCFDataset
from src.loss import OccupancyLoss

# DDP 分布式训练库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------
# 📐 指标计算函数 (极度关键：处理 Logits)
# ---------------------------------------------------------
def calculate_iou(pred_logits, target_labels, threshold=0.5):
    """
    计算占用场点集的 Batch 平均 IoU
    注意：网络输出的是 Logits，必须先 Sigmoid 变成概率，才能和 target (0/1) 计算！
    """
    probs = torch.sigmoid(pred_logits)
    p_bin = (probs > threshold).float()
    
    inter = (p_bin * target_labels).sum(dim=1) 
    union = p_bin.sum(dim=1) + target_labels.sum(dim=1) - inter
    iou = (inter + 1e-7) / (union + 1e-7)
    return iou.mean()

def main():
    # =========================================================
    # 2. 参数解析与命令行覆盖
    # =========================================================
    parser = argparse.ArgumentParser(description="Freq-MLP OCF Training")
    parser.add_argument('--config', type=str, default='configs/recons.yaml', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    # 读取 YAML 配置
    config_path = os.path.join(PROJECT_ROOT, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 命令行覆盖逻辑
    if args.data_path: cfg['paths']['data_path'] = args.data_path
    if args.save_dir: cfg['paths']['save_dir'] = args.save_dir
    if args.batch_size: cfg['training']['batch_size'] = args.batch_size
    if args.lr: cfg['training']['learning_rate'] = args.lr
    if args.epochs: cfg['training']['num_epochs'] = args.epochs

    # =========================================================
    # 3. DDP 环境初始化
    # =========================================================
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1

    def print_info(*msg):
        if local_rank == 0: print(*msg)

    # =========================================================
    # 4. 数据准备 (90 / 5 / 5 切分)
    # =========================================================
    save_dir = os.path.abspath(os.path.join(PROJECT_ROOT, cfg['paths']['save_dir']))
    if local_rank == 0: os.makedirs(save_dir, exist_ok=True)

    dataset = OCFDataset(cfg)
    
    total = len(dataset)
    torch.manual_seed(42) # 保证每次运行切分结果一致

    train_size = int(0.90 * total)
    val_size = int(0.05 * total)
    test_size = total - train_size - val_size
    
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    if local_rank == 0:
        # 保存索引 (为后续断点续训或评估做准备)
        indices_path = os.path.join(save_dir, "train_val_indices.pt")
        torch.save({'train': train_set.indices, 'val': val_set.indices}, indices_path)
        
        test_indices_save_path = os.path.abspath(os.path.join(PROJECT_ROOT, cfg['paths']['test_indices_path']))
        os.makedirs(os.path.dirname(test_indices_save_path), exist_ok=True)
        torch.save(test_set.indices, test_indices_save_path)
        
        print_info(f"📊 数据就绪: 训练({len(train_set)}) | 验证({len(val_set)}) | 测试({len(test_set)} 封存)")

    # Dataloader 实例化
    bs = cfg['training']['batch_size']
    if is_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoader(train_set, batch_size=bs, sampler=train_sampler, 
                                  num_workers=8, pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, 
                                  num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=4)

    # =========================================================
    # 5. 初始化模型、优化器与 Loss
    # =========================================================
    model = AnalyticallyGuidedDecoder(
        embedding_dim=cfg['model']['embedding_dim'],
        H=cfg['model']['freq_H'],
        W=cfg['model']['freq_W'],
        hidden_dims=cfg['model']['hidden_dims']
    ).to(device)
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 抛弃 SIREN 后，GELU MLP 允许使用 weight_decay 来防止过拟合
    lr = float(cfg['training']['learning_rate'])
    wd = float(cfg['training']['weight_decay'])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)  
    
    epochs = cfg['training']['num_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = OccupancyLoss(cfg)

    # =========================================================
    # 6. 极简训练大循环 (拒绝刷屏，静默狂飙)
    # =========================================================
    best_iou = 0.0
    print_info(f"\n🚀 开始纯净模式训练，设备: {device} | 模式: {'DDP' if is_ddp else 'Single-GPU'}")
    print_info("-" * 75)
    print_info(f"{'Epoch':<10} | {'Phase':<5} | {'Loss':<8} | {'mIoU':<8} | {'LR':<8} | {'Time'}")
    print_info("-" * 75)

    start_global_time = time.time()

    for epoch in range(epochs):
        if is_ddp: train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        
        # --- A. 训练阶段 ---
        model.train()
        train_loss_list = []
        train_iou_list = []
        
        for batch in train_loader:
            # 优雅的字典解包与设备转移
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # 前向传播 (传入坐标、频域资产、宏观语义)
            logits = model(
                coords=batch['coords'],
                freq_real=batch['freq_real'],
                freq_imag=batch['freq_imag'],
                embedding=batch['embedding']
            )
            
            # 混合 Loss 计算
            loss, _, _ = criterion(logits, batch['labels'], batch['weights'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 指标记录
            iou = calculate_iou(logits.detach(), batch['labels'])
            train_loss_list.append(loss.item())
            train_iou_list.append(iou.item())

        # --- B. 验证阶段 ---
        model.eval()
        val_loss_list = []
        val_iou_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                logits = model(
                    coords=batch['coords'],
                    freq_real=batch['freq_real'],
                    freq_imag=batch['freq_imag'],
                    embedding=batch['embedding']
                )
                
                loss, _, _ = criterion(logits, batch['labels'], batch['weights'])
                iou = calculate_iou(logits, batch['labels'])
                
                val_loss_list.append(loss.item())
                val_iou_list.append(iou.item())

        # --- C. DDP 同步指标 ---
        avg_train_loss = np.mean(train_loss_list)
        avg_train_iou = np.mean(train_iou_list)
        avg_val_loss = np.mean(val_loss_list)
        avg_val_iou = np.mean(val_iou_list)

        metrics = torch.tensor([avg_train_loss, avg_train_iou, avg_val_loss, avg_val_iou]).to(device)
        if is_ddp:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
        
        t_loss, t_iou, v_loss, v_iou = metrics.tolist()
        
        # --- D. 极简终端输出 (首轮，以及每 5 轮显示) ---
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            ep_time = time.time() - epoch_start_time
            
            if epoch == 0 or (epoch + 1) % 5 == 0:
                print_info(f"Ep {epoch+1:<6} | Train | {t_loss:.5f} | {t_iou:.4f}   | {current_lr:.1e} | {ep_time:.1f}s")
                print_info(f"           | Val   | {v_loss:.5f} | {v_iou:.4f}   | {'-':<7} |")
                
                # 模型存档逻辑
                model_to_save = model.module if is_ddp else model
                if v_iou > best_iou:
                    best_iou = v_iou
                    torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'freq_mlp_best.pth'))
                    print_info(f"   🏆 发现最佳模型! (mIoU: {best_iou:.4f}) 已覆盖 best.pth")
                
                if (epoch + 1) % cfg['training']['save_freq'] == 0:
                    torch.save(model_to_save.state_dict(), os.path.join(save_dir, f'freq_mlp_ep{epoch+1}.pth'))
                
                print_info("-" * 75)

        # 学习率推进
        scheduler.step()

    # 结束
    if local_rank == 0:
        total_time = (time.time() - start_global_time) / 3600.0
        print_info(f"\n🎉 训练圆满结束！最高验证 mIoU: {best_iou:.4f} | 总耗时: {total_time:.2f} 小时")
        
    if is_ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()
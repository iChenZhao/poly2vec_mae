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
from tqdm import tqdm

# DDP 分布式训练库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------
# 1. 路径与环境补丁
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "./"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.decoder_v1 import AnalyticallyGuidedDecoder
from src.loader_ocf import OCFDataset
from src.loss import OccupancyLoss

# ---------------------------------------------------------
# 📐 指标计算函数
# ---------------------------------------------------------
def calculate_iou(pred_logits, target_labels, threshold=0.5):
    """ 计算 Batch 平均 IoU """
    with torch.no_grad():
        probs = torch.sigmoid(pred_logits)
        p_bin = (probs > threshold).float()
        inter = (p_bin * target_labels).sum(dim=1) 
        union = p_bin.sum(dim=1) + target_labels.sum(dim=1) - inter
        iou = (inter + 1e-7) / (union + 1e-7)
    return iou.mean()

def main():
    # =========================================================
    # 2. 参数解析与命令行覆盖 (满血功能保留)
    # =========================================================
    parser = argparse.ArgumentParser(description="Full Scale Freq-OCF Training")
    parser.add_argument('--config', type=str, default='configs/recons.yaml', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    # 读取 YAML 配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 优先级：命令行覆盖 > YAML 配置
    if args.data_path: cfg['paths']['data_path'] = args.data_path
    if args.save_dir: cfg['paths']['save_dir'] = args.save_dir
    if args.batch_size: cfg['training']['batch_size'] = args.batch_size
    if args.lr: cfg['training']['learning_rate'] = args.lr
    if args.epochs: cfg['training']['num_epochs'] = args.epochs

    # =========================================================
    # 3. 环境初始化 (DDP + Single GPU 兼容)
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
    # 4. 数据准备 (严谨的 90 / 5 / 5 切分)
    # =========================================================
    dataset = OCFDataset(cfg)
    total = len(dataset)
    torch.manual_seed(42)

    train_size = int(0.90 * total)
    val_size = int(0.05 * total)
    test_size = total - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    if local_rank == 0:
        save_dir = cfg['paths']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        # 保存索引用于后续评估
        torch.save({'train': train_set.indices, 'val': val_set.indices, 'test': test_set.indices}, 
                   os.path.join(save_dir, "dataset_split_indices.pt"))
        print_info(f"📊 数据就绪 | 总数: {total} | 训练: {len(train_set)} | 验证: {len(val_set)} | 测试: {len(test_set)}")

    # DataLoader 实例化
    bs = cfg['training']['batch_size']
    nw = 8 # 根据 4090 性能建议设为 8
    train_sampler = DistributedSampler(train_set) if is_ddp else None
    train_loader = DataLoader(train_set, batch_size=bs, sampler=train_sampler, 
                              shuffle=(train_sampler is None), num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=4)

    # =========================================================
    # 5. 初始化模型架构、优化器与 Loss
    # =========================================================
    model = AnalyticallyGuidedDecoder(
        embedding_dim=cfg['model']['embedding_dim'],
        H=cfg['model']['freq_H'],
        W=cfg['model']['freq_W'],
        hidden_dims=cfg['model']['hidden_dims']
    ).to(device)
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), 
                            lr=float(cfg['training']['learning_rate']), 
                            weight_decay=float(cfg['training']['weight_decay']))
    
    # 采用余弦退火调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['num_epochs'])
    criterion = OccupancyLoss(cfg)

    # =========================================================
    # 6. 训练大循环 (加入进度条与紧凑输出)
    # =========================================================
    best_iou = 0.0
    epochs = cfg['training']['num_epochs']
    
    print_info(f"\n🚀 开始训练 | GPU: {device} | 模式: {'DDP' if is_ddp else 'Single'}")
    print_info("-" * 100)

    for epoch in range(epochs):
        if is_ddp: train_sampler.set_epoch(epoch)
        
        # --- A. 训练阶段 ---
        model.train()
        train_loss_list, train_iou_list = [], []
        
        # 引入 tqdm 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs}", 
                    disable=(local_rank != 0), dynamic_ncols=True)
        
        for batch in pbar:
            # 补丁：只将 Tensor 移动到 GPU，跳过 triangles 等原始数据
            batch_cuda = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            optimizer.zero_grad()
            logits = model(batch_cuda['coords'], batch_cuda['freq_real'], batch_cuda['freq_imag'], batch_cuda['embedding'])
            
            loss, bce_l, dice_l = criterion(logits, batch_cuda['labels'], batch_cuda['weights'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 记录指标
            iou = calculate_iou(logits, batch_cuda['labels'])
            train_loss_list.append(loss.item())
            train_iou_list.append(iou.item())
            
            # 实时更新进度条后缀 (显示当前 Loss 和 mIoU)
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "BCE": f"{bce_l:.4f}",
                "Dice": f"{dice_l:.4f}",
                "mIoU": f"{iou.item():.4f}"
            })

        # --- B. 验证阶段 ---
        model.eval()
        val_iou_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch_cuda = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                logits = model(batch_cuda['coords'], batch_cuda['freq_real'], batch_cuda['freq_imag'], batch_cuda['embedding'])
                val_iou_list.append(calculate_iou(logits, batch_cuda['labels']).item())

        # --- C. 指标汇总与同步 ---
        avg_train_loss = np.mean(train_loss_list)
        avg_train_iou = np.mean(train_iou_list)
        avg_val_iou = np.mean(val_iou_list)

        # DDP 下同步验证集 IoU
        if is_ddp:
            metrics = torch.tensor([avg_val_iou]).to(device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            avg_val_iou = (metrics[0] / world_size).item()
        
        # --- D. 紧凑型终端日志输出 ---
        if local_rank == 0:
            lr_now = optimizer.param_groups[0]['lr']
            # 格式：Epoch | T-Loss | T-mIoU | V-mIoU | LR
            print_info(f"✨ Ep {epoch+1:03d} Summary | Train Loss: {avg_train_loss:.5f} | Train mIoU: {avg_train_iou:.4f} | Val mIoU: {avg_val_iou:.4f} | LR: {lr_now:.1e}")
            
            # 存档逻辑
            model_to_save = model.module if is_ddp else model
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                torch.save(model_to_save.state_dict(), os.path.join(cfg['paths']['save_dir'], 'best_model.pth'))
                print_info(f"   🏆 New Best Model! mIoU: {best_iou:.4f}")
            
            if (epoch + 1) % cfg['training']['save_freq'] == 0:
                torch.save(model_to_save.state_dict(), os.path.join(cfg['paths']['save_dir'], f'model_ep{epoch+1:03d}.pth'))
            
            print_info("-" * 100)

        scheduler.step()

    if local_rank == 0:
        print_info(f"\n🎉 训练圆满完成！最高验证 mIoU: {best_iou:.4f}")
    
    if is_ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()
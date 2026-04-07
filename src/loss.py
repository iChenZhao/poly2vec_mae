import torch
import torch.nn as nn
import torch.nn.functional as F

class OccupancyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        l_cfg = cfg['loss_weights']
        self.bce_weight = l_cfg.get('bce_weight', 1.0)
        self.dice_weight = l_cfg.get('dice_weight', 10.0)
        self.ohem_ratio = l_cfg.get('ohem_ratio', 0.8)

    def forward(self, pred_logits, target_labels, weights=None):
        """
        pred_logits: [B, N, 1]
        target_labels: [B, N, 1]
        weights: [B, N, 1]
        """
        # 1. 加权 BCE 计算
        # 注意：先计算原始 BCE，再乘权重，确保边缘采样点贡献更大的梯度
        bce_all = F.binary_cross_entropy_with_logits(pred_logits, target_labels, reduction='none')
        
        if weights is not None:
            bce_all = bce_all * weights

        # OHEM 逻辑：在 Batch 层面筛选最难的 top_k 个点
        B, N, _ = pred_logits.shape
        bce_flat = bce_all.view(-1)
        num_elements = bce_flat.numel()
        top_k = int(self.ohem_ratio * num_elements)
        
        if top_k < num_elements:
            bce_val, _ = torch.topk(bce_flat, top_k)
            bce_loss = bce_val.mean()
        else:
            bce_loss = bce_all.mean()

        # 2. Log-Cosh Dice Loss (相比 1-Dice 更平滑，不易梯度消失)
        probs = torch.sigmoid(pred_logits)
        inter = (probs * target_labels).sum(dim=1)
        union = probs.sum(dim=1) + target_labels.sum(dim=1)
        dice_score = (2. * inter + 1e-6) / (union + 1e-6)
        
        # log(cosh(x)) 在 x 接近 0 时梯度更温和
        dice_loss = torch.log(torch.cosh(1. - dice_score)).mean()

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss, bce_loss.detach(), dice_loss.detach()
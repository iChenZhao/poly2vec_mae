import torch
import torch.nn as nn
import math

class AnalyticallyGuidedDecoder(nn.Module):
    def __init__(self, embedding_dim=384, H=63, W=32, hidden_dims=[512, 512, 256, 128], use_embedding=True): # 增加开关
        super().__init__()
        
        self.H = H
        self.W = W
        self.num_freqs = H * W  # 63 * 32 = 2016
        
        # 1. 注册解析频率网格 [2016, 2]
        self.register_buffer('freqs', self._get_cft_freq_grid(H, W))
        
        # 2. 核心解码器：根据 yaml 配置动态构建 GELU MLP
        self.use_embedding = use_embedding
        if use_embedding:
            in_dim = 2 + self.num_freqs + embedding_dim # 2402
        else:
            in_dim = 2 + self.num_freqs # 2018
        
        layers = []
        current_dim = in_dim
        
        # 动态添加隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
            
        # 最后一层输出 1 维 logits
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, coords, freq_real, freq_imag, embedding):
        """
        前向传播
        :param coords: [B, N, 2] 空间查询坐标 (归一化在 -1 到 1)
        :param freq_real: [B, 63, 32] 频域实部
        :param freq_imag: [B, 63, 32] 频域虚部
        :param embedding: [B, 384] 全局语义特征
        :return: [B, N, 1] 占用概率的 Logits
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        B = coords.shape[0]
        N = coords.shape[1]

        # --- 预处理频域数据 ---
        amplitudes = torch.sqrt(freq_real**2 + freq_imag**2 + 1e-8) # [B, 63, 32]
        phases = torch.atan2(freq_imag, freq_real)                  # [B, 63, 32]
        
        amplitudes = amplitudes.view(B, -1) # [B, 2016]
        phases = phases.view(B, -1)         # [B, 2016]

        # --- 计算空间引发的偏移与瞬时相位 ---
        spatial_shift = torch.matmul(coords, self.freqs.transpose(0, 1)) # [B, N, 2016]
        theta = 2.0 * math.pi * spatial_shift + phases.unsqueeze(1)      # [B, N, 2016]

        # --- 连续解析响应 ---
        responses = amplitudes.unsqueeze(1) * torch.cos(theta) # [B, N, 2016]

        # --- 宏观与微观特征拼接 ---
        emb_expanded = embedding.unsqueeze(1).expand(-1, N, -1) # [B, N, 384]
        if self.use_embedding:
            emb_expanded = embedding.unsqueeze(1).expand(-1, N, -1)
            h_in = torch.cat([coords, responses, emb_expanded], dim=-1)
        else:
            h_in = torch.cat([coords, responses], dim=-1) # 这就是 2018 维
        # --- MLP 解码生成 logits ---
        logits = self.network(h_in) # [B, N, 1]
        
        return logits

    def _get_cft_freq_grid(self, H, W):
        """生成严格匹配 63x32 的归一化频域网格，并修正 XY-UV 映射轴"""
        
        # 1. 修正尺寸与DC中心对齐: 
        # H=63, DC在索引31 -> u的数学范围应严格为 [-31, 31]，总长63
        u = torch.arange(-(H//2), H//2 + 1) 
        
        # W=32, rfft 仅保留非负X轴频率 -> v范围 [0, 31]，总长32
        v = torch.arange(0, W)
        
        # ij 模式：grid_u 对应高度(Y)，grid_v 对应宽度(X)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        
        # 2. 致命修正: 交换栈叠顺序以匹配 coords 的 [X, Y] 结构
        # coords[..., 0] 是 X 坐标 -> 必须乘以 X方向的频率 (grid_v)
        # coords[..., 1] 是 Y 坐标 -> 必须乘以 Y方向的频率 (grid_u)
        freqs = torch.stack([grid_v.flatten(), grid_u.flatten()], dim=-1).float()
        
        return freqs
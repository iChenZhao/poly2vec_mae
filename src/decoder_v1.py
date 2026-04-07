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
        self.res_norm = nn.LayerNorm(2016)

    def forward(self, coords, freq_real, freq_imag, embedding):
        B, N, _ = coords.shape
        
        # 1. 坐标映射 [-1, 1] -> [0, 1]
        norm_coords = (coords + 1.0) / 2.0
        norm_x = norm_coords[..., 0:1]
        norm_y = norm_coords[..., 1:2]

        # 2. 预处理频率和振幅
        amplitudes = torch.sqrt(freq_real**2 + freq_imag**2 + 1e-8).view(B, 1, -1)
        phases = torch.atan2(freq_imag, freq_real).view(B, 1, -1)
        
        # 提取频率网格
        v_flat = self.freqs[..., 0].view(-1)[None, None, :] # X方向频率 (0~31)
        u_flat = self.freqs[..., 1].view(-1)[None, None, :] # Y方向频率 (-31~31)

        # 3. 【核心修正】：计算共轭对称权重 (Factor 2)
        # 只要 X 频率 (v) 大于 0，该频率的响应就要乘以 2，以补偿 rFFT 丢掉的负半轴
        # 这样高频细节的能量才能与 DC 分量对齐
        weights = torch.ones_like(v_flat)
        weights[v_flat > 0] = 2.0  # 关键：双边频谱补偿

        # 4. 解析波浪计算 (分块计算防止 OOM)
        chunk_size = 2000
        responses_list = []
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            curr_x = norm_x[:, i:end_idx, :]
            curr_y = norm_y[:, i:end_idx, :]
            
            spatial_shift = (curr_x * v_flat) + (curr_y * u_flat)
            theta = 2.0 * math.pi * spatial_shift + phases
            # 应用对称权重 weights
            curr_res = weights * amplitudes * torch.cos(theta)
            responses_list.append(curr_res)
            
        responses = torch.cat(responses_list, dim=1) # [B, N, 2016]

        # 5. 【性能飞跃】：响应特征正则化
        # 使用 LayerNorm 或者简单的缩放，防止 2016 维的响应淹没 384 维的 Embedding
        responses = self.res_norm(responses)
        
        # 6. 特征拼接与预测
        emb_expanded = embedding.unsqueeze(1).expand(-1, N, -1)
        h_in = torch.cat([coords, responses, emb_expanded], dim=-1)
        
        # 最终 MLP 
        logits = self.network(h_in)
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
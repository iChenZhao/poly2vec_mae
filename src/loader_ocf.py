import torch
import numpy as np
from torch.utils.data import Dataset
from matplotlib.path import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

class OCFDataset(Dataset):
    def __init__(self, cfg):
        self.data_path = cfg['paths']['data_path']
        print(f"📦 正在载入多模态频域数据集: {self.data_path}")
        
        # 1. 载入原始嵌套字典数据
        raw_data = torch.load(self.data_path, map_location='cpu')
        
        # 2. 提取并保存 metadata (以备后续需要全局缩放等参数)
        self.metadata = raw_data.get('metadata', {})
        
        # 3. 【核心修改】将真正的样本列表赋值给 self.data
        # 这样就能无缝对接你后续所有的遍历和索引逻辑
        self.data = raw_data.get('samples', [])
        
        # 采样参数
        self.num_points = cfg['sampling']['num_points']
        self.ratios = cfg['sampling']['ratios']
        self.jitter_std = cfg['sampling']['edge_jitter_std']
        self.corner_sigma = cfg['sampling']['corner_gauss_sigma']
        
        # 权重参数
        self.weights_cfg = cfg['loss_weights']
        
        # 几何骨架缓存
        self.geometry_cache = []
        self._pre_analyze_all()
        print(f"✅ 几何骨架分析完成，准备就绪。数据量: {len(self.data)}")

    def _pre_analyze_all(self):
        print("⚡ 执行拓扑合并与几何分析 (清理内部边)...")
        for sample in self.data:
            tris = sample['triangles']
            
            # 1. 将所有三角形转换为 Shapely 多边形
            polygons = [Polygon(t) for t in tris]
            
            # 2. 执行并集操作，溶解掉内部所有公共边
            merged = unary_union(polygons)
            
            # 3. 提取纯净的外边界线段
            boundary_edges = []
            
            def extract_from_poly(poly):
                # 处理外轮廓
                coords = np.array(poly.exterior.coords)
                for i in range(len(coords) - 1):
                    boundary_edges.append((coords[i], coords[i+1]))
                # 处理内圈孔洞 (如果有)
                for interior in poly.interiors:
                    coords_int = np.array(interior.coords)
                    for i in range(len(coords_int) - 1):
                        boundary_edges.append((coords_int[i], coords_int[i+1]))

            if isinstance(merged, Polygon):
                extract_from_poly(merged)
            elif isinstance(merged, MultiPolygon):
                for p in merged.geoms:
                    extract_from_poly(p)
            
            boundary_edges = np.array(boundary_edges)
            
            # 计算纯净边界的边长权重
            diff = boundary_edges[:, 0] - boundary_edges[:, 1]
            edge_lengths = np.sqrt(np.sum(diff**2, axis=1))
            edge_probs = edge_lengths / (edge_lengths.sum() + 1e-12)

            # 4. 提取纯净边界顶点 (真正的轮廓转角)
            boundary_verts = np.unique(boundary_edges.reshape(-1, 2), axis=0)

            # 5. 构造判定 Path (保持原有的高效 Path 判定)
            path_verts = []
            path_codes = []
            for tri in tris:
                path_verts.extend([tri[0], tri[1], tri[2], tri[0]])
                path_codes.extend([Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
            
            combined_path = Path(np.array(path_verts), np.array(path_codes))

            self.geometry_cache.append({
                'edges': boundary_edges,
                'edge_probs': edge_probs,
                'corners': boundary_verts,
                'path': combined_path
            }) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1. 获取全局语义向量
        embedding = torch.as_tensor(sample['embedding']).float() # [384]
        
        # 2. 获取频域资产 (确保是 float32)
        freq_real = torch.as_tensor(sample['freq_real']).float() # [H, W]
        freq_imag = torch.as_tensor(sample['freq_imag']).float() # [H, W]

        # 3. 空间采样 (保留你原有的优秀逻辑)
        cache = self.geometry_cache[idx]
        
        n_global = int(self.num_points * self.ratios['global'])
        n_edge = int(self.num_points * self.ratios['edge'])
        n_corner = int(self.num_points * self.ratios['corner'])
        n_inside = int(self.num_points * self.ratios['inside'])

        p_list, w_list = [], []

        # A. 全局均匀采样
        if n_global > 0:
            p_global = np.random.uniform(-1.0, 1.0, (n_global, 2)).astype(np.float32)
            p_list.append(p_global)
            w_list.append(np.full(n_global, self.weights_cfg['global_weight']))

        # B. 边缘带状采样
        if n_edge > 0 and len(cache['edges']) > 0:
            edge_idx = np.random.choice(len(cache['edges']), n_edge, p=cache['edge_probs'])
            selected_edges = cache['edges'][edge_idx]
            t = np.random.rand(n_edge, 1).astype(np.float32)
            p_edge = selected_edges[:, 0] * (1 - t) + selected_edges[:, 1] * t
            p_edge += np.random.normal(0, self.jitter_std, (n_edge, 2))
            p_list.append(p_edge)
            w_list.append(np.full(n_edge, self.weights_cfg['edge_weight']))

        # C. 锐角顶点采样
        if n_corner > 0 and len(cache['corners']) > 0:
            corner_idx = np.random.choice(len(cache['corners']), n_corner)
            selected_corners = cache['corners'][corner_idx]
            p_corner = selected_corners + np.random.normal(0, self.corner_sigma, (n_corner, 2))
            p_list.append(p_corner)
            w_list.append(np.full(n_corner, self.weights_cfg['corner_weight']))

        # D. 内部填充采样
        if n_inside > 0:
            p_inside_list = []
            max_attempts = 10 # 防止陷入死循环
            attempts = 0
            while len(p_inside_list) < n_inside and attempts < max_attempts:
                tmp_p = np.random.uniform(-1.0, 1.0, (n_inside * 2, 2)).astype(np.float32)
                mask = cache['path'].contains_points(tmp_p)
                p_inside_list.extend(tmp_p[mask])
                attempts += 1
            
            # 如果尝试多次仍填不满（比如物体极小），用全局点兜底
            p_inside = np.array(p_inside_list[:n_inside])
            if len(p_inside) < n_inside:
                pad_size = n_inside - len(p_inside)
                p_inside = np.concatenate([p_inside, np.random.uniform(-1.0, 1.0, (pad_size, 2)).astype(np.float32)])
            
            p_list.append(p_inside.astype(np.float32))
            w_list.append(np.full(n_inside, self.weights_cfg['inside_weight']))

        # 合并采样点并剪裁
        p_final = np.concatenate(p_list, axis=0).astype(np.float32)
        p_final = np.clip(p_final, -1.05, 1.05)      
        w_final = np.concatenate(w_list, axis=0).astype(np.float32)

        # 标签判定
        labels = cache['path'].contains_points(p_final, radius=1e-4).astype(np.float32)

        # 返回字典
        return {
            'embedding': embedding,
            'freq_real': freq_real,
            'freq_imag': freq_imag,
            'coords': torch.from_numpy(p_final),
            'labels': torch.from_numpy(labels).unsqueeze(-1),
            'weights': torch.from_numpy(w_final).unsqueeze(-1) #,
            #'triangles': sample['triangles']
        }
import torch
import numpy as np
from typing import Dict, List
from loguru import logger

class IFCRankAllocator:
    """
    IFC 秩分配调度器。
    基于 Soft Scaling 公式实现秩分配:
    $ r_i = R_{min} + (R_{max} - R_{min}) \cdot \text{Softmax}(\tau \cdot IFC) $
    """
    def __init__(self, max_rank: int = 64, min_rank: int = 4, tau: float = 2.0):
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.tau = tau # 逆温度参数 (控制分配稀疏度)

    def allocate(self, ifc_scores: torch.Tensor, node_names: List[str]) -> Dict[str, int]:
        """
        基于 IFC 分数计算个性化秩方案。
        """
        # 使用带温度系数的连续映射
        scaled_ifc = torch.exp(self.tau * ifc_scores)
        scaled_ifc /= scaled_ifc.max() # 归一化到最强的瓶颈节点
        
        # 计算连续秩
        continuous_ranks = self.min_rank + (self.max_rank - self.min_rank) * scaled_ifc
        
        # 离散化并映射回组件名
        discrete_ranks = torch.round(continuous_ranks).int().cpu().tolist()
        
        rank_pattern = {name: int(rank) for name, rank in zip(node_names, discrete_ranks)}
        
        # 统计分布
        avg_rank = sum(rank_pattern.values()) / len(rank_pattern)
        logger.success(f"秩分配完成。平均秩: {avg_rank:.2f} | 目标节点数: {len(rank_pattern)}")
        
        return rank_pattern

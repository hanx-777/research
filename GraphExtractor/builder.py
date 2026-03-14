import torch
from typing import Tuple, List
from loguru import logger

class ImplicitGraphBuilder:
    """
    隐式图构建器：将归因指标转化为符合 Transformer 架构因果流的有向邻接矩阵 $\mathcal{A}$。
    """
    def __init__(self, N: int, device: torch.device):
        self.N = N
        self.device = device

    def build_tensors(self, pert_impi: torch.Tensor, window_size: int = 15) -> torch.Tensor:
        """
        构建邻接矩阵 $\mathcal{A} \in \mathbb{R}^{N \times N}$。
        $ \mathcal{A}_{i,j} $ 表示节点 i 对节点 j 的因果影响带宽，权重基于下游节点的 PertImpi。
        """
        logger.info(f"正在构建张量化邻接矩阵 (节点数: {self.N}, 窗口大小: {window_size})...")
        
        # 使用 GPU 张量直接构建邻接矩阵
        A = torch.zeros((self.N, self.N), device=self.device)
        
        # 按照 Transformer 序列因果顺序构建边 (i -> j iff j > i)
        for i in range(self.N):
            # 建立因果链路：节点 i 流向后续 window_size 内的所有节点
            end_idx = min(i + 1 + window_size, self.N)
            for j in range(i + 1, end_idx):
                # 边权重由目标节点的重要性驱动，符合 PageRank 的“含金量”概念
                A[i, j] = pert_impi[j]
        
        # 行归一化 (Row-stochastic) 确保信息传播守恒
        row_sum = A.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0 # 处理孤立节点
        A = A / row_sum
        
        return A

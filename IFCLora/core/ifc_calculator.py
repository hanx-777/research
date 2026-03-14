import torch
from loguru import logger

class IFCCalculator:
    """
    IFC (Information-Flow Centrality) 综合指标计算。
    公式: $ IFC_i = \alpha \cdot Flow_i + \beta \cdot GradSensi_i + \gamma \cdot PertImpi_i $
    """
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _power_iteration_pagerank(self, A: torch.Tensor, d: float = 0.85, max_iter: int = 100) -> torch.Tensor:
        """
        在 GPU 上利用幂迭代法原生求解矩阵 PageRank (Flow)。
        $ \mathbf{PR}^{(t+1)} = \frac{1-d}{N}\mathbf{1} + d \cdot \mathbf{A}^T \mathbf{PR}^{(t)} $
        """
        N = A.shape[0]
        pr = torch.ones(N, 1, device=A.device) / N
        
        # 将转置邻接矩阵存储以加速迭代
        AT = A.t().contiguous()
        
        for i in range(max_iter):
            pr_new = (1 - d) / N + d * torch.mm(AT, pr)
            # 检查收敛性 (L1 Norm)
            if torch.norm(pr_new - pr, p=1) < 1e-6:
                logger.info(f"PageRank 已在第 {i} 代收敛。")
                break
            pr = pr_new
            
        return pr.squeeze()

    def _z_score_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """应用 Min-Max 归一化，将指标缩放至 [0, 1] 区间以便融合。"""
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    def calculate(self, A: torch.Tensor, grad_v: torch.Tensor, pert_v: torch.Tensor) -> torch.Tensor:
        """
        计算最终的综合 IFC 得分向量。
        """
        logger.info(f"正在计算全局 IFC 指标 (融合权重: a={self.alpha}, b={self.beta}, g={self.gamma})...")
        
        # 1. 求解 Flow (基于张量邻接矩阵的 PageRank)
        flow_v = self._power_iteration_pagerank(A)
        
        # 2. 三维指标归一化与加权融合
        ifc_scores = (
            self.alpha * self._z_score_normalize(flow_v) +
            self.beta * self._z_score_normalize(grad_v) +
            self.gamma * self._z_score_normalize(pert_v)
        )
        
        return ifc_scores

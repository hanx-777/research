import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from loguru import logger

class MechanisticTracer:
    """
    IFCLora 隐式电路追踪器。
    基于机械可解释性 (Mechanistic Interpretability) 捕获模型各组件的统计归因。
    
    指标定义:
    1. GradSensi (梯度敏感度): $ \mathcal{S}_i = \| \nabla_{W} L \|_1 $
    2. PertImpi (扰动重要性): $ \mathcal{I}_i = \| \nabla_{a} L \odot a \|_1 $ (泰勒一阶展开项)
    """
    def __init__(self, model: nn.Module, target_modules: List[str]):
        self.model = model
        self.target_modules = target_modules
        self.device = next(model.parameters()).device
        
        # 节点元数据存储
        self.node_names = []
        self._map_topological_order()
        self.N = len(self.node_names)
        
        # 张量化指标存储 (GPU 加速)
        self.grad_sensi = torch.zeros(self.N, device=self.device)
        self.pert_impi = torch.zeros(self.N, device=self.device)
        self.hooks = []

    def _map_topological_order(self):
        """建立模型组件的拓扑顺序索引映射。"""
        temp_names = []
        for name, _ in self.model.named_modules():
            if any(name.endswith(t) for t in self.target_modules):
                temp_names.append(name)
        # 按照 Transformer 层级逻辑排序 (i.e. layers.0 -> layers.31)
        self.node_names = sorted(temp_names, key=lambda x: [int(s) for s in x.split('.') if s.isdigit()])
        self.name_to_idx = {name: i for i, name in enumerate(self.node_names)}

    def _hook_factory(self, idx: int):
        """生成支持张量自动归档的 Hook 对。"""
        def forward_hook(module, inputs, outputs):
            # 缓存激活流 a_i
            module.act_cache = outputs[0].detach() if isinstance(outputs, tuple) else outputs.detach()
            
        def backward_hook(module, grad_in, grad_out):
            if not hasattr(module, 'act_cache'): return
            g = grad_out[0].detach() # 捕获局部梯度 \nabla_{a} L
            a = module.act_cache
            
            # 在 GPU 上原地累加
            self.grad_sensi[idx] += g.abs().sum()
            self.pert_impi[idx] += (g * a).abs().sum()
            
        return forward_hook, backward_hook

    def register(self):
        logger.info(f"正在为 {self.N} 个目标组件注册 Mechanistic Hooks...")
        for name, module in self.model.named_modules():
            if name in self.name_to_idx:
                f_hook, b_hook = self._hook_factory(self.name_to_idx[name])
                self.hooks.append(module.register_forward_hook(f_hook))
                self.hooks.append(module.register_full_backward_hook(b_hook))

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()

    def get_metrics(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """导出提取到的张量化归因数据。"""
        return self.grad_sensi.clone(), self.pert_impi.clone(), self.node_names

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from loguru import logger

class IFCCircuitTracer:
    """
    IFCLora 核心电路追踪引擎。
    提取多维指标：Flow (激活流), GradSensi (梯度敏感度), PertImpi (扰动重要性)。
    参考 GraphGhost 设计，并内化核心逻辑。
    """
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.node_metrics = {}
        self.hooks = []
        # 目标组件：Llama 各层的投影矩阵
        self.targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def _get_ifc_hook(self, name):
        """生成捕获激活值与梯度的 Hook。"""
        def forward_hook(module, input, output):
            module.act_cache = output[0].detach() if isinstance(output, tuple) else output.detach()
            
        def backward_hook(module, grad_in, grad_out):
            if not hasattr(module, 'act_cache'): return
            
            grad = grad_out[0].detach()
            act = module.act_cache.detach()
            
            # 1. GradSensi (梯度敏感度): ||grad||_1
            grad_sensi = grad.float().abs().sum().item()
            
            # 2. PertImpi (扰动重要性): 泰勒展开 |grad * act|
            pert_imp = (grad.float() * act.float()).abs().sum().item()
            
            if name not in self.node_metrics:
                self.node_metrics[name] = {"grad_sensi": [], "pert_imp": []}
            self.node_metrics[name]["grad_sensi"].append(grad_sensi)
            self.node_metrics[name]["pert_imp"].append(pert_imp)
            
        return forward_hook, backward_hook

    def register_hooks(self):
        logger.info("正在注册 IFCLora 归因钩子 (Multi-metric Hooks)...")
        for name, module in self.model.named_modules():
            if any(name.endswith(t) for t in self.targets):
                f_hook, b_hook = self._get_ifc_hook(name)
                self.hooks.append(module.register_forward_hook(f_hook))
                self.hooks.append(module.register_full_backward_hook(b_hook))

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def extract_metrics(self, calibration_set: List[str]) -> nx.DiGraph:
        """
        在多样本校准集上提取指标，并构建带权拓扑图。
        """
        self.node_metrics = {}
        self.model.eval()

        for prompt in tqdm(calibration_set, desc="校准多维指标"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            self.model.zero_grad()
            loss.backward()

        # 聚合指标 (Mean Aggregation)
        final_nodes = {}
        for name, data in self.node_metrics.items():
            final_nodes[name] = {
                "grad_sensi": np.mean(data["grad_sensi"]),
                "pert_imp": np.mean(data["pert_imp"])
            }

        # 构建图以计算 Flow (PageRank)
        G = nx.DiGraph()
        import re
        def get_layer_info(n):
            match = re.search(r'layers\.(\d+)', n)
            return int(match.group(1)) if match else -1

        sorted_names = sorted(final_nodes.keys(), key=lambda x: (get_layer_info(x), x))
        
        for i, name in enumerate(sorted_names):
            # 将归因数据存入节点
            G.add_node(name, **final_nodes[name])
            
            # 构建因果边以支撑 PageRank 计算 (Flow)
            for j in range(i + 1, min(i + 10, len(sorted_names))):
                v = sorted_names[j]
                # 边的权重取决于目标节点的重要性，这符合 PageRank 的“含金量”逻辑
                G.add_edge(name, v, weight=final_nodes[v]["pert_imp"])

        logger.success(f"已提取包含 {G.number_of_nodes()} 个节点的 IFCLora 基础图。")
        return G

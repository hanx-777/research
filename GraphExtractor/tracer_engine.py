import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Tuple
from loguru import logger

class CircuitTracer:
    """
    Academic implementation of GraphGhost's circuit tracing logic.
    Captures the 'Implicit Reasoning Graph' by monitoring the attribution flow
    through the transformer modules.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        # Target internal components matching GraphGhost's granularity
        self.targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def _hook_fn(self, name):
        def forward_hook(module, input, output):
            self.activations[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients[name] = grad_out[0].detach()
        return forward_hook, backward_hook

    def register_hooks(self):
        logger.info("Registering attribution hooks on model components...")
        for name, module in self.model.named_modules():
            if any(name.endswith(t) for t in self.targets):
                f_hook, b_hook = self._hook_fn(name)
                self.hooks.append(module.register_forward_hook(f_hook))
                self.hooks.append(module.register_full_backward_hook(b_hook))

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def trace(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """Performs a single attribution pass using |Activation * Gradient|."""
        self.model.zero_grad()
        # Ensure we compute loss for backward
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        attribution = {}
        for name in self.activations:
            if name in self.gradients:
                # Use the Scientific Attribution method
                score = (self.activations[name].float() * self.gradients[name].float()).abs().sum().item()
                attribution[name] = score
        return attribution

class GraphBuilder:
    """
    Constructs the Circuit Graph (Topology) from attribution data.
    Mimics GraphGhost's Graph construction with sequence-aware edges.
    """
    @staticmethod
    def build_circuit(attribution: Dict[str, float], threshold_ratio: float = 0.05) -> nx.DiGraph:
        G = nx.DiGraph()
        
        if not attribution:
            return G

        # 1. Normalize attribution scores
        max_score = max(attribution.values())
        normalized_attr = {k: v / max_score for k, v in attribution.items()}
        
        # 2. Add Nodes (Prune insignificant nodes)
        for name, score in normalized_attr.items():
            if score >= threshold_ratio:
                G.add_node(name, weight=score)

        # 3. Add Causal Edges
        import re
        def get_layer_info(name):
            match = re.search(r'layers\.(\d+)', name)
            layer_idx = int(match.group(1)) if match else -1
            is_mlp = 'mlp' in name
            return layer_idx, is_mlp

        sorted_nodes = sorted(G.nodes(), key=lambda x: (get_layer_info(x), x))
        
        for i, u in enumerate(sorted_nodes):
            u_idx, u_is_mlp = get_layer_info(u)
            
            for j in range(i + 1, min(i + 15, len(sorted_nodes))):
                v = sorted_nodes[j]
                v_idx, v_is_mlp = get_layer_info(v)
                
                # Causal logic: 
                # Same layer: Attention components flow toward MLP components
                # Across layers: Layer N flows to Layer N+1
                edge_weight = normalized_attr[v]
                if v_idx == u_idx:
                    if not u_is_mlp and v_is_mlp:
                        G.add_edge(u, v, weight=edge_weight)
                elif v_idx == u_idx + 1:
                    G.add_edge(u, v, weight=edge_weight * 0.8)
                elif v_idx > u_idx + 1:
                    # Long-range residual influence
                    G.add_edge(u, v, weight=edge_weight * 0.2)

        return G

import torch
import numpy as np
import networkx as nx
import pickle
import re
import os
from typing import Dict, List, Optional, Any
from loguru import logger

class TopologyRankAllocator:
    """
    Topology-Aware Rank Allocator for Sparse Fine-Tuning.
    Analyzes the 'Implicit Reasoning Graphs' extracted via graphghost and 
    computes dynamic LoRA ranks using PageRank Centrality and Soft Scaling.
    """

    def __init__(
        self, 
        graph_path: str, 
        max_rank: int = 64, 
        min_rank: int = 4, 
        beta: float = 2.0,
        target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ):
        """
        Args:
            graph_path: Path to the pickled graph from graphghost.
            max_rank: R_max, the upper bound of rank allocation.
            min_rank: R_min, ensures minimal plasticity for all modules.
            beta: Inverse temperature for Soft Scaling (controls sparsity).
            target_modules: Standard LoRA target modules for Llama-style architectures.
        """
        self.graph_path = graph_path
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.beta = beta
        self.target_modules = target_modules
        self.graph = self._load_graph()

    def _load_graph(self) -> nx.DiGraph:
        logger.info(f"Loading graph from {self.graph_path}...")
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Graph file not found at {self.graph_path}")
            
        with open(self.graph_path, 'rb') as f:
            data = pickle.load(f)
        
        G = nx.DiGraph()
        if isinstance(data, nx.DiGraph):
            G = data
        elif isinstance(data, list):
            # Assume list of (u, v, weight)
            G.add_weighted_edges_from(data)
        elif isinstance(data, dict) and 'edges' in data:
            for edge in data['edges']:
                G.add_edge(edge['src'], edge['dst'], weight=edge.get('importance', edge.get('weight', 1.0)))
        else:
            logger.warning("Unrecognized graph format, attempting to parse as networkx object.")
            G = data
        return G

    def compute_centrality(self) -> Dict[str, float]:
        """Calculates importance scores s_m using PageRank Centrality."""
        logger.info("Computing PageRank Centrality...")
        try:
            # alpha=0.85 is standard; weight='weight' uses attribution scores from graphghost
            scores = nx.pagerank(self.graph, weight='weight', alpha=0.85)
        except Exception as e:
            logger.warning(f"PageRank convergence failed, using degree centrality: {e}")
            scores = nx.degree_centrality(self.graph)
        return scores

    def get_rank_pattern(self) -> Dict[str, int]:
        """
        Implements the Topology-guided Soft Scaling.
        
        Mathematical Logic (Let's think step by step):
        1. Extract importance scores s_m for each module in the reasoning graph.
        2. Normalize scores s_m to [0, 1] range for stability.
        3. Apply Soft Scaling formula:
           r_m = R_min + (R_max - R_min) * [exp(beta * s_norm_m) / exp(beta)]
        """
        scores_dict = self.compute_centrality()
        
        # In Qwen3 extraction, nodes are already full paths like 'model.layers.0.self_attn.q_proj'
        # We only keep nodes that are actually present in our target_modules list or contain layer info
        valid_nodes = [n for n in scores_dict.keys() if "layers" in n.lower()]
        
        if not valid_nodes:
            logger.error("No valid model nodes found in graph. Check node labeling.")
            return {}

        s_vals = np.array([scores_dict[n] for n in valid_nodes])
        # Min-Max normalization
        s_norm = (s_vals - s_vals.min()) / (s_vals.max() - s_vals.min() + 1e-9)
        
        # Soft Scaling logic
        weights = np.exp(self.beta * s_norm)
        weights = weights / weights.max() # Relative scaling
        
        ranks = self.min_rank + (self.max_rank - self.min_rank) * weights
        ranks = np.round(ranks).astype(int)

        rank_pattern = {}
        for idx, node in enumerate(valid_nodes):
            # The node in the graph (e.g., 'model.layers.0.self_attn.q_proj') 
            # might need some suffix cleanup depending on how PEFT expects the pattern.
            # Usually PEFT rank_pattern expects the name relative to the base model.
            
            # If the node is 'model.layers.0.self_attn.q_proj', PEFT pattern can match it.
            # We filter for modules that match our target_modules if necessary.
            if any(target in node for target in self.target_modules):
                # PEFT rank_pattern matches the module name (the part that LoRA is applied to)
                # Example: 'model.layers.0.self_attn.q_proj'
                rank_pattern[node] = int(ranks[idx])
        
        logger.success(f"Generated rank pattern for {len(rank_pattern)} sub-modules.")
        return rank_pattern

    def _map_node_to_hf(self, node_name: str) -> Optional[str]:
        # This function is now deprecated for full-path graphs but kept for backward compatibility
        return node_name

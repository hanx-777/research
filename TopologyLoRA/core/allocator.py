import networkx as nx
import numpy as np
import pickle
import torch
import os
from typing import Dict
from loguru import logger

class TopologyRankAllocator:
    """
    The Core of TopologyLoRA:
    Allocates heterogeneous ranks to LLM layers based on topological centrality
    in the reasoning circuit graph.
    """
    def __init__(self, graph_path: str, max_rank: int = 64, min_rank: int = 4, beta: float = 2.0):
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Reasoning graph not found at {graph_path}. Run GraphExtractor first.")
            
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
            
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.beta = beta

    def get_rank_pattern(self) -> Dict[str, int]:
        """
        Implements PageRank-guided Soft Scaling.
        r_m = R_min + (R_max - R_min) * Normalized(exp(beta * PageRank_score))
        """
        logger.info("Computing topological centrality (PageRank)...")
        # Step 1: Compute Centrality
        scores_dict = nx.pagerank(self.graph, weight='weight', alpha=0.85)
        
        # Step 2: Normalize and Scaled Rank Allocation
        nodes = list(scores_dict.keys())
        s_vals = np.array([scores_dict[n] for n in nodes])
        
        # Numerical normalization [0, 1]
        s_norm = (s_vals - s_vals.min()) / (s_vals.max() - s_vals.min() + 1e-9)
        
        # Soft Scaling using Inverse Temperature beta
        weights = np.exp(self.beta * s_norm)
        weights /= weights.max()
        
        # Calculate dynamic ranks
        raw_ranks = self.min_rank + (self.max_rank - self.min_rank) * weights
        
        rank_pattern = {}
        for idx, node_name in enumerate(nodes):
            # node_name in graph looks like 'model.layers.31.self_attn'
            # We add sub-modules (q, k, v, etc.) to the pattern
            # PEFT will match these paths in the LoRA injection
            final_rank = int(np.round(raw_ranks[idx]))
            
            # Target both Attention and MLP projection layers
            sub_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            for sm in sub_modules:
                full_path = f"{node_name}.{sm}"
                rank_pattern[full_path] = final_rank
                
        logger.success(f"Topological allocation complete. Ranks distributed between {self.min_rank} and {self.max_rank}.")
        return rank_pattern

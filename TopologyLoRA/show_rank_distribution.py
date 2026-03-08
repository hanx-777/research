import yaml
import pickle
import numpy as np
import json
from core.topology_allocator import TopologyRankAllocator
from collections import Counter
import os

def main():
    # Load config to get same params as training
    config_path = "config/qwen3_gsm8k.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"--- TopologyLoRA Rank Distribution Analysis ---")
    print(f"Graph: ../reasoning_circuit_robust.pkl")
    print("-" * 50)

    allocator = TopologyRankAllocator(
        graph_path="../reasoning_circuit_robust.pkl",
        max_rank=cfg['max_rank'],
        min_rank=cfg['min_rank'],
        beta=cfg['beta'],
        target_modules=cfg['target_modules']
    )
    
    rank_pattern = allocator.get_rank_pattern()
    
    if not rank_pattern:
        print("Error: No rank pattern generated.")
        return

    # 1. Statistics Summary
    ranks = list(rank_pattern.values())
    rank_counts = Counter(ranks)
    sorted_ranks = sorted(rank_counts.items(), key=lambda x: x[0], reverse=True)
    
    # 2. Top Hubs
    top_modules = sorted(rank_pattern.items(), key=lambda x: x[1], reverse=True)[:10]

    # Save to TXT for easy reading
    with open("rank_distribution.txt", "w", encoding="utf-8") as f:
        f.write("--- TopologyLoRA Rank Distribution ---\n")
        f.write(f"Graph: ../reasoning_circuit_robust.pkl\n\n")
        f.write("[Rank Counts]\n")
        for rank, count in sorted_ranks:
            f.write(f"Rank {rank:2d}: {count:3d} modules\n")
        
        f.write("\n[Top 10 Reasoning Hubs]\n")
        for module, rank in top_modules:
            f.write(f" - {module}: Rank {rank}\n")
    
    # Save to JSON for machine reading
    with open("rank_distribution.json", "w", encoding="utf-8") as f:
        json.dump(rank_pattern, f, indent=4)

    print(f"Success!")
    print(f" - TXT distribution saved to: rank_distribution.txt")
    print(f" - JSON full mapping saved to: rank_distribution.json")

if __name__ == "__main__":
    main()

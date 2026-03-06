import os
import sys
import torch
import pickle
import argparse
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# 动态添加 GraphGhost-main 到路径，确保能调用其 circuit_tracer 模块
sys.path.append(os.path.abspath("../GraphGhost-main"))

try:
    from circuit_tracer.attribution import Tracer
    from circuit_tracer.graph import CircuitGraph
except ImportError:
    logger.error("Failed to import circuit_tracer. Ensure GraphGhost-main is in the same parent directory.")
    sys.exit(1)

def extract_reasoning_graph(model_id: str, prompt: str, output_path: str):
    """
    Extends GraphGhost's Tracer to extract an implicit reasoning graph 
    from a specific inference pass.
    """
    logger.info(f"Loading model {model_id} for topology extraction...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 1. Initialize GraphGhost Tracer
    # Let's think step by step:
    # First, wrap the model in a Tracer that tracks causal influence across layers.
    tracer = Tracer(model, tokenizer)

    logger.info(f"Tracing reasoning path for input: {prompt[:50]}...")
    
    # 2. Run Attribution (Forward + Backward Pass for Sensitivity Analysis)
    # This identifies which layers are most 'active' for this specific logic.
    with tracer.trace(prompt) as tr:
        # We target the last token's logit as our 'objective' for the reasoning graph
        logits = tr.output.logits
        target_score = logits[0, -1, :].max()
        target_score.backward()

    # 3. Construct Graph from Tracer's state
    # Each node represents a transformer block (Attn/MLP), 
    # and edges represent the flow of information with importance weights.
    circuit_graph = tracer.generate_graph(threshold=0.01) # Ignore noise below threshold
    
    logger.success(f"Extracted reasoning graph with {len(circuit_graph.nodes)} nodes.")

    # 4. Save to PKL for TopologyLoRA
    # We save it in a format TopologyRankAllocator expects (NetworkX DiGraph)
    with open(output_path, 'wb') as f:
        pickle.dump(circuit_graph.to_networkx(), f)
    
    logger.info(f"Implicit reasoning graph saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--prompt", type=str, required=True, help="A representative task prompt (e.g., from GSM8K)")
    parser.add_argument("--output", type=str, default="../gsm8k_graph.pkl")
    args = parser.parse_args()

    extract_reasoning_graph(args.model_id, args.prompt, args.output)

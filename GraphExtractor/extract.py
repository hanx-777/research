import torch
import pickle
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tracer_engine import CircuitTracer, GraphBuilder
from loguru import logger

def run_scientific_extraction(model_id: str, calibration_set: list, output_file: str, threshold: float):
    """
    Performs multi-sample attribution to extract a robust 'Common Circuit'.
    This avoids prompt-specific bias and noise.
    """
    logger.info(f"Loading {model_id} for robust circuit extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )

    tracer = CircuitTracer(model)
    tracer.register_hooks()
    
    # List to store attribution results for each sample
    sample_attributions = []

    logger.info(f"Calibrating on {len(calibration_set)} diverse reasoning prompts...")
    
    for i, prompt in enumerate(calibration_set):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        try:
            # Perform Taylor-based attribution pass
            attr_data = tracer.trace(inputs['input_ids'])
            sample_attributions.append(attr_data)
            logger.info(f"Calibration Progress: {i+1}/{len(calibration_set)}")
        except Exception as e:
            logger.error(f"Error processing prompt {i}: {e}")

    # Aggregation: Compute the expected importance (Mean) across the entire calibration set
    # Let's think step by step:
    # Aggregating ensures we find the 'Reasoning Backbone' of the model.
    logger.info("Aggregating scores into a robust attribution map...")
    final_attr = {}
    if not sample_attributions:
        logger.error("No attribution data collected. Exiting.")
        return

    all_keys = sample_attributions[0].keys()
    for key in all_keys:
        # Mean across samples
        final_attr[key] = np.mean([attr[key] for attr in sample_attributions])
    
    # Build the final pruned graph
    logger.info(f"Building circuit with saliency threshold: {threshold}...")
    builder = GraphBuilder()
    graph = builder.build_circuit(final_attr, threshold_ratio=threshold)
    
    # Save the robust topology
    with open(output_file, 'wb') as f:
        pickle.dump(graph, f)
        
    logger.success(f"Successfully extracted robust reasoning circuit with {graph.number_of_nodes()} components.")
    tracer.clear_hooks()

if __name__ == "__main__":
    # Rigorous Calibration Set: Diverse mathematical reasoning tasks
    reasoning_tasks = [
        "Question: Natalia sold clips to 48 friends. Each friend bought 2 clips. How many clips did she sell? Answer:",
        "Question: If x + 5 = 12, what is x? Answer:",
        "Question: A group of 20 people needs 5 cars. How many people per car? Answer:",
        "Question: The price of a toy is $10. With a 20% discount, what is the price? Answer:",
        "Question: A train travels at 60km/h, how far does it go in 3.5 hours? Answer:",
        "Question: If a rectangle has a length of 10 and width of 5, what is the area? Answer:",
        "Question: 15 plus 27 minus 10 equals what? Answer:",
        "Question: If John has 3 apples and eats 1, how many are left? Answer:"
    ]

    run_scientific_extraction(
        model_id=r"D:\code\model\Qwen3-1.7B",
        calibration_set=reasoning_tasks,
        output_file="../reasoning_circuit_robust.pkl",
        threshold=0.05
    )

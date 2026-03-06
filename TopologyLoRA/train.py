import argparse
import yaml
import torch
import os
from loguru import logger

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from trl import SFTTrainer

from core.topology_allocator import TopologyRankAllocator
from data.gsm8k_loader import get_gsm8k_dataset

def main():
    parser = argparse.ArgumentParser(description="TopologyLoRA Fine-Tuning Entry Point")
    parser.add_argument("--config", type=str, default="config/llama3_gsm8k.yaml")
    parser.add_argument("--graph", type=str, required=True, help="Path to graphghost reasoning graph (.pkl)")
    args = parser.parse_args()

    # Load Hyperparameters
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Initializing Experiment: {cfg['exp_name']}")

    # 1. Topological Rank Allocation Logic
    # Let's think step by step:
    # First, analyze the graph from graphghost to find bottlenecks.
    allocator = TopologyRankAllocator(
        graph_path=args.graph,
        max_rank=cfg['max_rank'],
        min_rank=cfg['min_rank'],
        beta=cfg['beta'],
        target_modules=cfg['target_modules']
    )
    rank_pattern = allocator.get_rank_pattern()

    # 2. Setup Model & Tokenizer
    model_id = cfg['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading {model_id} in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # 3. Apply TopologyLoRA
    # Using 'rank_pattern' to assign dynamic ranks to specific modules based on topology
    peft_config = LoraConfig(
        r=cfg['min_rank'], # Default rank for any module not in pattern
        lora_alpha=cfg['lora_alpha'],
        target_modules=cfg['target_modules'],
        rank_pattern=rank_pattern,
        lora_dropout=cfg.get('dropout', 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Prepare Dataset
    dataset = get_gsm8k_dataset(tokenizer)

    # 5. Launch Training
    output_dir = os.path.join("./results", cfg['exp_name'])
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['grad_acc'],
        learning_rate=float(cfg['lr']),
        num_train_epochs=cfg['epochs'],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        lr_scheduler_type="cosine",
        warmup_steps=100
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
    )

    logger.info("Starting TopologyLoRA fine-tuning loop...")
    trainer.train()
    
    # Save Artifacts
    model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    logger.success("Training complete. Adapter saved.")

if __name__ == "__main__":
    main()

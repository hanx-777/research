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

# 重构后的模块导入
from GraphExtractor.tracer import MechanisticTracer
from GraphExtractor.builder import ImplicitGraphBuilder
from IFCLora.core.ifc_calculator import IFCCalculator
from IFCLora.core.allocator import IFCRankAllocator
from IFCLora.data.gsm8k_loader import get_gsm8k_dataset

def main():
    parser = argparse.ArgumentParser(description="IFCLora: Information-Flow Centrality Based LoRA")
    parser.add_argument("--config", type=str, default="config/llama3_ifc.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    logger.info(f"正在启动 IFCLora 实验流程: {cfg['exp_name']}")

    # 1. 基础模型加载 (用于电路追踪)
    model_id = cfg['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("正在加载基础模型以执行电路追踪探测...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # 2. IFCLora 探测阶段
    # 第一步: 注册 Mechanistic Hooks
    tracer = MechanisticTracer(model, cfg['target_modules'])
    tracer.register()
    
    # 第二步: 执行多样本校准提取归因指标
    logger.info("执行多样本校准 (Calibration) 以提取稳健指标...")
    calibration_prompts = [
        "Question: If x + 5 = 12, what is x? Answer:",
        "Question: A train travels at 60km/h for 3 hours. How far does it go? Answer:",
        "Question: Natalie has 48 clips. She sells half. How many left? Answer:"
    ]
    for prompt in calibration_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        model(**inputs, labels=inputs["input_ids"]).loss.backward()
    
    # 第三步: 导出指标并移除 Hooks
    grad_v, pert_v, node_names = tracer.get_metrics()
    tracer.remove()
    
    # 3. IFC 拓扑求解阶段
    # 构建张量图
    builder = ImplicitGraphBuilder(len(node_names), model.device)
    A = builder.build_tensors(pert_v)
    
    # 求解 IFC
    calculator = IFCCalculator(alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'])
    ifc_scores = calculator.calculate(A, grad_v, pert_v)
    
    # 4. 秩分配阶段
    allocator = IFCRankAllocator(max_rank=cfg['max_rank'], min_rank=cfg['min_rank'], tau=cfg['scaling_beta'])
    rank_pattern = allocator.allocate(ifc_scores, node_names)

    # 5. 注入与微调阶段
    # 重新初始化模型以准备微调 (注入个性化 Rank)
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=cfg['min_rank'], 
        lora_alpha=cfg['lora_alpha'],
        target_modules=cfg['target_modules'],
        rank_pattern=rank_pattern, # 注入 IFCLora 拓扑
        lora_dropout=cfg.get('dropout', 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 数据准备与训练
    dataset = get_gsm8k_dataset(tokenizer)
    training_args = TrainingArguments(
        output_dir=os.path.join("./results", cfg['exp_name']),
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['grad_acc'],
        learning_rate=float(cfg['lr']),
        num_train_epochs=cfg['epochs'],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
    )

    logger.info("IFCLora 微调启动...")
    trainer.train()

if __name__ == "__main__":
    main()

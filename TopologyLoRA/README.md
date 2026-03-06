# TopologyLoRA: Topology-Aware Sparse Fine-Tuning

Official implementation of **TopologyLoRA**, a framework that allocates LoRA ranks based on the **Implicit Reasoning Graphs** extracted from Large Language Models.

## Core Mechanism
TopologyLoRA utilizes the causal topology of the model to identify "bottleneck" circuits. Modules with high **PageRank Centrality** in the reasoning graph (extracted via `graphghost`) receive higher ranks ($r=64$), while peripheral modules are assigned minimal ranks ($r=4$).

## Setup
```bash
pip install -r requirements.txt
```

## Workflow
1. **Extract Reasoning Graph**:
   Run the graph extraction script from `graphghost` on your target task (e.g., GSM8K).
   
2. **Train TopologyLoRA**:
   ```bash
   python train.py --graph /path/to/reasoning_graph.pkl --config config/llama3_gsm8k.yaml
   ```

## Project Structure
- `core/`: Implements the `TopologyRankAllocator` and Soft Scaling logic.
- `data/`: Dataset loaders for GSM8K.
- `config/`: YAML configurations for experiments.
- `train.py`: Main training script integrating PEFT and TopologyLoRA.

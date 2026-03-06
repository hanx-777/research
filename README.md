# TopologyLoRA: Topology-Aware Sparse Fine-Tuning via Implicit Reasoning Graphs

Official implementation of **TopologyLoRA**, a parameter-efficient fine-tuning (PEFT) framework that dynamically allocates LoRA ranks based on the model's internal causal topology.

## 核心理念 (Core Philosophy)
传统 LoRA 在所有层中均匀分配秩（Rank），这忽略了预训练模型内部不同组件在特定任务（如数学推理）中的贡献差异。
**TopologyLoRA** 提出了一个两阶段流程：
1. **拓扑提取 (Extraction)**：通过因果归因分析，提取模型在处理任务时的“隐式推理图（Implicit Reasoning Graph）”。
2. **动态分配 (Allocation)**：利用图论中心性算法（PageRank），识别推理链路中的“瓶颈模块”，并为其分配更高的可塑性（Higher Ranks）。

---

## 项目结构 (Project Structure)

### 1. GraphExtractor (拓扑提取器)
该模块负责“透视”预训练模型，识别其推理电路。
- **`tracer_engine.py`**: 核心归因引擎。通过在 Transformer 组件（Q, K, V, MLP）上注册 Hooks，捕捉激活值与梯度，计算归因分数：$Attribution = |Activation \times Gradient|$。
- **`extract.py`**: 提取脚本入口。给它一个推理样本，它会生成一个代表模型思考路径的 `.pkl` 图文件。

### 2. TopologyLoRA (核心训练器)
该模块负责基于拓扑结构执行高效微调。
- **`core/allocator.py`**: 实现 PageRank 中心性计算与 **Soft Scaling** 映射。它将图节点的重要性转化为 LoRA 的 Rank：
  $$r_m = \text{Round}\left( R_{min} + (R_{max} - R_{min}) \cdot \frac{e^{\beta s_m}}{\max(e^{\beta S})} \right)$$
- **`train.py`**: 集成训练入口。利用 HuggingFace `peft` 的 `rank_pattern` 特性，注入异构 Rank 并启动 GSM8K 微调。
- **`data/gsm8k_loader.py`**: 针对数学推理任务的专用数据流处理。

---

## 安装与环境 (Installation)

本项目完全自包含，不依赖外部库。建议使用 Python 3.9+ 与 CUDA 环境。

```bash
pip install torch transformers peft accelerate networkx numpy loguru pyyaml trl
```

---

## 实验流程 (Experimental Workflow)

### 阶段一：提取隐式推理图
首先，我们需要确定模型在进行逻辑推理时，哪些组件是“关键枢纽”。

```bash
cd GraphExtractor
python extract.py \
    --model meta-llama/Meta-Llama-3-8B \
    --  
    --output ../reasoning_circuit.pkl
```
*这一步将生成 `reasoning_circuit.pkl`，它包含了上百个组件的因果关联。*

### 阶段二：启动拓扑感知微调
利用提取出的拓扑图，为核心组件分配高 Rank（如 64），为辅助组件保留低 Rank（如 4）。

```bash
cd ../TopologyLoRA
python train.py \
    --graph ../reasoning_circuit.pkl \
    --config config/llama3_gsm8k.yaml
```

---

## 数学细节 (Mathematical Rationale)
- **PageRank Centrality**: 我们不仅看某个层的归因分（Attribution Score），还通过 PageRank 分析该层在整个计算流中的全局影响力。
- **Soft Scaling**: 通过逆温度参数 $\beta$ 控制秩分配的稀疏度。较大的 $\beta$ 会使参数预算更集中地投入到推理“枢纽”中，从而在极低的平均 Rank 下保持极强的推理性能。

## 引用 (Citation)
如果您在学术研究中使用了本项目，请引用我们的工作：
```bibtex
@article{topology_lora2026,
  title={TopologyLoRA: Topology-Aware Sparse Fine-Tuning via Implicit Reasoning Graphs},
  author={Your Name},
  journal={Academic Conference},
  year={2026}
}
```

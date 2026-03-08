# TopologyLoRA: 拓扑感知稀疏微调框架 (Topology-Aware Sparse FT)

本项目实现了 **TopologyLoRA**：一种基于大模型内部“隐式推理图”动态分配 LoRA 秩（Rank）的微调框架。它能识别出模型中的“核心枢纽”模块并分配高 Rank，而对边缘模块分配低 Rank。

---

## 🚀 快速启动 (以 Qwen3-1.7B 为例)

### 1. 环境准备
确保你已激活 `dt_lora` 或包含以下库的环境：
```bash
pip install torch transformers peft trl networkx loguru pyyaml datasets
```

### 2. 运行微调实验
进入目录并指定拓扑图路径与配置文件：
```powershell
cd TopologyLoRA
python train.py --graph ../reasoning_circuit_robust.pkl --config config/qwen3_gsm8k.yaml
```

---

## 🛠️ 如何切换新模型 (换头指南)

要将 TopologyLoRA 适配到你的新模型（例如 DeepSeek, Llama-2 等），只需 3 步：

### 第一步：提取拓扑图 (Graph Extraction)
你需要先运行 `GraphExtractor` 中的脚本来分析新模型的推理路径：
```bash
cd GraphExtractor
python extract.py --model /your/local/model_path --output ../my_new_model_graph.pkl
```
*这一步会生成节点的因果关联图。*

### 第二步：创建配置文件
在 `config/` 目录下创建一个 YAML 文件（如 `my_model.yaml`）：
```yaml
exp_name: "my_model_experiment"
model_id: "D:/path/to/your/model"  # 填入本地路径或 HF ID
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 确认模型的 Linear 层名称

# 拓扑参数 (核心控制)
max_rank: 64    # 枢纽节点的最大 Rank
min_rank: 4     # 边缘节点的保底 Rank
beta: 2.5       # 稀疏度控制：数值越大，参数分配越集中在极少数核心节点上
```

### 第三步：运行并观察分布
在正式训练前，你可以使用我为你准备的分析工具查看 Rank 是如何分布的：
```bash
python show_rank_distribution.py --config config/my_model.yaml --graph ../my_new_model_graph.pkl
```

---

## 📂 项目结构说明
- `core/topology_allocator.py`: **核心大脑**。负责读取 `.pkl` 图文件，计算 PageRank 中心性，并执行 Soft Scaling 分配 Rank。
- `train.py`: **训练入口**。集成 PEFT 的 `rank_pattern` 功能，注入异构 Rank。
- `show_rank_distribution.py`: **可视化工具**。在训练前帮你预览哪些层是“推理核心”。
- `data/gsm8k_loader.py`: 针对数学推理任务的数据预处理逻辑。

---

## 📈 进阶调优技巧
- **想让模型更“聪明”？** 调大 `max_rank`（如 128），让核心层有更多可塑性。
- **想让模型更“轻量”？** 调大 `beta`（如 5.0），让 90% 的层都处于 `min_rank` 状态。
- **层名不匹配？** 检查 `.pkl` 里的节点名称，并在 `target_modules` 中包含它们。目前代码已支持全路径自动匹配。

---
*Created with ❤️ by Gemini CLI & TopologyLoRA Team.*

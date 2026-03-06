# TopologyLoRA 项目核心代码技术分析报告

本报告针对 `GraphExtractor/` 和 `TopologyLoRA/` 两个核心模块的工程实现进行深度剖析。这两个模块共同构成了一个从**机械可解释性（归因分析）**到**参数高效微调（秩分配）**的闭环系统。

---

## 模块一：GraphExtractor —— 拓扑电路探测器

该模块的目标是“解剖”预训练模型在执行特定逻辑（如数学推理）时的内部激活路径。

### 1.1 `tracer_engine.py`：归因追踪核心
该文件实现了基于 PyTorch Hooks 的非侵入式探测：
- **组件级追踪**：代码通过 `targets` 列表精准锁定 Llama 架构中的 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`。这比传统的“按层（Layer-wise）”分析更细致，达到了矩阵级分辨率。
- **泰勒归因算法**：实现了一阶泰勒展开归因。
  - **前向 Hook**：捕获并缓存激活值（Activation）。
  - **后向 Hook**：捕获梯度（Gradient）。
  - **重要性计算**：`score = |Activation * Gradient|`。在数学上，这代表了该矩阵对损失函数 $L$ 的一阶敏感度，即该参数发生扰动时对输出的影响力。
- **因果图构建**：
  - **节点**：每个矩阵是一个节点，属性为归因得分。
  - **边**：模拟 Transformer 内部的顺序流和残差流。逻辑是：同一层内 Attention 流向 MLP，相邻层之间按顺序连接，同时保留跳连（Skip-connection）的权重衰减连接。

### 1.2 `extract.py`：稳健性提取入口
- **多样本校准 (Calibration)**：该脚本不依赖单一 Prompt。它通过遍历一组多样化的推理任务（如加减乘除、面积计算等），收集所有样本的归因分。
- **均值平滑**：通过对多样本得分取均值（Mean Aggregation），代码滤除了特定词汇带来的噪声，提取出模型处理“一类逻辑”时的稳定骨架。
- **序列化输出**：最终将 `networkx.DiGraph` 对象保存为 `.pkl` 文件，作为下一阶段的输入。

---

## 模块二：TopologyLoRA —— 拓扑感知训练引擎

该模块的目标是基于提取出的图结构，实现参数预算的“精准扶贫”。

### 2.1 `core/allocator.py`：动态秩分配算法
这是本项目最核心的数学转换层：
- **PageRank 中心性**：代码调用了 PageRank 算法。它的工程意义在于：一个矩阵的重要程度不仅取决于它自身的归因得分，还取决于它在推理路径中是否处于“咽喉要道”。如果多个重要的推理步骤都必须经过某个矩阵，该矩阵的 PageRank 分值会极高。
- **Soft Scaling (软缩放) 映射**：
  - 代码引入了逆温度参数 $\beta$。
  - 通过公式 `r = min_rank + (max_rank - min_rank) * [exp(beta * s) / max(exp(beta * S))]` 将拓扑得分转化为整数 Rank。
  - 这保证了核心节点可以获得高达 64 的 Rank，而边缘节点仅保留 4 的 Rank，极大地节省了参数预算。

### 2.2 `train.py`：异构秩注入与训练
- **PEFT 深度集成**：代码利用了 HuggingFace `peft` 库中一个较少被提及但极其强大的特性：`rank_pattern`。
  - 通常 LoRA 只能设置一个全局 $r$。
  - 本代码生成的 `rank_pattern` 是一个复杂的字典，包含了数百个组件及其对应的个性化 Rank。
- **训练流程**：
  - 加载模型并执行 `prepare_model_for_kbit_training`（支持量化加速）。
  - 使用 `SFTTrainer` 启动微调。
  - 在 GSM8K 数据集上执行标准的 Causal LM 训练逻辑。

### 2.3 `data/gsm8k_loader.py`：推理任务适配
- **标签掩码 (Label Masking)**：代码在预处理时，将 Prompt（问题）部分的标签设为 `-100`。
- **意义**：这确保了在微调过程中，模型的梯度只由“生成正确答案”产生，而不会浪费在“复读问题”上，进一步提高了训练的针对性。

---

## 总结：数据流向图

1. **输入**：预训练 Llama-3 + 校准题目集。
2. **提取 (GraphExtractor)**：通过 Hook 捕获 `(Act, Grad)` -> 计算泰勒归因 -> 构建拓扑图 `.pkl`。
3. **分析 (TopologyLoRA/core)**：加载 `.pkl` -> 运行 PageRank -> 执行 Soft Scaling -> 生成动态 Rank 字典。
4. **微调 (TopologyLoRA/train)**：注入 `rank_pattern` -> 启动异构 LoRA 训练 -> 产出拓扑感知的微调 Adapter。

---

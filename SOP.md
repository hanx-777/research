# IFCLoRA 实验运行标准作业程序 (SOP)

本 SOP 旨在指导研究者如何利用 **IFCLoRA** 框架完成“探测-分析-微调”的完整闭环。

---

## 阶段 I：环境准备与基础配置
1.  **环境检查**：
    - 确保 GPU 驱动支持 BF16 或 FP16 精度。
    - 建议使用 Llama-3-8B 或 Qwen-2.5-7B 进行实验。
2.  **超参数设定**：
    - 修改 `IFCLora/config/llama3_ifc.yaml`：
      - `alpha`: 设置信息流权重（建议 0.4）。
      - `beta`: 设置梯度敏感度权重（建议 0.3）。
      - `gamma`: 设置扰动重要性权重（建议 0.3）。
      - `scaling_beta` ($\tau$): 逆温度系数。若希望 Rank 分配更稀疏（更聚焦），将其调大（例如 3.0 或更高）。

---

## 阶段 II：隐式图电路追踪 (Mechanistic Tracing)
1.  **目的**：获取模型在处理特定逻辑任务（如数学推理）时的“计算枢纽”分布。
2.  **核心过程**：
    - `train.py` 会首先调用 `MechanisticTracer` 注册 Hooks。
    - 脚本会自动使用内置的 `calibration_prompts`（校准提示词集）执行前向与反向传播。
    - **验证点**：查看日志输出中是否显示 `正在为 N 个目标组件注册 Mechanistic Hooks`。

---

## 阶段 III：IFC 指标求解与秩分配 (Analysis & Allocation)
1.  **目的**：将探测到的原始数据转化为具体的参数预算分配方案。
2.  **核心过程**：
    - `ImplicitGraphBuilder` 构建张量化的邻接矩阵 $\mathcal{A}$。
    - `IFCCalculator` 在 GPU 上执行 PageRank 求解，并根据融合公式计算最终的 IFC 分数。
    - `IFCRankAllocator` 根据 $IFC$ 得分，为数百个投影矩阵生成 `rank_pattern` 字典。
3.  **验证点**：
    - 检查日志输出的 `平均秩 (Avg Rank)`。确保其接近您的实验预算（例如 $r=8$ 附近）。

---

## 阶段 IV：拓扑感知微调 (SFT Training)
1.  **目的**：在分配好的非对称 Rank 结构上执行下游任务微调。
2.  **核心过程**：
    - `peft` 库将根据 `rank_pattern` 注入不同秩的低秩矩阵。
    - 启动 `SFTTrainer` 在 GSM8K 或其他数据集上进行微调。
3.  **验证点**：
    - 调用 `model.print_trainable_parameters()`，确认可训练参数量符合预期。
    - 观察训练 Loss 是否稳定下降。

---

## 阶段 V：结果分析与消融实验 (Analysis & Ablation)
1.  **消融实验推荐**：
    - **w/o Flow**: 设置 `alpha=0`，观察仅依赖梯度指标的效果。
    - **Uniform LoRA**: 将 $\tau$ 设为极小值（如 0.001），对比均匀分配的基线。
2.  **权重分析**：
    - 观察微调后的 `adapter_config.json`，分析哪些层（通常是中后部层）获得了更高的 Rank，这对于论文撰写中的“实验结果分析”章节至关重要。

---
## 常见问题排查 (Troubleshooting)
- **显存溢出 (OOM)**：如果探测阶段 OOM，请减小 `calibration_prompts` 的长度，或开启 `bitsandbytes` 4-bit 量化。
- **PageRank 未收敛**：如果日志显示 `PageRank 达到最大迭代次数`，请检查邻接矩阵是否全零（通常是因为 GradSensi 太小）。

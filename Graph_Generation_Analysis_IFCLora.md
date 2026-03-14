# IFCLora: 隐式推理图生成机制与结构深度解析

本报告详细描述了 `GraphExtractor` 模块如何从预训练模型中提取“隐式推理图”，以及该图的内部数据结构。

---

## 1. 生成机制：四阶段归因追踪流程

`GraphExtractor` 采用非侵入式的 **探测钩子 (Probing Hooks)** 技术，在不修改模型参数的前提下，动态捕获模型在处理逻辑推理任务时的内部状态。

### 阶段 I：组件级钩子注册 (Granular Hooking)
- **目标识别**：算法精准定位 Transformer 架构中的核心权重投影矩阵（Nodes）。
  - **Attention 子层**：`q_proj`, `k_proj`, `v_proj`, `o_proj`。
  - **MLP 子层**：`gate_proj`, `up_proj`, `down_proj`。
- **双向捕获**：为每个组件注册 `Forward Hook`（捕获激活值 $a$）和 `Backward Hook`（捕获梯度 $g$）。

### 阶段 II：多维归因采样 (Multi-dimensional Attribution)
当校准样本（Calibration Prompts）输入模型时，执行以下数学计算：
1. **梯度敏感度 (Gradient Sensitivity, $GradSensi$)**：
   $$GradSensi = \sum |g|$$
   衡量损失函数对该组件权重的变化有多敏感。
2. **扰动重要性 (Perturbation Importance, $PertImpi$)**：
   基于一阶泰勒展开近似：
   $$PertImpi = \sum |g \odot a|$$
   衡量如果将该组件的激活值置零（扰动），对最终推理结果造成的偏差程度。

### 阶段 III：时间步与样本聚合 (Temporal & Sample Aggregation)
- 为了消除单次推理的随机噪声，`GraphExtractor` 在多个推理样本上运行。
- 对每个组件在所有样本上的得分取算术平均值（Mean Aggregation），提取出稳定的**推理主干 (Reasoning Backbone)**。

### 阶段 IV：因果拓扑构建 (Causal Topology Construction)
- **节点映射**：每个被追踪的矩阵被转化为图中的一个节点 $\mathcal{V}$。
- **因果流向逻辑**：
  - **同层内 (Intra-layer)**：建立从 Attention 组件到 MLP 组件的有向边。
  - **跨层间 (Inter-layer)**：建立从第 $N$ 层组件到第 $N+1$ 层组件的顺序边。
  - **残差跳连 (Residual Stream)**：建立跨越 2-3 层的加权边，模拟 Transformer 内部的残差流量。

---

## 2. 图的内部结构 (Graph Anatomy)

生成的 `ifc_graph.pkl` 文件是一个 **有向加权图 (Directed Weighted Graph)**，其底层由 `NetworkX` 驱动。

### 2.1 节点属性 (Node Attributes)
每个节点代表模型中的一个具体权重矩阵（如 `layers.12.self_attn.v_proj`），包含以下元数据：
| 属性名 | 数据类型 | 描述 |
| :--- | :--- | :--- |
| `grad_sensi` | float | 该组件在该任务下的平均梯度敏感度得分。 |
| `pert_imp` | float | 该组件的一阶泰勒扰动重要性得分。 |

### 2.2 边属性 (Edge Attributes)
边代表信息在网络内部的因果流动：
- **权重 ($W_{edge}$)**：由目标节点的重要性决定（即 $W_{edge} = PertImpi_{target}$）。
- **逻辑含义**：如果节点 A 连接到节点 B，意味着信息从 A 流向 B，且该路径的“带宽”由 B 在推理任务中的重要性决定。这为后续 **PageRank (Information Flow)** 算法提供了计算基础。

---

## 3. 数学支撑：从图到 IFC 得分

`GraphExtractor` 生成的这种结构是为了支撑论文中的核心 **IFC 指标** 计算：

1. **Flow (信息流中心性)**：通过在图上运行 PageRank 算法，计算节点的拓扑重要性。
2. **混合决策**：
   $$IFC_i = \alpha \cdot Flow(\mathcal{G}) + \beta \cdot GradSensi_i + \gamma \cdot PertImpi_i$$

---

## 4. 总结：图的特征
- **高分辨率**：不再以“层”为单位，而是以“矩阵”为单位。
- **任务相关性**：图的拓扑结构会随着任务（如数学推理 vs 创意写作）的不同而动态变化。
- **科学性**：通过残差边设计，它真实还原了 Transformer 内部非线性的信息处理逻辑。

# IFCLoRA: Information-Flow Centrality Based Topology-Aware LoRA

Official implementation of **IFCLoRA**, a parameter-efficient fine-tuning (PEFT) framework that redefines rank allocation through the lens of **Mechanistic Interpretability** and **Graph Topology**.

## 1. Abstract & Motivation
Traditional LoRA assigns a uniform rank $r$ to all layers, ignoring the functional heterogeneity within Transformer architectures. **IFCLoRA** addresses this "Uniformity Paradox" by:
1.  **Circuit Tracing**: Identifying the model's "Implicit Reasoning Graph" using First-order Taylor Attribution.
2.  **Topological Centrality**: Computing the **Information-Flow Centrality (IFC)** via GPU-accelerated Power Iterations.
3.  **Adaptive Allocation**: Dynamically distributing rank budgets based on module-wise IFC scores, ensuring that "computational plasticity" is concentrated on logical bottlenecks.

---

## 2. Project Architecture
The project is decoupled into two high-cohesion modules:

### 📡 GraphExtractor (Mechanistic Probing)
- `tracer.py`: Manages high-performance GPU hooks to capture $GradSensi$ ($\|\nabla L\|_1$) and $PertImpi$ ($\|\nabla L \odot a\|_1$).
- `builder.py`: Constructs the tensorized adjacency matrix $\mathcal{A}$ based on the Transformer's causal residual flow.

### 🧠 IFCLoRA (Topology-Aware PEFT)
- `core/ifc_calculator.py`: Implements the multi-metric IFC solver:
  $$ IFC_i = \alpha \cdot Flow_i + \beta \cdot GradSensi_i + \gamma \cdot PertImpi_i $$
- `core/allocator.py`: Employs a **Soft Scaling** scheduler to map continuous IFC scores to discrete ranks:
  $$ r_i = R_{min} + (R_{max} - R_{min}) \cdot \frac{\exp(\tau \cdot IFC_i)}{\max(\exp(\tau \cdot IFC))} $$
- `train.py`: The end-to-end integration pipeline.

---

## 3. Mathematical Rationale
- **Information Flow ($Flow_i$)**: We model the LLM as a directed graph where information propagates from early layers to the logit node. $Flow$ is calculated as the steady-state probability (PageRank) of this graph.
- **Sparsity Control ($\tau$)**: The inverse temperature parameter $\tau$ determines the concentration of ranks. A higher $\tau$ leads to sparser, more bottleneck-focused allocation.

## 4. Quick Start
### Installation
```bash
pip install torch transformers peft accelerate networkx loguru pyyaml trl
```

### Execution
1.  **Extract Graph & Train**:
    ```bash
    python IFCLora/train.py --config IFCLora/config/llama3_ifc.yaml
    ```

---

## 5. Citation
If you find this work useful, please cite our technical report:
```bibtex
@article{ifclora2026,
  title={IFCLoRA: Topology-Aware Sparse Fine-Tuning via Information-Flow Centrality},
  author={[Your Name]},
  year={2026}
}
```

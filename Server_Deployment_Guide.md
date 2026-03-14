# IFCLoRA A800 服务器部署与实操指南 (Server Deployment & Troubleshooting Guide)

本指南专门针对在 NVIDIA A800 (80GB) 高性能计算集群上部署与运行 **IFCLoRA** 实验而设计。

---

## 一、 环境部署 (Environment Setup)

在 A800 服务器上，建议使用 `Conda` 构建独立的虚拟环境，以确保底层 CUDA 库与 Python 依赖的兼容性。

### 1. 创建虚拟环境与安装依赖
```bash
# 创建并激活环境
conda create -n ifclora python=3.10 -y
conda activate ifclora

# 安装核心依赖 (推荐使用国内镜像源加速)
pip install torch transformers peft accelerate networkx loguru pyyaml trl datasets bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Hugging Face 模型权限验证
Llama-3 是受限模型（Gated Model），请确保已完成以下步骤：
1.  **官网申请**：前往 [Meta-Llama-3-8B 页面](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 提交 Access 申请（通常几分钟内通过）。
2.  **CLI 登录**：在服务器终端执行：
    ```bash
    huggingface-cli login
    # 粘贴你的 Hugging Face Access Token (在 Settings -> Access Tokens 生成)
    ```

**国内服务器加速提示**：
若 A800 节点无法访问外网，请在启动脚本前配置镜像代理：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 二、 实验执行流 (Execution Flow)

### 1. 核心配置核对
在运行前，请检查 `IFCLora/config/llama3_ifc.yaml` 中的模型路径。
- **在线模式**：`model_id: "meta-llama/Meta-Llama-3-8B"`
- **离线模式**：`model_id: "/mnt/data/models/Llama-3-8B"` (根据实际路径修改)

### 2. 启动一键化探测与微调
```bash
# 使用单张 A800 运行以确保探测阶段 Hook 的稳定性
CUDA_VISIBLE_DEVICES=0 python IFCLora/train.py --config IFCLora/config/llama3_ifc.yaml
```

---

## 三、 突发情况预案 (Troubleshooting & Emergencies)

针对 8B 级别模型在 80GB 显存环境下的常见问题，请参考以下预案：

### 1. 显存溢出 (Out of Memory / OOM)
- **现象**：`RuntimeError: CUDA out of memory`.
- **原因**：探测阶段 (`MechanisticTracer`) 同时存储激活值与梯度，显存压力较大。
- **对策**：
  - 在 `train.py` 中减小校准样本数量（`calibration_prompts`）。
  - 若显存仍不足，在加载模型时启用 8-bit 量化：`load_in_8bit=True`。
  - A800 支持 BF16，确保 `TrainingArguments` 中 `bf16=True` 已开启，这比 FP32 节省一半显存。

### 2. 归因分数异常 (Zero/NaN Attribution)
- **现象**：日志显示 `PageRank 无法收敛` 或 `IFC 得分全为 NaN`。
- **原因**：梯度消失或 Prompt 未产生有效响应。
- **对策**：
  - 检查 `MechanisticTracer` 的正则匹配逻辑，确保目标组件（如 `q_proj`）在模型中确实存在。
  - 尝试增加 `calibration_prompts` 的多样性。

### 3. 数据集下载超时 (Dataset Connection Error)
- **现象**：`load_dataset("gsm8k")` 进程卡死或报错。
- **原因**：Hugging Face Datasets 服务器连接受限。
- **对策**：
  - 手动将数据集 `jsonl` 文件上传至 `IFCLora/data/`，并修改加载逻辑为 `load_dataset("json", ...)`。

### 4. 训练 Loss 不下降或变为 NaN
- **现象**：训练前 10 步 Loss 突变为 `NaN`。
- **原因**：学习率过高或异构秩注入后梯度冲突。
- **对策**：
  - 降低学习率（调至 `5e-5`）。
  - 适当增大 `warmup_steps`（建议设置为 100 步以上）。

---

## 四、 监控建议 (Monitoring)

建议在运行过程中开启另一个终端窗口实时监控显存：
```bash
watch -n 1 nvidia-smi
```
- **关键参考**：A800 在 BF16 精度下，运行 **IFCLoRA** 探测阶段显存占用应在 **30-40GB** 左右，训练阶段占用在 **20GB** 左右。

---
**提示**：若在 A800 集群上遇到任何多机多卡（Distributed）环境下的报错，请优先排查 `NCCL` 通讯环境变量设置。

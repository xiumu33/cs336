# CS336 Assignment 1 — 从零实现 Transformer 语言模型

本项目使用 PyTorch 从零实现一个 GPT 风格的 Transformer 语言模型，包括 BPE 分词、训练和推理。所有核心组件（线性层、注意力机制、归一化、优化器）均为手写实现，不依赖 PyTorch 内置的 Transformer 模块。

---

## 模块总览

### 1. `nn.py` — 神经网络基础组件

该模块实现了构建 Transformer 架构所需的所有核心神经网络组件。

| 类 / 函数 | 功能说明 |
|---|---|
| `Linear(in_features, out_features)` | 全连接线性层。权重采用截断正态分布初始化（标准差 = sqrt(2 / (d_in + d_out))，截断范围 [-3σ, 3σ]）。前向传播使用 `einsum('...i, oi -> ...o', x, self.weight)` 实现矩阵乘法。 |
| `Embedding(num_embeddings, embedding_dim)` | Token 嵌入层。权重采用截断正态分布初始化（标准差 = 1.0）。前向传播直接通过索引查表：`self.weight[token_ids]`。 |
| `RMSNorm(d_model, eps=1e-5)` | Root Mean Square 层归一化（Llama 中使用）。计算 `x / sqrt(mean(x²) + ε) * g`，其中 `g` 是可学习的缩放参数，初始化为全 1。内部先将输入转为 float32 计算以防止溢出，最后转回原始数据类型。 |
| `silu_fn(in_features)` | SiLU（Swish）激活函数：`x * sigmoid(x)`。 |
| `SwiGLU(d_model, d_ff)` | SwiGLU 前馈网络。包含两个升维层 W1、W3（d_model → d_ff）和一个降维层 W2（d_ff → d_model）。计算过程：`W2(silu(W1(x)) * W3(x))`。 |
| `RotaryPositionalEmbedding(theta, d_k, max_seq_len)` | 旋转位置编码（RoPE）。预计算所有位置的 cos/sin 频率表（最长到 max_seq_len）。前向传播时对 Q/K 张量的奇偶维度对施加旋转操作。cos/sin 缓存以 `persistent=False` 注册，不会保存到 state_dict 中。 |
| `softmax(x, dim=-1)` | 数值稳定的 softmax。计算前先减去维度最大值防止 exp 溢出。 |
| `scaled_dot_product_attention(Q, K, V, mask=None)` | 缩放点积注意力（SDPA）。计算 `softmax(Q·K^T / √d_k + mask) · V`。支持布尔类型 mask（False 的位置被设为 -∞）。 |
| `CausalSelfAttention(d_model, num_heads, ...)` | 多头因果自注意力层。通过三个 Linear 分别将输入投影为 Q/K/V，按头数拆分维度，可选地应用 RoPE，构造下三角因果 mask，调用 SDPA 计算注意力，最后合并所有头并通过输出投影层。 |
| `TransformerBlock(d_model, num_heads, d_ff, ...)` | 单个 Transformer 块，包含自注意力和前馈网络两个子层，均带残差连接。支持以下 ablation 选项：pre-norm / post-norm（归一化位置）、RMSNorm 开关、SwiGLU / SiLU FFN 类型切换。 |
| `TransformerLM(vocab_size, max_seq_len, ...)` | 完整的 Transformer 语言模型。依次堆叠 Token Embedding、N 个 TransformerBlock、最终 RMSNorm 和 LM Head（Linear → vocab_size），输出 logits。 |
| `TransformerLM.generate(prompt_ids, ...)` | 自回归文本生成。支持 temperature 温度缩放和 top-p 核采样。遇到 EOS token 时提前停止。使用 `torch.multinomial` 进行概率采样。 |

---

### 2. `losses.py` — 损失函数

| 函数 | 功能说明 |
|---|---|
| `cross_entropy(logits, targets)` | 数值稳定的交叉熵损失。对每个位置计算 `log(∑exp(o - M)) + M - o_y`，其中 M 为每组 logits 的最大值。最后返回所有位置的平均损失标量。使用 `torch.gather` 按 targets 索引提取目标位置的 logits。 |

---

### 3. `optimizer.py` — 优化器与梯度工具

| 类 / 函数 | 功能说明 |
|---|---|
| `AdamW(params, lr, betas, eps, weight_decay)` | 从零实现的 AdamW 优化器。维护一阶矩（exp_avg）和二阶矩（exp_avg_sq）的指数移动平均，应用偏差校正后计算自适应学习率。核心特点是 **解耦权重衰减**（decoupled weight decay）：`θ = θ - α·λ·θ` 与自适应梯度更新分开执行。 |
| `clip_gradient_norm(parameters, max_norm)` | 全局 L2 梯度范数裁剪。计算所有参数梯度的全局 L2 范数 `total_norm = sqrt(∑‖g_i‖²)`，若超过阈值 max_norm，则按比例 `max_norm / total_norm` 缩放所有梯度。用于防止梯度爆炸。 |

---

### 4. `scheduler.py` — 学习率调度器

| 函数 | 功能说明 |
|---|---|
| `get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)` | 带线性预热的余弦退火学习率调度。第一阶段（t < warmup_iters）：从 0 线性增长到 max_lr；第二阶段（warmup_iters ≤ t ≤ cosine_cycle_iters）：余弦衰减从 max_lr 到 min_lr；第三阶段（t > cosine_cycle_iters）：保持 min_lr 不变。 |

---

### 5. `tokenizer.py` — BPE 分词器

| 类 / 方法 | 功能说明 |
|---|---|
| `BPETokenizer(vocab, merges, special_tokens)` | 字节级 BPE 分词器。维护 ID ↔ bytes 双向映射、合并规则优先级字典（pair → rank，rank 越小越优先）以及特殊 token 的正则匹配器（按长度降序排列，优先最长匹配）。使用 GPT-2 预分词正则将文本切分为单词块。 |
| `BPETokenizer.encode(text)` | 将字符串编码为 token ID 列表。首先用正则匹配特殊 token 并将文本切分为普通文本片段和特殊 token；特殊 token 直接映射为 ID，普通文本片段交由内部的 BPE 合并流程处理；最后拼接所有 ID 返回。 |
| `BPETokenizer._encode_text_segment(text)` | 核心 BPE 编码流程：(1) 使用 GPT-2 正则进行预分词，将文本切分为单词/标点块；(2) 将每个块转为 UTF-8 字节序列；(3) 贪心地反复合并当前优先级最高（rank 最小）的相邻字节对，直到没有可合并的规则；(4) 将最终的字节块映射为 token ID。 |
| `BPETokenizer.decode(ids)` | 将 token ID 列表解码回 UTF-8 字符串。拼接所有 ID 对应的字节序列后解码，使用 `errors="replace"` 处理不完整的字节序列。 |
| `BPETokenizer.encode_iterable(iterable)` | 内存高效的流式编码器。接收一个可迭代的文本块对象（如文件句柄），通过生成器逐个 yield token ID，适合处理无法一次性加载到内存的大文件。 |

---

### 6. `data.py` — 数据加载

| 函数 | 功能说明 |
|---|---|
| `get_batch(dataset, batch_size, context_length, device)` | 从 1D numpy 数组数据集中随机采样。随机选取 batch_size 个起始位置，每个位置提取长度为 context_length 的 token 序列作为输入 x，向后偏移 1 位作为目标 y。数据转为 PyTorch 张量后移动到指定设备。 |

---

### 7. `checkpointing.py` — 模型持久化

| 函数 | 功能说明 |
|---|---|
| `save_checkpoint(model, optimizer, iteration, out)` | 保存训练状态。将 `model_state_dict`、`optimizer_state_dict` 和当前 `iteration` 打包为字典，通过 `torch.save` 写入磁盘（支持路径或文件流）。 |
| `load_checkpoint(src, model, optimizer)` | 从检查点恢复训练状态。使用 `map_location='cpu'` 加载以保证跨设备可移植性。恢复模型权重和优化器状态后，返回保存时的迭代次数。 |

---

### 8. `train_bpe.py` — BPE 分词器训练

| 函数 | 功能说明 |
|---|---|
| `train_bpe(input_path, vocab_size, special_tokens)` | 在给定语料上从头训练字节级 BPE 分词器。步骤：(1) 初始化词表为基础字节 0-255；(2) 按特殊 token 切分语料，确保特殊 token 不参与频率统计；(3) 使用 GPT-2 正则进行预分词并统计每个单词的频率；(4) 在剩余合并次数内，每轮选择当前频率最高（平局时按字典序最大）的字节对进行合并，使用倒排索引高效更新受影响的单词和统计信息；(5) 将合并产生的新 token 和特殊 token 加入最终词表。 |
| `bytes_to_unicode()` | 返回 0-255 字节到可见 Unicode 字符的映射（GPT-2 标准做法）。非可打印字节映射到私有 Unicode 区域。 |
| `save_tokenizer_files(vocab, merges, out_dir)` | 将词表和合并规则保存为 `vocab.json` 和 `merges.txt`。通过 `bytes_to_unicode()` 将 bytes 转为可见字符串后写入。 |

---

### 9. `preprocess.py` — 语料预处理

| 函数 | 功能说明 |
|---|---|
| `bytes_to_unicode()` | 同 `train_bpe.py`：字节到可见 Unicode 字符的映射。 |
| `load_trained_tokenizer(vocab_path, merges_path, special_tokens)` | 从磁盘加载已训练的 BPE 分词器。反向使用 `bytes_to_unicode()` 映射，将 `vocab.json` 中的可见字符串还原为原始 bytes，从 `merges.txt` 还原合并规则对。 |
| `process_corpus(input_txt, output_bin, tokenizer)` | 使用 `encode_iterable` 流式处理大型文本语料，将 token 以 uint16 格式分批（每 100 万个 token 写入一次）写入二进制文件。整个过程内存友好，适合处理大规模数据。 |

---

### 10. `inference.py` — 交互式文本生成

| 函数 | 功能说明 |
|---|---|
| `bytes_to_unicode()` | 字节到 Unicode 映射，用于加载分词器。 |
| `load_trained_tokenizer(vocab_path, merges_path, special_tokens)` | 从磁盘文件加载分词器。 |
| `main()` | 交互式文本生成脚本。加载训练好的 checkpoint 和模型，进入 REPL 循环：用户输入 prompt → 分词 → 调用 `model.generate()`（支持 temperature 和 top-p）→ 解码 → 打印生成结果。输入 `q` / `exit` 退出。通过命令行参数 `--temperature`、`--top_p`、`--max_new_tokens` 控制生成行为。 |

---

### 11. `sgd.py` — 带衰减的 SGD 优化器（学习率调试实验）

| 类 / 函数 | 功能说明 |
|---|---|
| `SGD(params, lr)` | 带步数衰减的朴素 SGD。更新公式：`θ = θ - (lr / √(t+1)) · g`，每个参数维护独立的步数计数器 t。 |
| `run_experiment(learning_rate)` | 运行 10 轮优化实验，目标函数为 `mean(W²)`（最优解在 W=0）。用于诊断不同学习率的表现：lr=10 平稳收敛，lr=100 快速收敛到最优，lr=1000 发散（loss 爆炸）。 |

---

### 12. `main_train.py` — 训练入口脚本

主训练脚本，将所有模块串联为完整的训练流程。支持丰富的命令行参数配置超参数和消融实验：

- **模型参数**：`--d_model`, `--num_layers`, `--num_heads`, `--d_ff`, `--vocab_size`, `--context_length`
- **消融实验开关**：`--no_rms_norm`（禁用 RMSNorm）、`--norm_mode`（pre/post）、`--no_rope`（禁用 RoPE）、`--ffn_type`（swiglu/silu）
- **优化器参数**：`--lr`, `--min_lr`, `--warmup_iters`, `--max_iters`, `--max_norm`
- **数据路径**：`--train_data_path`, `--valid_data_path`（uint16 二进制文件）
- **日志监控**：`--wandb_project`, `--run_name`（WandB 集成）

训练循环流程：余弦学习率调度 → 随机批次采样 → 前向传播 → 交叉熵损失 → 反向传播 → 梯度裁剪 → AdamW 参数更新。每 100 步记录训练/验证 loss 到 WandB，每 1000 步保存检查点，支持断点续训。

---

## 项目结构

```
assignment1/
├── cs336_basics/
│   ├── nn.py              # Transformer 神经网络模块（Linear, Attention, TransformerBlock, TransformerLM 等）
│   ├── losses.py           # 交叉熵损失函数
│   ├── optimizer.py        # AdamW 优化器 + 梯度裁剪
│   ├── scheduler.py        # 余弦退火学习率调度器（带线性预热）
│   ├── tokenizer.py        # BPE 分词器（encode / decode / 流式编码）
│   ├── train_bpe.py        # 在语料上训练 BPE 分词器
│   ├── preprocess.py       # 语料预处理：文本 → 二进制 token 文件
│   ├── data.py             # 从二进制数据中随机采样批次
│   ├── checkpointing.py    # 模型检查点的保存与加载
│   ├── sgd.py              # SGD 优化器 + 学习率调试实验
│   ├── inference.py        # 交互式文本生成脚本
│   ├── main_train.py       # 训练主入口
│   └── learn.ipynb         # Jupyter notebook
├── tests/                  # 测试代码
└── cs336_assignment1_basics.pdf  # 作业描述文档
```

## 使用方法

```bash
# 训练 BPE 分词器
python -m cs336_basics.train_bpe

# 将语料预处理为二进制格式
python -m cs336_basics.preprocess

# 训练模型
python -m cs336_basics.main_train \
    --train_data_path data/train.bin \
    --valid_data_path data/valid.bin \
    --d_model 512 --num_layers 4 --num_heads 8 \
    --max_iters 10000

# 交互式生成
python -m cs336_basics.inference \
    --checkpoint_path out/ckpt.pt \
    --temperature 0.8 --top_p 0.9
```

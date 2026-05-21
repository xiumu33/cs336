# main_train.py — 训练入口脚本

## 概述

本模块是完整训练流程的入口脚本，将所有子模块（模型、优化器、调度器、数据加载、检查点）串联为端到端的训练循环，支持丰富的命令行参数配置和消融实验。

---

## 主要功能

### main() — 训练主函数

#### 完整命令行参数

**模型超参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--batch_size` | 32 | 批次大小 |
| `--context_length` | 256 | 上下文长度 |
| `--d_model` | 512 | 隐藏层维度 |
| `--num_layers` | 4 | Transformer 层数 |
| `--num_heads` | 8 | 注意力头数 |
| `--d_ff` | 2048 | FFN 中间层维度 |
| `--vocab_size` | 10000 | 词表大小 |

**消融实验开关**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--no_rms_norm` | False | 禁用 RMSNorm（全部使用 nn.Identity） |
| `--norm_mode` | `"pre"` | 归一化位置：`"pre"`（Llama 风格）或 `"post"`（原始 Transformer） |
| `--no_rope` | False | 禁用 RoPE 位置编码（NoPE 实验） |
| `--ffn_type` | `"swiglu"` | FFN 类型：`"swiglu"` 或 `"silu"`（标准 SiLU FFN） |

**优化器超参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--lr` | 6e-4 | 最大学习率 |
| `--min_lr` | 6e-5 | 最小学习率（退火终点） |
| `--warmup_iters` | 1000 | 预热步数 |
| `--max_iters` | 10000 | 总训练步数 |
| `--max_norm` | 1.0 | 梯度裁剪阈值 |

**路径与系统**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--train_data_path` | **必填** | 训练数据二进制文件路径 |
| `--valid_data_path` | **必填** | 验证数据二进制文件路径 |
| `--out_dir` | `"out"` | 输出目录（检查点和日志） |
| `--device` | cuda/cpu | 运行设备 |

**WandB 日志**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--wandb_project` | `"cs336-pretraining"` | WandB 项目名 |
| `--run_name` | None | 实验名称（用于区分不同实验） |

---

#### 训练流程（逐步说明）

**步骤 1：加载数据**

使用 `np.memmap` 以只读模式映射二进制 token 文件。memmap 延迟加载，只有被访问的数据页才会读入内存，非常适合大规模数据集。

**步骤 2：消融实验逻辑**

根据命令行参数计算实际配置：
- `no_rope=True` → `rope_theta = None`，内部不初始化 RoPE
- `no_rms_norm=True` → `use_rms_norm = False`

**步骤 3：初始化模型**

创建 `TransformerLM` 实例，传入所有超参数和消融实验配置，移至目标设备。

**步骤 4：初始化优化器**

创建 `AdamW` 实例，学习率按命令行设置，`weight_decay=0.1`。

**步骤 5：检查点恢复**

如果输出目录存在 `ckpt.pt`，调用 `load_checkpoint` 恢复模型权重、优化器状态和迭代次数，实现断点续训。

**步骤 6：初始化 WandB**

启动 WandB 监控，记录所有超参数配置。

**步骤 7：主训练循环**（`for it in range(start_iter, max_iters)`）

每步执行：

1. **更新学习率**：调用 `get_lr_cosine_schedule(it, ...)` 计算当前学习率，写入优化器的 param_groups
2. **采样训练批次**：`get_batch(train_data, ...)` 随机采样 x, y
3. **前向传播**：`logits = model(x)` 获取预测
4. **计算损失**：`loss = cross_entropy(logits, y)`
5. **反向传播**：`loss.backward()` 计算梯度
6. **梯度裁剪**：`clip_gradient_norm(model.parameters(), max_norm)`
7. **参数更新**：`optimizer.step()` 执行 AdamW 更新

**验证与日志**（每 100 步或最后一步）：

1. 模型切换到 eval 模式
2. 采样验证批次，计算验证 loss
3. 打印 `train_loss`, `val_loss`, `lr` 到控制台
4. 上传到 WandB

**检查点保存**（每 1000 步）：调用 `save_checkpoint` 保存当前状态。

**训练结束**：保存最终检查点 `ckpt_final.pt`，调用 `wandb.finish()`。

---

## 完整源代码

```python
import argparse
import os
import torch
import numpy as np
import wandb
from nn import TransformerLM
from optimizer import AdamW, clip_gradient_norm
from scheduler import get_lr_cosine_schedule
from data import get_batch
from checkpointing import save_checkpoint, load_checkpoint
from losses import cross_entropy


def main():
    parser = argparse.ArgumentParser()
    # --- 模型基础超参数 ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)
    
    # --- 实验/消融 (Ablation) 开关 ---
    # Ablation 1: 移除 RMSNorm
    parser.add_argument("--no_rms_norm", action="store_true", help="Disable RMSNorm completely")
    # Ablation 2: Pre-norm vs Post-norm
    parser.add_argument("--norm_mode", type=str, default="pre", choices=["pre", "post"], help="Normalization placement")
    # Ablation 3: 移除 RoPE (NoPE)
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Embeddings")
    # Ablation 4: SwiGLU vs SiLU
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"], help="Type of Feed-Forward Network")

    # --- 优化器超参数 ---
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--max_norm", type=float, default=1.0)
    
    # --- 路径与系统 ---
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # --- WandB 设置 ---
    parser.add_argument("--wandb_project", type=str, default="cs336-pretraining")
    parser.add_argument("--run_name", type=str, default=None, help="WandB 实验名称")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载数据 (使用 memmap)
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode='r')

    print(f"训练集大小: {len(train_data)} tokens")
    print(f"验证集大小: {len(val_data)} tokens")

    # 2. 处理消融实验逻辑
    actual_rope_theta = None if args.no_rope else 10000.0
    use_rms_norm = not args.no_rms_norm

    # 3. 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size, 
        context_length=args.context_length,
        d_model=args.d_model, 
        num_layers=args.num_layers,
        num_heads=args.num_heads, 
        d_ff=args.d_ff,
        rope_theta=actual_rope_theta,
        device=args.device,
        use_rms_norm=use_rms_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type
    ).to(args.device)

    print(f"Model Config: Norm={args.norm_mode}, UseNorm={use_rms_norm}, FFN={args.ffn_type}, RoPE={not args.no_rope}")

    # 4. 初始化优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # 5. 检查点恢复逻辑
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resuming from iteration {start_iter}")

    # 6. 初始化 WandB 监控
    wandb.init(
        project=args.wandb_project,
        name=args.run_name, 
        config=args
    )

    # 7. 主训练循环
    for it in range(start_iter, args.max_iters):
        # A. 更新学习率
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. 训练步
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        logits = model(x)
        loss = cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        clip_gradient_norm(model.parameters(), args.max_norm)
        
        optimizer.step()

        # C. 验证与日志记录
        if it % 100 == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
                print(f"Iter {it}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}, lr {lr:.2e}")
                wandb.log({
                    "train/loss": loss.item(), 
                    "val/loss": v_loss.item(), 
                    "lr": lr, 
                    "iter": it + 1
                })

        # D. 保存检查点 (每 1000 步保存一次)
        if it % 1000 == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)

    # 训练结束保存最终模型
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    wandb.finish()

if __name__ == "__main__":
    main()
```

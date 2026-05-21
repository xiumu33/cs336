# inference.py — 交互式文本生成

## 概述

本模块实现一个命令行交互式文本生成脚本，加载训练好的 Transformer 语言模型和 BPE 分词器，在终端中实现"输入 prompt → 生成文本"的 REPL 循环。

---

## 1. bytes_to_unicode — 字节到可见 Unicode 映射

### 函数 `bytes_to_unicode()`

与其他模块中的同名函数一致。返回 0-255 字节到可见 Unicode 字符的映射，用于加载分词器时反向还原。

---

## 2. load_trained_tokenizer — 加载已训练的分词器

### 函数 `load_trained_tokenizer(vocab_path, merges_path, special_tokens)`

与 `preprocess.py` 中的同名函数逻辑一致。从 `vocab.json` 和 `merges.txt` 加载并还原 BPETokenizer 实例。

**错误处理**：若分词器文件不存在，打印错误信息并 `sys.exit(1)`。

---

## 3. main — 交互式生成主循环

### 函数 `main()`

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--vocab_size` | int | 10000 | 词表大小 |
| `--context_length` | int | 256 | 上下文长度（模型最大输入长度） |
| `--d_model` | int | 512 | 隐藏层维度 |
| `--num_layers` | int | 4 | Transformer 层数 |
| `--num_heads` | int | 16 | 注意力头数 |
| `--d_ff` | int | 1344 | FFN 中间层维度 |
| `--rope_theta` | float | 10000.0 | RoPE 基准频率 |
| `--checkpoint_path` | str | **必填** | 训练好的检查点文件路径 |
| `--tokenizer_dir` | str | `data/tokenizer_results` | 分词器文件所在目录 |
| `--temperature` | float | 0.8 | 温度参数：越高越随机，越低越确定 |
| `--top_p` | float | 0.9 | Nucleus Sampling 阈值 |
| `--max_new_tokens` | int | 100 | 最多生成的 token 数 |
| `--device` | str | cuda/cpu | 运行设备 |

#### 执行流程

**第 1 步：加载分词器**

从 `tokenizer_dir` 目录下读取 `vocab.json` 和 `merges.txt`，构建 BPETokenizer。获取 EOS token（`<|endoftext|>`）的 ID 用于生成时的提前停止。

**第 2 步：初始化模型**

按照命令行指定的架构参数创建 `TransformerLM` 实例。**模型结构必须与训练时完全一致**，否则权重加载会失败。

**第 3 步：加载权重**

- 使用 `torch.load` 加载检查点文件
- 自动兼容两种格式：含 `model_state_dict` 键的检查点字典，或直接保存的 state_dict
- 调用 `model.load_state_dict()` 加载权重
- 模型移至目标设备并切换到 eval 模式

**第 4 步：REPL 循环**

```
Prompt > 用户输入文本
...
Response:
模型生成的文本
...
```

- 用户输入 prompt → tokenizer.encode 分词 → 封装为 batch=1 的 tensor
- 调用 `model.generate()` 生成 token 序列
- tokenizer.decode 解码为文本并打印
- 输入 `q` / `exit` / `quit` 退出
- 支持 Ctrl+C 优雅退出

**生成参数说明**：
- `temperature`：对 logits 除以温度值，T < 1 使分布更尖锐（更确定），T > 1 使分布更平滑（更随机）
- `top_p`：核采样，只保留累积概率刚好超过 p 的最小高概率 token 集合，截断概率极低的"长尾"token

---

## 完整源代码

```python
import torch
import argparse
import os
import json
import sys
from tokenizer import BPETokenizer
from nn import TransformerLM


def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
    int -> str
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def load_trained_tokenizer(vocab_path, merges_path, special_tokens=["<|endoftext|>"]):
    """
    加载训练好的 BPE 分词器 (逻辑与 preprocess.py 一致)
    """
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"错误: 找不到分词器文件。\nVocab: {vocab_path}\nMerges: {merges_path}")
        sys.exit(1)
    
    
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
        # 将可见字符串还原为原始 bytes
        vocab = {
            int(k): bytes([byte_decoder[c] for c in v]) 
            for k, v in vocab_raw.items()
        }
    
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(' ')
            if len(parts) == 2:
                # 还原 p1, p2 为 bytes
                p1 = bytes([byte_decoder[c] for c in parts[0]])
                p2 = bytes([byte_decoder[c] for c in parts[1]])
                merges.append((p1, p2))
    
    return BPETokenizer(vocab, merges, special_tokens)



def main():
    parser = argparse.ArgumentParser(description="CS336 Transformer Inference Script")
    # --- 模型参数 (必须与训练时完全一致！) ---
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # --- 生成参数 ---
    parser.add_argument("--checkpoint_path", type=str, required=True, help="ckpt.pt 的路径")
    parser.add_argument("--tokenizer_dir", type=str, default="data/tokenizer_results")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度：越低越保守，越高越随机")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus Sampling 阈值")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="生成的最大长度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()

    # 1. 加载 Tokenizer
    vocab_path = os.path.join(args.tokenizer_dir, "vocab.json")
    merges_path = os.path.join(args.tokenizer_dir, "merges.txt")
    tokenizer = load_trained_tokenizer(vocab_path, merges_path)
    
    # 获取 EOS Token ID 用于提前停止
    eos_token_id = tokenizer.byte_to_id.get(b"<|endoftext|>", None)

    # 2. 初始化模型架构
    print(f"正在初始化模型 (d_model={args.d_model}, layers={args.num_layers})...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device
    )

    # 3. 加载权重
    print(f"正在加载权重: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print("错误: 找不到 Checkpoint 文件")
        return

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"权重加载失败！请检查模型参数是否与训练时一致。\n详细错误: {e}")
        return

    model.to(args.device)
    model.eval()
    print("模型加载完成！")

    # 4. 交互式生成循环
    print("\n" + "="*30)
    print("开始对话 (输入 'q' 或 'exit' 退出)")
    print("="*30 + "\n")

    while True:
        try:
            user_input = input("Prompt > ")
            if user_input.lower() in ["q", "exit", "quit"]:
                break
            
            if not user_input.strip():
                continue

            # 编码输入
            input_ids = tokenizer.encode(user_input)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=args.device)

            # 生成
            with torch.no_grad():
                output_ids = model.generate(
                    input_tensor,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos_token_id,
                    temperature=args.temperature,
                    top_p=args.top_p
                )

            # 解码输出
            generated_text = tokenizer.decode(output_ids[0].tolist())
            
            print(f"\nResponse:\n{generated_text}\n")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n退出...")
            break

if __name__ == "__main__":
    main()
```

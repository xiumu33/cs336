# preprocess.py — 语料预处理

## 概述

本模块将训练好的 BPE 分词器应用于原始语料，通过流式编码将文本转换为二进制 token 文件，供训练高效读取。

---

## 1. bytes_to_unicode — 字节到可见 Unicode 映射

### 函数 `bytes_to_unicode()`

与 `train_bpe.py` 中同名函数完全一致。返回 0-255 字节到可见 Unicode 字符的映射字典。在此模块中用于加载分词器时反向还原 bytes。

---

## 2. load_trained_tokenizer — 加载已训练的分词器

### 函数 `load_trained_tokenizer(vocab_path, merges_path, special_tokens)`

#### 参数

- `vocab_path`：`vocab.json` 文件路径
- `merges_path`：`merges.txt` 文件路径
- `special_tokens`：特殊 token 列表

#### 返回值

- `BPETokenizer` 实例

#### 加载流程

1. **建立反向映射**：`byte_decoder = {v: k for k, v in byte_encoder.items()}`，将可见 Unicode 字符反查回 0-255 整数值

2. **还原词表**：读取 `vocab.json`，对每个 token 的值（可见字符串）逐字符通过 byte_decoder 反查，组合为原始 bytes

3. **还原合并规则**：逐行读取 `merges.txt`，按空格拆分每一行为两个 token 字符串，分别还原为 bytes 后组成 `(bytes_a, bytes_b)` 对

4. 用还原的 vocab、merges、special_tokens 构建并返回 BPETokenizer 实例

---

## 3. process_corpus — 流式预处理语料

### 函数 `process_corpus(input_txt, output_bin, tokenizer, chunk_size_mb=50)`

#### 参数

- `input_txt`：原始文本语料路径
- `output_bin`：输出二进制文件路径
- `tokenizer`：已加载的 BPETokenizer 实例
- `chunk_size_mb`：每次从磁盘读取的文本块大小（MB），默认 50MB

#### 流式处理流程

1. **文件块生成器**（内部函数 `file_chunk_generator`）：
   - 按 chunk_size 逐块读取文本文件
   - 一次只加载一块在内存中，避免 OOM

2. **流式编码**：
   - 将 chunk 生成器传入 `tokenizer.encode_iterable()`，得到一个逐 token 产出的 ID 流
   - 通过生成器链实现全程流式处理，内存占用极小

3. **批量写盘**：
   - 维护一个大小为 100 万 token 的内存 buffer
   - buffer 满时转为 numpy uint16 数组，写入二进制文件
   - 最后处理剩余不足 100 万的 buffer

4. **输出格式**：连续 uint16 字节流，训练时可用 `np.memmap` 高效读取

---

## 完整源代码

```python
import os
import json
import numpy as np
from typing import List, Dict
from tokenizer import BPETokenizer

def bytes_to_unicode():
    """
    返回一个字节到可见 Unicode 字符的映射字典。
    该映射确保所有 256 个字节值都有对应的可见字符
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

def load_trained_tokenizer(vocab_path: str, merges_path: str, special_tokens: List[str]):
    """
    从磁盘加载训练好的分词器，处理 byte_to_unicode 反向映射
    """
    print(f"正在从 {os.path.dirname(vocab_path)} 加载分词器...")
    
    # 1. 建立反向映射表
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    # 2. 加载并还原词表
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
        # 将可见的 Unicode 字符串还原为原始 bytes
        vocab = {
            int(k): bytes([byte_decoder[c] for c in v]) 
            for k, v in vocab_raw.items()
        }
    
    # 3. 加载并还原合并规则
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip('\n')
            if not line: continue
            
            # 使用 rsplit 确保在 token 本身包含空格的情况下依然稳健
            parts = line.split(' ')
            if len(parts) == 2:
                # 还原 p1, p2 为原始 bytes
                p1 = bytes([byte_decoder[c] for c in parts[0]])
                p2 = bytes([byte_decoder[c] for c in parts[1]])
                merges.append((p1, p2))
    
    print(f"成功加载词表，当前词表规模: {len(vocab)}")
    return BPETokenizer(vocab, merges, special_tokens)

def process_corpus(input_txt: str, output_bin: str, tokenizer: BPETokenizer, chunk_size_mb: int = 50):
    # 1. 内部生成器：负责按块从硬盘读文本
    def file_chunk_generator(file_path, size):
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(size)
                if not chunk:
                    break
                yield chunk

    # 2. 检查与准备
    if not os.path.exists(input_txt):
        raise FileNotFoundError(f"找不到语料文件: {input_txt}")
    
    chunk_size = 1024 * 1024 * chunk_size_mb
    
    if os.path.exists(output_bin):
        os.remove(output_bin)

    print(f"使用 encode_iterable 开始流式预处理...")

    # 3. 核心流式逻辑
    # 创建文本块生成器
    chunks = file_chunk_generator(input_txt, chunk_size)
    # 丢进 encode_iterable，得到一个"不停吐出 ID"的生成器
    token_stream = tokenizer.encode_iterable(chunks)

    total_tokens = 0

    # 为了高效写入硬盘，我们依然需要一个小缓存（Buffer）
    # 每积攒 100 万个 Token 写入一次硬盘
    write_batch_size = 1_000_000 
    token_buffer = []

    with open(output_bin, "ab") as f_out:
        for token_id in token_stream:
            token_buffer.append(token_id)
            
            if len(token_buffer) >= write_batch_size:
                np_ids = np.array(token_buffer, dtype=np.uint16)
                f_out.write(np_ids.tobytes())
                total_tokens += len(token_buffer)
                token_buffer = []
            
        
        # 处理最后剩余的 buffer
        if token_buffer:
            np_ids = np.array(token_buffer, dtype=np.uint16)
            f_out.write(np_ids.tobytes())
            total_tokens += len(token_buffer)

    print(f"处理完成！总 Token: {total_tokens}")

def main():
    # --- 配置区 ---
    BASE_DIR = "data/TinyStoriesV2-GPT4-train"
    input_file = "data/TinyStoriesV2-GPT4-train.txt"
    output_file = "data/TinyStoriesV2-GPT4-train.bin"
    
    vocab_json = os.path.join(BASE_DIR, "vocab.json")
    merges_txt = os.path.join(BASE_DIR, "merges.txt")
    special_tokens = ["<|endoftext|>"]
    
    # 1. 加载分词器
    tokenizer = load_trained_tokenizer(vocab_json, merges_txt, special_tokens)
    
    # 2. 执行数据清洗与预处理
    process_corpus(input_file, output_file, tokenizer)

if __name__ == "__main__":
    main()
```

# train_bpe.py — BPE 分词器训练

## 概述

本模块实现在给定语料上从头训练字节级 BPE 分词器的完整流程，包含倒排索引优化的高效合并算法。

---

## 1. bytes_to_unicode — 字节到可见 Unicode 映射

### 函数 `bytes_to_unicode()`

返回一个字典，将 0-255 的整数字节值映射为可见的 Unicode 字符。这是 GPT-2 源码中的标准做法。

**映射策略**：
- 可打印的 ASCII 字符（`!` 到 `~`，`¡` 到 `¬`，`®` 到 `ÿ`）直接映射为自身
- 其余不可打印的控制字符和非 ASCII 字节映射到私有 Unicode 区域（从 U+0100 开始）

**用途**：训练后将 bytes 转为可见字符串保存到 JSON，避免 `vocab.json` 中出现乱码。

---

## 2. train_bpe — 训练 BPE 分词器

### 函数 `train_bpe(input_path, vocab_size, special_tokens)`

#### 参数

- `input_path`：输入语料文件路径（纯文本）
- `vocab_size`：目标词表大小（含基础字节 256 + 合并 token + 特殊 token）
- `special_tokens`：需要保留的特殊 token 列表（如 `["<|endoftext|>"]`）

#### 返回值

- `vocab`：`dict[int, bytes]`，训练好的词汇表（ID → 字节序列）
- `merges`：`list[tuple[bytes, bytes]]`，按生成顺序排列的 BPE 合并规则

#### 完整训练流程

**步骤 1：初始化基础词表**

词表从 `{0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}` 开始。计算需合并次数：`num_merges = vocab_size - 256 - len(special_tokens)`。

**步骤 2：读取并分割语料**

- 读取全部文本
- 如果有特殊 token：用正则 `(special1|special2|...)` 的捕获组进行 split，将语料在特殊 token 处切开
- 过滤掉特殊 token 本身，只保留普通文本片段用于 BPE 统计

**步骤 3：预分词并统计词频**

- 对每个普通文本片段，使用 GPT-2 正则预分词
- 将每个单词转为 UTF-8 字节元组，统计每个单词的出现频率
- 将单词以 list 形式存入 words_list（list 可变，便于后续合并），频率存入 counts_list

**步骤 4：构建高效数据结构**

- `stats`（`defaultdict(int)`）：存储每对相邻字节的全局频率
- `indices`（`defaultdict(set)`）：倒排索引，记录每对出现在哪些单词的下标
- 遍历所有单词初始化这两个结构

**步骤 5：迭代合并循环**（执行 num_merges 次）

每轮执行：

1. **寻找最佳对**：`best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]`
   - 优先选频率最高的；频率相同选字典序最大的
   - 如果最佳对频率 ≤ 0 或 stats 为空，提前停止

2. **获取受影响单词**：通过倒排索引 indices 快速找到所有包含 best_pair 的单词下标

3. **遍历更新每个受影响单词**：
   - 扫描单词中的匹配位置
   - 更新旧邻居对的频率（左邻居、右邻居），频率减掉该单词的频次
   - 频率降为 0 的对从 stats 中删除（防止 max 选到无效对）
   - 将匹配的两个字节合并为新 token
   - 添加新邻居对的频率和倒排索引

4. **清理**：删除已合并的 best_pair 的 stats 和 indices 记录

**步骤 6：构建最终词表**

- 按合并顺序，给每个新 token 分配 ID（从 256 开始递增）
- 将特殊 token 追加到词表末尾

---

## 3. save_tokenizer_files — 保存分词器文件

### 函数 `save_tokenizer_files(vocab, merges, out_dir)`

1. 创建输出目录
2. 保存 `vocab.json`：通过 bytes_to_unicode 将每个 token 的 bytes 转为可见字符串后写入 JSON
3. 保存 `merges.txt`：每行一个合并对，格式为 `s1 s2`（s1 和 s2 为转换后的可见字符串）

---

## 完整源代码

```python
import os
from collections import defaultdict, Counter
import regex as re  # type: ignore
import json


def train_bpe(
    input_path: str | os.PathLike,  # 输入语料文件的路径
    vocab_size: int,             # 目标词表大小（基础字节 + 合并 Token + 特殊 Token）
    special_tokens: list[str],   # 需要保留的特殊 Token 列表
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE (Byte-Pair Encoding) 分词器。
    """
    
    # --- 1. 初始化基础词表 ---
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    
    # --- 2. 读取语料，并按特殊 Token 分割 ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        special_regex = "|".join(re.escape(t) for t in special_tokens)
        parts = re.split(f"({special_regex})", text)
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        train_segments = [text]

    # --- 3. 预分词（Pre-tokenization）并统计词频 ---
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    raw_counts = Counter()
    for segment in train_segments:
        words = gpt2_pat.findall(segment)
        for word in words:
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1
            
    # --- 构建高效数据结构以支持快速合并 ---
    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)

    stats = defaultdict(int)
    indices = defaultdict(set)
    
    for idx, word in enumerate(words_list):
        freq = counts_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            stats[pair] += freq
            indices[pair].add(idx)
            
    merges = []

    # --- 4. 迭代合并流程 ---
    for _ in range(num_merges):
        if not stats:
            break
            
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        
        if stats[best_pair] <= 0:
            break
            
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        
        relevant_indices = list(indices[best_pair])
        
        for idx in relevant_indices:
            word = words_list[idx]
            freq = counts_list[idx]
            
            i = 0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    
                    if i > 0:
                        prev_pair = (word[i-1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                        
                    if i < len(word) - 2:
                        next_pair = (word[i+1], word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]
                      
                    word[i] = new_token
                    del word[i+1]
                    
                    if i > 0:
                        new_prev = (word[i-1], word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)
                    
                    if i < len(word) - 1:
                        new_next = (word[i], word[i+1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)
                else:
                    i += 1
        
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]

    # --- 5. 构建最终的词表 ---
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]
        
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[len(vocab)] = s_bytes

    return vocab, merges


def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
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


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    byte_encoder = bytes_to_unicode()

    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    
    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")
    
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    save_tokenizer_files(vocab, merges, output_dir)

if __name__ == "__main__":
    main()
```

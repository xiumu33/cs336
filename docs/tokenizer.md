# tokenizer.py — BPE 分词器

## 概述

本模块实现了一个完整的字节级 BPE（Byte-Pair Encoding）分词器，包含编码、解码和流式处理功能。该实现遵循 GPT-2 的分词范式：字节级处理（无 OOV 问题）+ GPT-2 预分词正则 + 贪心 BPE 合并。

---

## BPETokenizer 类

### 初始化 `__init__(vocab, merges, special_tokens)`

创建分词器实例时建立以下数据结构：

1. **双向映射**：
   - `id_to_byte`（vocab）：ID → bytes，直接引用传入的 vocab 字典
   - `byte_to_id`：bytes → ID，通过反转 vocab 构建，用于编码时查表

2. **合并规则优先级字典** `merges`：
   - 结构：`{(byte_a, byte_b): rank}`，rank 为合并规则在 merges 列表中的索引
   - rank 越小表示在 BPE 训练中越早出现，编码时优先级越高（贪心合并策略）

3. **特殊 token 正则匹配器**：
   - 按长度降序排列所有特殊 token，构建正则：`(token1|token2|...)`
   - 降序排列确保最长匹配优先，防止短 token 截断长 token（如 `<|a|>` 干扰 `<|ab|>`）
   - 使用 `re.escape()` 处理 token 中的特殊字符（如 `|`、`[`）

4. **GPT-2 预分词正则**：
   - 模式：`'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
   - 作用：在 BPE 合并前将文本切分为"单词"块，防止合并跨越标点或空格边界

---

### encode(text) — 编码字符串为 token ID 序列

**边界情况**：空字符串直接返回 `[]`。

**快速路径**（无特殊 token 时）：
若未定义特殊 token，整段文本直接作为普通文本交给 `_encode_text_segment` 处理。

**完整路径**（含特殊 token 时）：

1. 使用 `special_regex.finditer(text)` 遍历所有特殊 token 匹配位置
2. 对于每个匹配到的特殊 token：
   - 取出**前置文本**（`text[last_pos:match.start()]`），交给 `_encode_text_segment` 进行 BPE 编码
   - 特殊 token 本身直接查表 `byte_to_id` 得到 ID（不参与 BPE 合并）
3. 处理**剩余收尾文本**（`text[last_pos:]`），同样进行 BPE 编码
4. 拼接所有 ID 列表返回

---

### _encode_text_segment(text) — 纯文本 BPE 编码核心

这是对不含特殊 token 的普通文本执行 BPE 算法的核心方法：

**第 1 步：预分词**
使用 GPT-2 正则将文本切分为单词/标点块。
例如：`"Hello world!"` → `['Hello', ' world', '!']`

**第 2 步：字节化**
将每个预分词块转为 UTF-8 字节序列，每个字节作为一个独立的部分（Part）。
例如：`"Hello"` → `[b'H', b'e', b'l', b'l', b'o']`

**第 3 步：贪心 BPE 合并**
对每个预分词块的字节序列反复执行：
1. 遍历所有相邻对，找到在 merges 字典中存在、且 rank 最小的字节对
2. 如果找不到可合并的对，退出循环
3. 扫描当前序列，将所有匹配该对的两个相邻字节合并为一个 bytes 对象
4. 更新序列，进入下一轮 while 循环

**第 4 步：映射为 ID**
将合并后的所有字节块通过 `byte_to_id` 字典查表转为 token ID。

---

### decode(ids) — 解码 token ID 序列为字符串

1. 查表 `id_to_byte` 将每个 ID 转为 bytes 片段
2. 拼接所有 bytes 片段为完整字节流
3. UTF-8 解码，使用 `errors="replace"` 处理不完整字节序列（在 BPE 中可能发生多字节字符被拆解的情况）

---

### encode_iterable(iterable) — 流式编码器

内存友好的生成器模式编码器：
1. 接收一个可迭代对象（如文件句柄的文本块）
2. 对每个文本块调用 `encode(chunk)`
3. 通过 `yield from` 逐个产出 token ID
4. 适合处理无法一次性读入内存的大型语料文件

---

## 完整源代码

```python
import regex as re  # 使用 regex 而非内置 re，因为它支持 Unicode 类别（如 \p{L}）
from collections.abc import Iterable

"""
For special_tokens:
    推理/编码阶段 (Tokenizer.encode)
        在模型使用分词器将文本转为 ID 时，必须优先匹配特殊 Token。
    代码逻辑：
        正则匹配：构建一个包含所有特殊 Token 的正则表达式。
        优先级：先扫描文本，一旦发现特殊 Token，直接将其转为对应的 ID。
        普通处理：特殊 Token 之间的文本，再走正常的 GPT-2 预分词和 BPE 合并流程。
"""

class BPETokenizer:
    """
    字节级 BPE（Byte-Pair Encoding）分词器实现。
    
    该分词器将任意字符串编码为整数 ID 序列，并能将 ID 序列还原。
    它采用字节级处理，确保不会出现未知词（OOV）错误。
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        初始化分词器。
        
        参数:
            vocab: 词汇表，建立整数 ID 到 字节块(bytes) 的映射。
            merges: 合并规则列表。列表中的每一项是一个二元组 (bytes_a, bytes_b)，
                   表示在训练过程中 bytes_a 和 bytes_b 被合并的顺序。
            special_tokens: 特殊标记列表（如 <|endoftext|>），这些标记不会被 BPE 规则拆分。
        """
        # 1. 建立双向映射，方便查表
        self.vocab = vocab  # ID -> 字节块
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()} # 字节块 -> ID
        
        # 2. 将合并规则转换为Rank字典。
        # BPE 编码时，必须优先应用在训练阶段较早出现的合并规则。
        # 字典结构为: {(byte_a, byte_b): 顺序索引}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        
        self.special_tokens = special_tokens or []
        
        # 3. 构建特殊 Token 的正则表达式
        if self.special_tokens:
            # 关键：必须按照长度从长到短排序（reverse=True）。
            # 这样正则引擎会优先匹配最长的特殊标记，防止重叠标记（如 <|a|><|b|>）被错误拆分。
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            # 使用 re.escape 确保标记中的特殊字符（如 | 或 [ ）被当作普通字符处理
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None

        # 4. GPT-2 官方预分词正则表达式。
        # 它的作用是在应用 BPE 合并前，先将文本切分成单词、标点、数字等逻辑块。
        # 这样做是为了防止 BPE 规则跨越单词或标点（例如：防止将 "dog" 的末尾和 "." 合并）。
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def encode(self, text: str) -> list[int]:
        """
        将输入的原始字符串编码为整数 ID 列表。
        """
        # --- 步骤 1: 边界情况检查 ---
        if not text:
            return []

        # --- 步骤 2: 情况 A - 快速路径 (Fast Path) ---
        if not self.special_regex:
            return self._encode_text_segment(text)

        # --- 步骤 3: 情况 B - 处理含有特殊标记的复杂文本 ---
        tokens = []
        last_pos = 0
        
        for match in self.special_regex.finditer(text):
            pre_text = text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            
            special_tok = match.group()
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])
            
            last_pos = match.end()
            
        # --- 步骤 4: 处理"收尾文本" ---
        remaining_text = text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))
            
        return tokens

    def _encode_text_segment(self, text: str) -> list[int]:
        """
        内部核心函数：对不含特殊 Token 的纯文本片段应用 BPE 合并逻辑。
        """
        ids = []
        pre_tokens = self.gpt2_pat.findall(text)
        
        for p_tok in pre_tokens:
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]
            
            while len(byte_parts) >= 2:
                best_pair = None
                min_rank = float('inf')
                
                for i in range(len(byte_parts) - 1):
                    pair = (byte_parts[i], byte_parts[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair
                
                if best_pair is None:
                    break 
                
                new_byte_parts = []
                i = 0
                while i < len(byte_parts):
                    if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                        new_byte_parts.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i += 1
                byte_parts = new_byte_parts
            
            for part in byte_parts:
                ids.append(self.byte_to_id[part])
                
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        将 ID 列表解码为原始字符串。
        """
        byte_segments = [self.id_to_byte[i] for i in ids]
        full_bytes = b"".join(byte_segments)
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        内存高效的迭代编码器。
        """
        for chunk in iterable:
            yield from self.encode(chunk)
```

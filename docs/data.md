# data.py — 数据加载

## 概述

本模块提供从预处理后的 token 数据集中随机采样训练批次的功能。

---

## get_batch — 随机批次采样

### 函数 `get_batch(dataset, batch_size, context_length, device)`

#### 参数

- `dataset`：1 维 numpy 数组（dtype 为 uint16 或 int），存储整个语料的 token ID 序列
- `batch_size`：批次大小 B
- `context_length`：上下文长度 m（即每个样本的 token 数）
- `device`：目标设备字符串（'cpu'、'cuda'、'mps'）

#### 返回值

- `x`：输入 token 序列，形状 `(batch_size, context_length)`，dtype 为 long
- `y`：目标 token 序列，形状 `(batch_size, context_length)`，dtype 为 long
- x 和 y 的关系：`y[i] = x[i+1]`，即 y 是 x 向后偏移 1 位的下一个 token，用于语言模型的下一个 token 预测任务

#### 算法步骤

1. 计算合法的最大起始索引：`max_idx = len(dataset) - context_length - 1`（因为还需要取 y 的最后一个位置）

2. 使用 `torch.randint` 在 `[0, max_idx]` 范围内随机生成 batch_size 个起始位置

3. 对每个起始位置 i：
   - x 切片：`dataset[i : i + context_length]`
   - y 切片：`dataset[i+1 : i + context_length + 1]`

4. 将所有切片堆叠为 numpy 数组 → 转为 PyTorch 张量 → 移动到目标设备 → 转为 long 类型

---

## 完整源代码

```python
import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 Numpy 数组数据集中随机采样一个批次。
    
    参数:
        dataset: 1D Numpy 数组 (Token IDs)
        batch_size: 批大小 (B)
        context_length: 上下文长度 (m)
        device: 设备字符串 ('cpu', 'cuda', 'mps')
    """
    # 1. 确定合法的最大起点索引
    # 我们需要取长度为 context_length 的片段，且目标还要往后偏移一位
    # 所以最后一个可用的起点是 len(dataset) - context_length - 1
    n = len(dataset)
    max_idx = n - context_length - 1
    
    # 2. 随机产生 batch_size 个起始位置
    # np.random.randint 在 [0, max_idx] 之间产生随机整数
    ix = torch.randint(0, max_idx + 1, (batch_size,))
    
    # 3. 根据索引提取输入和目标
    # x: dataset[i : i+m]
    # y: dataset[i+1 : i+m+1]
    # 我们先在 CPU 上提取数据，然后一次性转为 Tensor
    x_stack = [dataset[i : i + context_length] for i in ix]
    y_stack = [dataset[i + 1 : i + context_length + 1] for i in ix]
    
    # 4. 转换为 PyTorch 张量并移动到指定设备
    # 注意：dataset 通常是 int32 或 int64，转为 torch 后通常使用 torch.long (int64)
    x = torch.from_numpy(np.array(x_stack)).to(device).long()
    y = torch.from_numpy(np.array(y_stack)).to(device).long()
    
    return x, y
```

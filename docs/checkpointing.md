# checkpointing.py — 模型检查点持久化

## 概述

本模块提供训练状态的保存与恢复功能，支持断点续训。

---

## 1. save_checkpoint — 保存检查点

### 函数 `save_checkpoint(model, optimizer, iteration, out)`

#### 参数

- `model`：PyTorch 模型实例（需要有 state_dict）
- `optimizer`：PyTorch 优化器实例（需要有 state_dict）
- `iteration`：当前训练迭代次数（int）
- `out`：保存目标，可以是文件路径字符串、PathLike 对象、或二进制文件流

#### 功能

将以下三个组件打包为一个字典，通过 `torch.save` 序列化写入磁盘：

```python
checkpoint = {
    'model_state_dict': model.state_dict(),       # 模型权重
    'optimizer_state_dict': optimizer.state_dict(), # 优化器状态（动量、步数等）
    'iteration': iteration                         # 当前迭代次数
}
```

---

## 2. load_checkpoint — 加载检查点

### 函数 `load_checkpoint(src, model, optimizer)`

#### 参数

- `src`：检查点来源，可以是文件路径字符串、PathLike 对象、或二进制文件流
- `model`：已初始化的模型实例（权重将被覆盖）
- `optimizer`：已初始化的优化器实例（状态将被覆盖）

#### 返回值

- `int`：保存时的迭代次数，用于从断点继续训练

#### 功能

1. 使用 `torch.load(src, map_location='cpu')` 加载检查点字典
   - `map_location='cpu'` 确保在无 GPU 的机器上也能加载 GPU 上保存的检查点
2. `model.load_state_dict(checkpoint['model_state_dict'])` 恢复模型权重
3. `optimizer.load_state_dict(checkpoint['optimizer_state_dict'])` 恢复优化器状态
4. 返回 `checkpoint['iteration']` 作为恢复后的起始迭代

---

## 完整源代码

```python
import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
):
    """
    保存当前训练状态。
    """
    # 1. 构建一个包含所有必要信息的字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # 2. 使用 torch.save 将字典写入目标（可以是路径或文件流）
    torch.save(checkpoint, out)

def load_checkpoint(
    src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点恢复状态，并返回保存时的迭代次数。
    """
    # 1. 加载字典
    # 使用 map_location='cpu' 是一个好习惯，可以防止在没有 GPU 的机器上加载时报错
    checkpoint = torch.load(src, map_location='cpu')
    
    # 2. 恢复模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 恢复优化器状态（动量、步数等）
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 4. 返回保存时的迭代次数
    return checkpoint['iteration']
```

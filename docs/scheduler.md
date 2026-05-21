# scheduler.py — 学习率调度器

## 概述

本模块实现带线性预热的余弦退火学习率调度策略，这是现代 Transformer 训练的标准做法。

---

## get_lr_cosine_schedule — 预热 + 余弦退火学习率

### 函数 `get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)`

#### 参数

- `it`：当前迭代次数（从 0 开始）
- `max_learning_rate`：最大学习率，预热终点/退火起点
- `min_learning_rate`：最小学习率，退火终点
- `warmup_iters`：预热步数，此阶段学习率从 0 线性增长到最大值
- `cosine_cycle_iters`：余弦退火总步数，退火阶段在此步数内完成

#### 返回值

- 当前迭代对应的学习率（float）

#### 三阶段调度

**阶段 1 — 线性预热**（`it < warmup_iters`）：
```
lr = max_lr * (it / warmup_iters)
```
学习率从 0 线性增长到 max_lr。预热的目的是让模型在训练初期平稳启动，避免因随机初始化的梯度方差过大导致训练不稳定。

**阶段 2 — 余弦退火**（`warmup_iters ≤ it ≤ cosine_cycle_iters`）：
```
decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
coeff = 0.5 * (1 + cos(π * decay_ratio))
lr = min_lr + coeff * (max_lr - min_lr)
```
`coeff` 从 1 平滑过渡到 0，使学习率以余弦曲线从 max_lr 衰减到 min_lr。

**阶段 3 — 平稳保持**（`it > cosine_cycle_iters`）：
```
lr = min_lr
```
训练后期保持最小学习率继续微调。

---

## 完整源代码

```python
import math

def get_lr_cosine_schedule(
    it: int, 
    max_learning_rate: float, 
    min_learning_rate: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int
) -> float:
    """
    计算带预热的余弦退火学习率。
    
    it: 当前迭代次数 (t)
    max_learning_rate: 最大学习率 (alpha_max)
    min_learning_rate: 最小学习率 (alpha_min)
    warmup_iters: 预热步数 (T_w)
    cosine_cycle_iters: 总退火步数 (T_c)
    """
    
    # 1. 预热阶段：线性增长
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    # 2. 退火后阶段：保持最小学习率
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # 3. 余弦退火阶段
    # 计算当前在退火阶段的进度 (0.0 到 1.0)
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    
    # 计算余弦系数：从 1.0 降到 0.0
    # math.cos(math.pi * decay_ratio) 的范围是 [1, -1]
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # 最终学习率 = 最小值 + 系数 * (最大值 - 最小值)
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
```

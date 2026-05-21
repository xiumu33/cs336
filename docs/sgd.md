# sgd.py — SGD 优化器与学习率调试实验

## 概述

本模块实现了一个带步数衰减的朴素 SGD 优化器，并包含一个学习率调试实验用于观察不同学习率对收敛的影响。

---

## 1. SGD — 带衰减的随机梯度下降

### 初始化 `__init__(params, lr=1e-3)`

- 参数校验：学习率必须 ≥ 0
- 将学习率存入 `defaults` 字典，调用父类 `torch.optim.Optimizer.__init__`

### 单步更新 `step()`

对每个参数执行：

**状态初始化**（首次运行）：
- `state['t'] = 0`：每参数独立的步数计数器

**更新公式**：
```
θ = θ - (lr / sqrt(t + 1)) * g
```

其中分母 `sqrt(t+1)` 使得学习率随步数增加而衰减（类似 AdaGrad 的衰减策略，但更简单）。

**步数递增**：`t += 1`

---

## 2. run_experiment — 学习率调试实验

### 函数 `run_experiment(learning_rate)`

运行一个简单的 10 轮优化实验来诊断学习率是否合适：

- 初始化一个 10×10 的参数矩阵 W，从 N(0, 25) 中随机采样
- 优化目标：`loss = mean(W²)`，理论最优解在 W = 0（loss = 0）
- 每轮打印当前 loss，观察收敛/发散行为

### 实验结果

| 学习率 | 表现 | 分析 |
|---|---|---|
| lr = 10 | 平稳收敛，loss 从 ~23 逐渐降到 ~3 | 合适的步长，稳定下降 |
| lr = 100 | 快速收敛，第 1 步 loss 不变（步数衰减），之后迅速到 0 | 较大步长配合衰减机制，收敛最快 |
| lr = 1000 | 发散爆炸，loss 从 ~22 爆炸到 ~10²⁴ | 步长过大直接"飞出"，验证了梯度裁剪和合适学习率的重要性 |

这个实验直观展示了：**学习率是深度学习中最关键的超参数之一**，过大导致发散，过小导致收敛缓慢。

---

## 完整源代码

```python
import torch
import math
from collections.abc import Callable
from typing import Optional

# 1. 按照 PDF 实现带有衰减逻辑的 SGD
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        """
        params: 传入模型需要优化的参数（通常是 model.parameters()）。
        defaults: 一个字典，存储默认超参数（如学习率 lr）。
        调用 super().__init__ 后，PyTorch 会将参数组织在 self.param_groups 中
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 获取该参数对应的状态字典（用于记录步数 t）
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                
                t = state["t"]
                grad = p.grad.data
                
                # 执行 PDF 中的更新公式 (20): p = p - (lr / sqrt(t+1)) * grad
                p.data -= lr / math.sqrt(t + 1) * grad
                
                # 更新步数
                state["t"] += 1
        return loss

# 2. 运行学习率调试实验 (Problem learning_rate_tuning)
def run_experiment(learning_rate):
    print(f"\n--- Testing LR = {learning_rate} ---")
    # 初始化权重 (10x10 矩阵)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=learning_rate)
    
    for t in range(10): # 只跑 10 轮
        opt.zero_grad()
        # 目标函数：Loss = weights^2 的平均值（极小值点应在全 0 处）
        loss = (weights**2).mean()
        print(f"Iter {t}: Loss = {loss.item():.4f}")
        loss.backward()
        opt.step()

# 测试不同学习率
lrs_to_test = [1e1, 1e2, 1e3]
for lr in lrs_to_test:
    run_experiment(lr) 


"""
--- Testing LR = 10.0 ---
Iter 0: Loss = 23.0576
Iter 1: Loss = 14.7569
Iter 2: Loss = 10.8781
Iter 3: Loss = 8.5110
Iter 4: Loss = 6.8939
Iter 5: Loss = 5.7158
Iter 6: Loss = 4.8205
Iter 7: Loss = 4.1193
Iter 8: Loss = 3.5573
Iter 9: Loss = 3.0988

--- Testing LR = 100.0 ---
Iter 0: Loss = 25.5835
Iter 1: Loss = 25.5835
Iter 2: Loss = 4.3894
Iter 3: Loss = 0.1050
Iter 4: Loss = 0.0000
Iter 5: Loss = 0.0000
Iter 6: Loss = 0.0000
Iter 7: Loss = 0.0000
Iter 8: Loss = 0.0000
Iter 9: Loss = 0.0000

--- Testing LR = 1000.0 ---
Iter 0: Loss = 21.8585
Iter 1: Loss = 7890.9087
Iter 2: Loss = 1362884.1250
Iter 3: Loss = 151606320.0000
Iter 4: Loss = 12280111104.0000
Iter 5: Loss = 775015628800.0000
Iter 6: Loss = 39786769285120.0000
Iter 7: Loss = 1711797065220096.0000
Iter 8: Loss = 63093176952422400.0000
Iter 9: Loss = 2025992011177263104.0000

"""
```

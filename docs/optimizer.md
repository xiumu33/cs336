# optimizer.py — 优化器与梯度工具

## 概述

本模块实现了 AdamW 优化器和全局梯度裁剪功能，均为从零手写，不依赖 PyTorch 内置的 AdamW 实现。

---

## 1. AdamW — 自适应矩估计优化器（解耦权重衰减）

### 初始化 `__init__(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`

**参数校验**：
- `lr` ≥ 0
- `beta1` ∈ [0, 1)，`beta2` ∈ [0, 1)
- `eps` ≥ 0

**超参数存储**：将 `lr`、`betas`、`eps`、`weight_decay` 存入 `defaults` 字典，传递给父类 `torch.optim.Optimizer`。

### 单步更新 `step()`

对每个参数组中的每个参数执行：

**状态初始化**（首次运行）：
- `step`：步数计数器，初始为 0
- `exp_avg`（m）：一阶矩（梯度指数移动平均），初始为全零
- `exp_avg_sq`（v）：二阶矩（梯度平方的指数移动平均），初始为全零

**更新矩估计**：
```
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g²
```

**偏差校正**：
```
bias_correction1 = 1 - beta1^t
bias_correction2 = 1 - beta2^t
step_size = lr * sqrt(bias_correction2) / bias_correction1
```

**参数更新**：
```
θ = θ - step_size * m / (sqrt(v) + eps)
```

**解耦权重衰减**（AdamW 核心特性）：
```
θ = θ - lr * weight_decay * θ
```
权重衰减与自适应学习率更新完全独立，这是 AdamW 区别于原始 Adam（在梯度中加 L2 正则项）的关键差异。

---

## 2. clip_gradient_norm — 全局梯度范数裁剪

### 函数 `clip_gradient_norm(parameters, max_norm)`

用于防止梯度爆炸的"自动刹车系统"：

1. **过滤**：筛出所有 `p.grad is not None` 的参数
2. **计算全局 L2 范数**：`total_norm = sqrt(∑‖g_i‖²)`，其中 `g_i` 是每层参数的梯度。使用 `.detach()` 确保范数计算不进入计算图。
3. **裁剪判断**：若 `total_norm > max_norm`，缩放系数 `coef = max_norm / (total_norm + ε)`，原地乘以所有梯度：`g_i *= coef`

这种全局范数裁剪按照所有层梯度的总"长度"进行等比例缩放，保留梯度方向，只限制步长。

---

## 完整源代码

```python
import torch
import math
from torch.optim import Optimizer
from collections.abc import Iterable


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # 1. 基本参数检查
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        # 2. 将超参数存入 defaults 字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """执行单步优化更新"""
        loss = None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # 3. 状态初始化 (第一次运行步时执行)
                if len(state) == 0:
                    state['step'] = 0
                    # m: 一阶矩 (梯度的指数移动平均)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v: 二阶矩 (梯度平方的指数移动平均)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # 4. 更新矩估计 (Algorithm 1)
                # m = beta1 * m + (1 - beta1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = beta2 * v + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 5. 计算偏差校正后的学习率 alpha_t
                # 这一步是为了消除初始值为 0 带来的偏移
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)

                # 6. 更新参数：theta = theta - alpha_t * m / (sqrt(v) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 7. 应用解耦的权重衰减 (AdamW 的核心特性)
                # theta = theta - alpha * lambda * theta
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

        return loss


def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    """
    实现梯度裁剪（Global Norm Clipping）。
    """
    # 1. 过滤掉没有梯度的参数
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    
    # 2. 计算全局 L2 范数
    total_norm = 0.0
    for p in params_with_grad:
        # 使用 .detach() 确保计算范数的操作不计入计算图
        param_norm = torch.norm(p.grad.detach(), p=2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 3. 检查是否超过阈值
    eps = 1e-6  # 数值稳定性常数
    if total_norm > max_norm:
        # 计算缩放因子
        clip_coef = max_norm / (total_norm + eps)
        
        # 4. 原地（in-place）修改每个参数的梯度
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)
```

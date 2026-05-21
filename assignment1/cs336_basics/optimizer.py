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
                # 这是一个专门为优化器设计的复合算子，名字可以拆解为：add (加) + constant (常数) + div (除)。
                # p.addcdiv_(tensor1, tensor2, value=1.0)。 p=p+value×( tensor1 / tensor2 )
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 7. 应用解耦的权重衰减 (AdamW 的核心特性)
                # theta = theta - alpha * lambda * theta
                # p.add_(other, alpha=1.0) p=p+(alpha×other)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

        return loss


def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    """
    实现梯度裁剪（Global Norm Clipping）。
    
    参数:
        parameters: 可迭代的参数列表（通常是 model.parameters()）
        max_norm: 允许的最大梯度的 L2 范数 (M)
    想象你在下山（优化模型），正常的步子很稳。但突然遇到一个极陡的悬崖（梯度爆炸），如果不加控制，你这一步跨出去可能会直接飞出景区（权重更新过大，Loss 变成 NaN）。
    梯度裁剪就像一个**“自动刹车系统”**：它会检查你所有步子的总长度。如果总长度超过了安全阈值 M，它就把你的步子按比例收回来，确保你依然走在正确的方向上，但步长在安全范围内。
    """
    # 1. 过滤掉没有梯度的参数
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    
    # 2. 计算全局 L2 范数
    # 我们需要把所有参数梯度的平方和加起来，最后开根号
    total_norm = 0.0
    for p in params_with_grad:
        # 使用 .detach() 确保计算范数的操作不计入计算图
        param_norm = torch.norm(p.grad.detach(), p=2) # 算出当前这一层梯度的长度 L_i
        total_norm += param_norm.item() ** 2         # 把 L_i 的平方累加到总和中
    total_norm = total_norm ** 0.5
    
    # 3. 检查是否超过阈值
    eps = 1e-6  # 数值稳定性常数
    if total_norm > max_norm:
        # 计算缩放因子
        clip_coef = max_norm / (total_norm + eps)
        
        # 4. 原地（in-place）修改每个参数的梯度
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)
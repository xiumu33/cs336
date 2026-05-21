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
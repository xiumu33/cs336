# nn.py — 神经网络基础组件

## 概述

本模块实现了构建 Transformer 语言模型所需的所有核心神经网络组件，包括线性层、嵌入层、归一化、激活函数、注意力机制、Transformer 块和完整的语言模型。所有组件均为手写实现，继承自 `torch.nn.Module`。

---

## 1. Linear — 全连接线性层

### 初始化 `__init__(in_features, out_features)`

- 创建形状为 `(out_features, in_features)` 的权重矩阵 `W`，注册为 `nn.Parameter`
- 权重初始化采用 **截断正态分布**（Xavier 初始化变体），标准差 `σ = sqrt(2 / (d_in + d_out))`，截断范围 `[-3σ, 3σ]`
- 没有偏置项（bias-free），这是 Llama 风格的设计选择

### 前向传播 `forward(x)`

- 使用 `torch.einsum('...i, oi -> ...o', x, self.weight)` 实现矩阵乘法
- `einsum` 的好处是可以自适应处理任意 batch 维度（2D、3D、4D 均可）

---

## 2. Embedding — Token 嵌入层

### 初始化 `__init__(num_embeddings, embedding_dim)`

- 创建形状为 `(num_embeddings, embedding_dim)` 的权重矩阵
- 权重以截断正态分布初始化，标准差 `σ = 1.0`，截断范围 `[-3σ, 3σ]`

### 前向传播 `forward(token_ids)`

- 直接通过 PyTorch 高级索引查表：`self.weight[token_ids]`
- 输入的 token_ids 形状为 `(...)` 任意维度，输出为 `(..., embedding_dim)`

---

## 3. RMSNorm — Root Mean Square 层归一化

这是 Llama 系列模型使用的归一化方法，相比 LayerNorm 去掉了"减去均值"的操作，计算更快。

### 初始化 `__init__(d_model, eps=1e-5)`

- 可学习的缩放参数 `weight`（即公式中的 `g`），**初始化为全 1**
- 小常数 `eps` 防止除零

### 前向传播 `forward(x)`

1. 将输入转为 float32（防止平方计算溢出）
2. 沿最后一维计算均方根：`rms = sqrt(mean(x²) + ε)`
3. 归一化并乘可学习参数：`result = (x_float / rms) * self.weight`
4. 转回原始数据类型返回

---

## 4. silu_fn — SiLU 激活函数

公式：`x * sigmoid(x)`，也称为 Swish 激活函数。是一个独立函数（非 nn.Module），供 SwiGLU 内部调用。

---

## 5. SwiGLU — SwiGLU 前馈网络

这是 Llama 使用的 FFN 变体，相比标准 FFN 多一个门控支路。

### 初始化 `__init__(d_model, d_ff)`

- `W1`（Linear）：门控支路升维层，d_model → d_ff
- `W3`（Linear）：信号支路升维层，d_model → d_ff
- `W2`（Linear）：降维层，d_ff → d_model

### 前向传播 `forward(x)`

1. 门控支路：`gate = silu(W1(x))`
2. 信号支路：`signal = W3(x)`
3. 逐元素相乘：`gate * signal`
4. 降维投影：`W2(gate * signal)`

---

## 6. RotaryPositionalEmbedding — 旋转位置编码 (RoPE)

RoPE 通过旋转变换将位置信息编码到 Q/K 向量中，使注意力分数能感知 token 间的相对位置。

### 初始化 `__init__(theta, d_k, max_seq_len)`

1. 计算频率：`freqs = 1 / (theta^(2k/d_k))`，共 `d_k/2` 个频率（因为旋转成对进行）
2. 生成位置序列 `[0, 1, ..., max_seq_len-1]`
3. 外积得到 `(max_seq_len, d_k/2)` 的角度矩阵
4. 预计算 cos/sin 缓存，注册为 buffer（`persistent=False`，不存 state_dict）

### 前向传播 `forward(x, token_positions)`

1. 根据位置索引提取对应 cos/sin
2. 维度对齐：当 x 含 Head 维度（4D）且 cos 含 Batch 维度（3D）时，在 Head 维插入维度
3. 拆分为偶/奇维度：`x_even = x[..., 0::2]`, `x_odd = x[..., 1::2]`
4. 施加旋转：`output_even = x_even*cos - x_odd*sin`, `output_odd = x_even*sin + x_odd*cos`

---

## 7. softmax — 数值稳定的 Softmax

### 函数 `softmax(x, dim=-1)`

1. 减去维度最大值：`x_stable = x - max(x, dim=dim)`
2. 计算指数：`exp(x_stable)`
3. 归一化：`exp / sum(exp)`

减去最大值的操作确保指数输入 ≤ 0，防止 `exp` 溢出为 `inf`。

---

## 8. scaled_dot_product_attention — 缩放点积注意力

### 函数 `scaled_dot_product_attention(Q, K, V, mask=None)`

1. 计算注意力分数：`scores = Q @ K^T / sqrt(d_k)`，使用 einsum 实现
2. 应用 mask：将 mask 为 False 的位置设为 `-inf`
3. Softmax 归一化（在 Key 序列维度上）
4. 加权求和：`output = probs @ V`

支持多头注意力中任意形状的 `(... , n, d_k)` 张量。

---

## 9. CausalSelfAttention — 多头因果自注意力

### 初始化 `__init__(d_model, num_heads, bias, max_seq_len, theta)`

- `Q/K/V 投影层`：各为一个 Linear(d_model, d_model)
- `输出投影层`：Linear(d_model, d_model)
- `RoPE 模块`：如果传入了 theta 和 max_seq_len 则初始化，否则为 None

### 前向传播 `forward(x, token_positions=None)`

1. 投影：分别通过 Q/K/V 的 Linear 层
2. 拆分头：使用 `einops.rearrange` 将 `(..., s, d_model)` 变为 `(..., h, s, d_k)`
3. 应用 RoPE（如果模块存在）：对 Q/K 施加旋转位置编码
4. 构造下三角因果 mask：`torch.tril(ones(s, s))` 确保每个 token 只能看到自身及之前的内容
5. 调用 `scaled_dot_product_attention(q, k, v, mask=mask)`
6. 合并头：`rearrange` 将 `(..., h, s, d_k)` 变回 `(..., s, d_model)`
7. 输出投影：`output_proj(attn_out)`

---

## 10. TransformerBlock — Transformer 块

一个标准的 Transformer 块 = 自注意力子层 + 前馈网络子层，均带残差连接。

### 初始化参数

- `use_rms_norm`：是否使用 RMSNorm（禁用则为 nn.Identity）
- `norm_mode`：`"pre"`（Llama 默认）或 `"post"`（原始 Transformer）
- `ffn_type`：`"swiglu"`（SwiGLU FFN）或 `"silu"`（标准 SiLU FFN，此时 d_ff 会自动设为 4*d_model）

### 前向传播 `forward(x, token_positions)`

**Pre-norm 模式**（默认）：
```
x = x + attn(ln1(x), token_positions)
x = x + ffn(ln2(x))
```

**Post-norm 模式**（消融实验用）：
```
x = ln1(x + attn(x, token_positions))
x = ln2(x + ffn(x))
```

Post-norm 通常更难训练，需要配合学习率预热使用。

---

## 11. TransformerLM — 完整的 Transformer 语言模型

### 初始化 `__init__(vocab_size, max_seq_len, d_model, num_layers, ...)`

1. `token_embeddings`：Embedding 层（vocab_size → d_model）
2. `layers`：N 个 TransformerBlock 的 ModuleList
3. `ln_final`：最终 RMSNorm（如果启用）；否则为 nn.Identity
4. `lm_head`：Linear 层（d_model → vocab_size），输出 logits

所有消融实验参数（use_rms_norm, norm_mode, ffn_type）透传给每个 TransformerBlock。

### 前向传播 `forward(token_ids)`

1. 生成位置信息：`arange(s)` 扩展为 `(batch, seq_len)` 用于 RoPE
2. Token 嵌入：`x = token_embeddings(token_ids)`
3. 逐层通过 TransformerBlock
4. 最终归一化：`x = ln_final(x)`
5. 投影到词表空间：`lm_head(x)`，返回 logits

### 文本生成 `generate(prompt_ids, max_new_tokens, eos_token_id, temperature, top_p)`

自回归生成循环：
1. 截取最后 `context_length` 个 token 作为模型输入
2. 前向传播获取 logits，只取最后一个位置的预测
3. Temperature 缩放：`logits = logits / temperature`
4. Top-p 核采样过滤：将累积概率超过 p 的低概率 token 设为 -∞
5. Softmax → `torch.multinomial` 采样得到下一个 token
6. 拼接到已生成序列末尾
7. 遇到 EOS token 提前停止

内部方法 `_top_p_filter(logits, p)`：对 logits 按降序排序，保留累积概率恰好达到 p 的最小 token 集合，其余设为 -∞。

---

## 完整源代码

```python
import torch
import torch.nn as nn
import math
from einops import rearrange

"""
全连接层
    1. init: 定义一个W矩阵
    2. forward: 如输入x进行矩阵乘法
"""

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        # 1. 定义权重 W (形状: out x in)
        # 注意要把 device 和 dtype 传进去，确保张量创建在正确位置
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # nn.Parameter 的作用是告诉 PyTorch："这个张量是模型的一部分，它是需要通过训练来学习的权重（Weights）。"
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # 2. 初始化权重 (截断正态分布)  Xavier 初始化
        # 根据 3.4.1: sigma^2 = 2 / (din + dout)
        std = (2.0 / (in_features + out_features)) ** 0.5
        # PDF 要求截断在 [-3sigma, 3sigma]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 einsum 处理，适应各种 Batch 维度情况
        # '...i' 表示输入 x 的最后一个维度 (in_features)
        # 'oi' 表示权重 W (out_features, in_features)
        # '-> ...o' 表示输出保留前面的维度，最后一个维度变成 out_features
        return torch.einsum('...i, oi -> ...o', x, self.weight)



class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 1. 必须初始化为全 1 (ones)
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, sequence_length, d_model)

        in_dtype = x.dtype
        # 2. 转换为 float32 以防平方计算时溢出
        x_float = x.to(torch.float32)

        # 3. 计算均方根 (Root Mean Square)
        # 公式: rms = sqrt( mean(x^2) + eps )
        # dim=-1 表示在隐藏层维度计算，keepdim=True 方便后续除法自动广播
        """
        在 PyTorch（以及 NumPy）中，广播（Broadcasting） 是指在对两个形状不同的张量进行算术运算时，系统自动"扩展"较小张量的维度，使其与较大张量匹配的机制。
        要使两个张量是可广播的（Broadcastable），必须满足以下核心规则：
        核心规则：从右往左看
            比较两个张量的形状时，要从最后一个维度（最右边）开始往前检查。对于每一对对应的维度，必须满足以下 两个条件之一：
                1.这两个维度的值相等。
                2.其中一个维度的值是 1。
            如果其中一个张量的维度较少，系统会自动在它的左侧补 1，直到两者的维度数量相等，然后再按上述规则检查。
        """
        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)

        # 4. 归一化并乘以可学习的增益参数 g
        result = (x_float / rms) * self.weight

        # 5. 转回原始类型
        return result.to(in_dtype)
        

def silu_fn(in_features):

    return in_features * torch.sigmoid(in_features)



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        # W1 和 W3 是并行升维层: d_model -> d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        # W2 是降维层: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model, device, dtype)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        gate = silu_fn(self.w1(x))
        signal = self.w3(x)
        # 形状: [..., d_ff]
        return self.w2(gate * signal)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化 RoPE 模块
        theta: 基准频率 (通常为 10000)
        d_k: 每个 Head 的维度 (必须是偶数)
        max_seq_len: 最大序列长度
        """
        super().__init__()
        self.d_k = d_k
        
        # 1. 计算频率频率 omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率，因为旋转是成对进行的
        # arange(0, d_k, 2) 产生 [0, 2, 4, ..., d_k-2]，对应公式中的2k-2(k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)  # 形状: (d_k/2,)
        
        # 2. 创建位置序列 [0, 1, ..., max_seq_len - 1]
        t = torch.arange(max_seq_len, device=device).float()  # 形状: (max_seq_len,)
        
        # 3. 计算所有位置的所有角度 (外积)
        # freqs_matrix 形状: (context_length, d_k/2)
        freqs_matrix = torch.outer(t, freqs)
        
        # 4. 预计算 cos 和 sin 并作为 buffer 注册
        # 使用 persistent=False 确保这些缓存不会被保存在 state_dict 中 (因为可以随时重新生成)
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 提取 cos/sin (..., context_length, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐
        # 只有当 x 是 4D (含 Head 维) 且 cos 是 3D (含 Batch 维) 时，才需要手动插入 Head 维。
        # 对于 test_rope 这种 3D x vs 2D cos 的情况，PyTorch 会自动左侧补 1，无需操作。
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 确保类型一致
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 3. 拆分并旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output
    

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # 1. 为了数值稳定性，减去指定维度上的最大值
        # dim=-1 通常是 Transformer 中的隐藏层或词表维度
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_stable = x - x_max
        
        # 2. 计算指数
        exp_x = torch.exp(x_stable)
        
        # 3. 计算分母的各指数之和
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
        
        # 4. 计算最终结果
        return exp_x / sum_exp

def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Q: (batch_size, ..., n, d_k)
    K: (batch_size, ..., m, d_k)
    V: (batch_size, ..., m, d_v)
    mask: (..., n, m) 或者是可以广播到该形状的布尔张量 (True 表示关注, False 表示屏蔽)
    """
    d_k = Q.size(-1)
    
    # 1. 计算分数: Q @ K^T / sqrt(d_k)
    # 交换最后两个维度进行矩阵乘法
    # 形状变化: (..., n, d_k) @ (..., d_k, m) -> (..., n, m)
    scores = torch.einsum('...nk, ...mk-> ...nm', Q, K) / math.sqrt(d_k)
    
    # 2. 应用掩码
    if mask is not None:
        # PDF 要求: 把 mask 为 False 的地方填入 -inf
        # 注意: 使用一个足够小的负数，通常 float('-inf') 在 torch 中是安全的
        scores = scores.masked_fill(mask == False, float('-inf'))
    
    # 3. Softmax 归一化 (在最后一个维度 m 上)
    # 注意: 这里的 dim=-1 指向的是 Key 序列的长度维度
    probs = softmax(scores, dim=-1)
    
    # 4. 对 Value 加权求和
    # 形状变化: (..., n, m) @ (..., m, d_v) -> (..., n, d_v)
    output = torch.einsum('...nm, ...mk-> ...nk', probs, V)
    
    return output

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = False, 
                 max_seq_len=None, theta=None, 
                 device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 1. 定义 Q, K, V 的投影层（PDF 要求 3 次矩阵乘法）
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # 2. 定义输出投影
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # 3. 实例化 RoPE
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        b, s, d = x.shape
        
        # 步骤 1 & 2: 投影与拆分头 (保持不变)
        # q = self.q_proj(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        # k = self.k_proj(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        # v = self.v_proj(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        q = rearrange(self.q_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        k = rearrange(self.k_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        v = rearrange(self.v_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        

        # 步骤 3: 应用 RoPE
        # 只有当模块存在时才应用
        if self.rope is not None:
            # 如果没传位置，且 RoPE 需要位置，则生成默认位置
            if token_positions is None:
                # 适配各种 Batch 维度，使用 expand 比 repeat 更高效
                batch_dims = x.shape[:-2]
                token_positions = torch.arange(s, device=x.device).expand(*batch_dims, s)
            
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 生成下三角矩阵
        mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))

        # 步骤 5: SDPA (SDPA 内部应能处理 mask 为 None 的情况)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # 步骤 6 & 7: 合并与输出投影
        attn_out = rearrange(attn_out, '... h s d -> ... s (h d)')
        return self.output_proj(attn_out)


import torch
import torch.nn as nn
from .nn import Embedding, RMSNorm, Linear, CausalSelfAttention, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int,
                 theta: float, device=None, dtype=None, 
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",   # 选项: "pre", "post"
                 ffn_type: str = "swiglu"  # 选项: "swiglu", "silu"
                 ):
        super().__init__()
        self.use_rms_norm = use_rms_norm
        self.norm_mode = norm_mode
        self.ffn_type = ffn_type

        # 1. 初始化 Attention
        # RoPE 的开关由外部传入的 theta 是否为 None 控制
        self.attn = CausalSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            max_seq_len=max_seq_len, 
            theta=theta,
            device=device, 
            dtype=dtype
        )

        # 2. 初始化 Norm 层 (Ablation 1)
        if use_rms_norm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            # 如果禁用 Norm，使用 Identity 占位，它直接返回输入，不改变任何东西
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        # 3. 初始化 FFN (Ablation 4)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            # 标准 FFN: x -> Linear -> SiLU -> Linear -> out
            # 注意: 为了公平对比，通常 SiLU FFN 的 d_ff 应该是 4 * d_model
            d_ff = 4 * d_model
            self.ffn = nn.Sequential(
                Linear(d_model, d_ff, device=device, dtype=dtype),
                nn.SiLU(),
                Linear(d_ff, d_model, device=device, dtype=dtype)
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm (Llama 默认, 也是作业基准)
        # 公式: x = x + Sublayer(Norm(x))
        if self.norm_mode == "pre":
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
        
        # Post-norm (原始 Transformer, Ablation 2)
        # 公式: x = Norm(x + Sublayer(x))
        elif self.norm_mode == "post":
            # 注意: Post-norm 通常很难训练，需要 Warmup
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))
            
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, 
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float, 
                 device=None, dtype=None,
                 # 新增实验参数
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",
                 ffn_type: str = "swiglu"):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # 1. Token Embedding 层
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # 2. 堆叠 Transformer Blocks
        # 将实验参数透传给每一个 Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, max_seq_len, rope_theta, 
                device=device, dtype=dtype,
                use_rms_norm=use_rms_norm,
                norm_mode=norm_mode,
                ffn_type=ffn_type
            )
            for _ in range(num_layers)
        ])
        
        # 3. 最终的输出层
        # 如果全局禁用了 Norm，这里的 Final Norm 也要变成 Identity
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            """
            forward(input):
                return input
            """
            self.ln_final = nn.Identity()


            
        # 最后是一个 Linear 层映射回词表大小 (LM Head)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        b, s = token_ids.shape
        
        # 准备位置信息用于 RoPE, shape: [S] -> [1, S] -> [B,S]
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)
        
        # 1. Embedding
        x = self.token_embeddings(token_ids)
        
        # 2. 逐层通过 Transformer Blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
            
        # 3. 最终归一化 (如果 use_rms_norm=False，这里就是直通)
        x = self.ln_final(x)
        
        # 4. 投影到词表空间得到 logits
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: torch.Tensor, 
        max_new_tokens: int, 
        eos_token_id: int = None, 
        temperature: float = 1.0, 
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        从模型生成文本 ID 序列。
        
        参数:
            prompt_ids: 提示词 ID (Batch, Seq_len)
            max_new_tokens: 最多生成的词数
            eos_token_id: 停止生成的 Token ID (如 <|endoftext|>)
            temperature: 温度系数 (越高越随机，越低越确定)
            top_p: 核采样阈值
        """
        # 设置为评估模式
        self.eval()
        
        # 将输入拷贝一份，避免修改原始数据
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # 1. 裁剪输入：模型只能处理 context_length 长度的内容
            # 如果生成的序列过长，只取最后的 context_length 个词
            idx_cond = generated[:, -self.context_length:]
            
            # 2. 前向传播得到 Logits
            # 我们只关心最后一个时间步的预测
            logits = self.forward(idx_cond) # (Batch, T, Vocab)
            logits = logits[:, -1, :]      # (Batch, Vocab)
            
            # 3. 应用温度 (Temperature)
            if temperature != 1.0:
                logits = logits / (temperature + 1e-8) # 加个 epsilon 防止除以 0
            
            # 4. 应用 Top-P (Nucleus Sampling) 过滤
            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)
            
            # 5. 归一化并采样
            probs = softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (Batch, 1)
            
            # 6. 拼接新词
            generated = torch.cat((generated, next_token), dim=1)
            
            # 7. 如果遇到了 EOS，提前结束生成
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return generated

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """内部工具函数：执行 Top-P 截断"""
        # 对词表分值进行降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # 计算累积概率分布
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
        
        # 创建掩码：我们要去掉累积概率超过 p 的 Token
        # 逻辑：保留最小的集合 V(p)，使其概率之和 >= p
        # 我们把所有超过 p 的位置标记为 True（需要移除）
        sorted_indices_to_remove = cumulative_probs > p
        
        # 关键修正：确保至少保留第一个词（最高概率词），
        # 并且我们要保留第一个"使概率超过 p"的那个词。
        # 做法是把标记位向右移动一格。
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 将被移除的 Token 分数设为负无穷
        # 这里需要利用 scatter 将排序后的掩码映射回原始词表索引位置
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
```

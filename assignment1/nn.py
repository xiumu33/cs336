import torch
import torch.nn as nn
import math
from einops import rearrange

class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device=None,dtype=None):
        super().__init__()
        factory_kwargs={
            'device':device,
            'dtype':dtype
        }
        self.weight=nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))


        nn.init.trunc_normal_(self.weight,mean=0.0,std=1.0,a=-3.0,b=3.0)

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.weight[token_ids]

class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features

        factory_kwargs={
            'device':device,
            'dtype':dtype
        }

        self.weight=nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))

        std=(2.0/(in_features+out_features)**0.5)
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3*std,b=-3*std)

    def forward(self,x:torch.Tensor)->torch.Tensor:

        return torch.einsum('...i, oi -> ...o',x,self.weight)
    

def softmax(x:torch.Tensor,dim:int=-1)->torch.Tensor:
    x_max=torch.max(x,dim=dim,keepdim=True).values
    x_stable=x-x_max

    exp_x=torch.exp(x_stable)

    sum_exp=torch.sum(exp_x,dim=dim,keepdim=True)

    return exp_x/sum_exp

def scaled_dot_product_attention(
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask:torch.Tensor=None
)->torch.Tensor:
    """
    Q:[...,n,d_k]
    K:[...,m,d_k]
    V:[...,m,d_v]
    mask:[n,m],True means save,and False means hide
    and if self-attention n==m
    """
    d_k=Q.size(-1)

    scores=torch.einsum("...nk, ...mk -> nm",Q,K)/math.sqrt(d_k)

    if mask is not None:
        scores=scores.masked_fill(mask==False,float('-inf'))

    probs=softmax(scores,dim=-1)

    output=torch.einsum("...nm, ...mk -> ...nk",probs,V)
    return output

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta:float,
            d_k:int,
            context_length:int,
            device=None
    ):
        super().__init__()
        
        self.d_k=d_k
        powers=torch.arange(0,d_k,2,device=device).float()/d_k

        freqs=1.0/(theta**powers)

        t=torch.arange(context_length,device=device).float()
        freqs_matrix=torch.outer(t,freqs)

        self.register_buffer('cos_cached',freqs_matrix.cos(),persistent=False)
        self.register_buffer('sin_cached',freqs_matrix.sin(),persistent=False)

    def forward(
        self,
        x:torch.Tensor,
        token_positions:torch.Tensor
    )->torch.Tensor:
        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]

        if x.ndim >cos.ndim and cos.ndim>=3:
            cos=cos.unsqueeze(1)
            sin=sin.unsqueeze(1)
        cos=cos.to(x.dtype)
        sin=sin.to(x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output

class CasualSelfAttention(nn.Module):
    def __init__(
            self,
            d_model:int,
            num_heads:int,
            max_seq_len:None,
            theta=None,
            device=None,
            dtype=None
    ):
        super().__init__()

        assert d_model %num_heads==0

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads

        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)

        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope=RotaryPositionalEmbedding(theta,self.d_k,max_seq_len,device=device)

        else:
            self.rope=None

        
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        """
        b:batch size
        s:sequenceLength
        d:dimension
        """
        b,s,d=x.shape
        q=rearrange(self.q_proj(x),'... s (h d) -> ... h s d',h=self.num_heads)
        k=rearrange(self.k_proj(x),'... s (h d) -> ... h s d',h=self.num_heads)
        v=rearrange(self.v_proj(x),'... s (h d) -> ... h s d',h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions=torch.arange(s,device=x.device).expand(b,s)

            q=self.rope(q,token_positions)
            k=self.rope(k,token_positions)

        mask=torch.tril(torch.ones(s,s,device=x.device,dtype=torch.bool))
        attn_out=scaled_dot_product_attention(Q=q,K=k,V=v,mask=mask)

        attn_out=rearrange(attn_out,'... h s d -> ... s (h d)')
        return self.output_proj(attn_out)


def silu_fn(in_features):
    return in_features*torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model:int,
            d_ff:int,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.d_ff=d_ff
        self.d_model=d_model
        self.w1=Linear(d_model,d_ff,device,dtype)
        self.w3=Linear(d_model,d_ff,device,dtype)
        self.w2=Linear(d_ff,d_model,device,dtype)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        gate=silu_fn(self.w1(x))
        signal=self.w3(x)

        return self.w2(gate*signal)
    

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model:int,
            eps:float=1e-5,
            device=None,
            dtype=None
    ):
        super().__init__()
        factory_kwargs={
            'device':device,
            'dtype':dtype
        }
        self.weight=nn.Parameter(torch.ones(d_model,**factory_kwargs))
        self.eps=eps
    def forward(self,x:torch.Tensor)->torch.Tensor:    
        # x:(batch_size,length,d_model)
        in_dtype=x.dtype

        x_float=x.to(torch.float32)

        # rms=sqrt(mead(x^2)+eps)

        ms=x_float.pow(2).mean(dim=-1,keepdim=True)
        rms=torch.sqrt(ms+self.eps)

        result=(x_float/rms)*self.weight

        return result.to(in_dtype)
    

"""
norm1+casualSelfAttention
norm2+FFN
"""
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        d_ff:int,
        max_seq_len:int,
        theta:float,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.attn=CasualSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )

        self.ln1=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.ln2=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        
        self.ffn=SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)




    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        x=x+self.attn(self.ln1(x),token_positions=token_positions)

        x=x+self.ffn(self.ln2(x))

        return x


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size:int,
            max_seq_len:int,
            d_model:int,
            num_layers:int,
            num_heads:int,
            d_ff:int,
            rope_theta:float,
            device=None,
            dtype=None,
            use_rms_norm:bool=True,
            norm_mode:str='pre',
            ffn_type:str='swiglu'
    ):
        super().__init__()
        self.max_seq_len=max_seq_len

        self.token_embeddings=Embedding(vocab_size,d_model,device=device,dtype=dtype)

        self.layers=nn.ModuleList([
            TransformerBlock(
                d_model,num_heads,d_ff,max_seq_len,rope_theta,
                device=device,dtype=dtype,
                use_rms_norm=use_rms_norm,
                norm_mode=norm_mode,
                ffn_type=ffn_type
            )
            for _ in range(num_layers)
        ])

        if use_rms_norm:
            self.ln_final=RMSNorm(d_model,device=device,dtype=dtype)
        else:
            self.ln_final=nn.Identity()

        self.lm_head=Linear(d_model,vocab_size,device=device,dtype=dtype)

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        b,s=token_ids.shape

        # for rope
        token_positions=torch.arange(s,device=token_ids.device).unsqueeze(0).expand(b,s)

        # 1. embedding
        x=self.token_embeddings(token_ids)

        # 2. 
        for layer in self.layers:
            x=layer(x,token_positions=token_positions)
        
        # 3. 
        x=self.ln_final(x)

        # 4. 
        return self.lm_head(x)
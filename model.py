import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model:
    """Store parameters for GQA and other model components"""
    def __init__(self, attention_heads, kv_heads, head_dim, hidden_size, dropout, flash_attn=True):
        self.attention_heads = attention_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.flash_attn = flash_attn


class RMSNorm(nn.Module):
    def __init__(self, dim : int, eps: float=1e-5):
        super().__init__()
        self.weight = torch.ones(dim)
        self.eps = eps
    def norm(self, X):
        return X*torch.rsqrt(X.pow(2).mean(-1, keepdim = 1) +self.eps)
    def forward(self, X):
        return (self.weight*self.norm(X.float())).type_as(X)
    


def precompute_freqs(dim: int, end: int= int(32* 1024), rope_base: float = 1e6 , rope_scaling: dict = None):
    freqs, attn_factor = 1.0/rope_base**torch.arange(0, dim, 2)/dim, 1
    if rope_scaling is not None:
        orin_size, factor, highb, lowb, attn_factor = (
        rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
        rope_scaling.get("high_freqs_boundary", 32.0), rope_scaling.get("low_freqs_boundary", 1.0), rope_scaling.get("attn_factor", 1.0)
        )
        if end/orin_size > 1.0:
            cyc_notion = lambda b: (dim*math.log(orin_size/b*2*math.pi))/(2*math.log(rope_base))
            low, high = max(math.floor(cyc_notion(lowb)), 0), min(math.ceil(cyc_notion(highb)), dim//2-1)
            ramp = torch.clamp((torch.arange(dim//2, device= freqs).float()-low)/max(high-low, 0.001), min=0, max=1)
            fraqs= fraqs*(1-ramp + ramp/factor)
    t = torch.arange(end, device= fraqs)
    fraqs = torch.outer(t, fraqs).float
    freqs_sin = torch.cat([torch.sin(fraqs), torch.sin(fraqs)], dim= -1)* attn_factor
    freqs_cos= torch.cat([torch.cos(fraqs),torch.cos(fraqs)],dim=-1)* attn_factor
    return freqs_sin, freqs_cos
    
def apply_pos(q,k,sin, cos, unsqueeze_dim =1):
    def retate(x): 
        return torch.cat((-x[...,x.shape(-1)//2:],x[..., :x.shape(-1)//2]), dim=-1)
    q_embad= ((q*cos.unsqueeze(unsqueeze_dim))+(retate(q)*sin.unsqueeze(unsqueeze_dim)))
    k_embad = ((k*cos.unsqueeze(unsqueeze_dim)+(retate(q)*sin.unsqueeze(unsqueeze_dim))))
    return q_embad, k_embad


    
def repeat_kv(x: torch.Tensor, n):
    batch_size, num_step, num_heads, dim= x.shape()
    return x[:,:,:,None,:].expand(batch_size, num_step, num_heads, n, dim ).reshape(batch_size, num_step, num_heads*n, dim)

class Attention_GQA(nn.modules):
    def __init__(self, model: Model):
        super().__init__()
        self.num_key_value_heads = model.attention_heads if model.kv_heads is None else model.kv_heads
        self.q_heads = model.attention_heads
        self.n = model.attention_heads // self.num_key_value_heads
        self.head_dim = model.head_dim
        self.is_causal = True
        self.w_q = nn.Linear(model.hidden_size, self.head_dim*self.q_heads, bias= False)
        self.w_k = nn.Linear(model.hidden_size, self.num_key_value_heads*self.head_dim, bias= False)
        self.w_v = nn.Linear(model.hidden_size, self.num_key_value_heads*self.head_dim, bias= False)
        self.w_o = nn.Linear(self.q_heads* self.head_dim, model.hidden_size)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_dropout = nn.Dropout(model.dropout)
        self.reid_dropout = nn.Dropout(model.dropout)
        self.dropout = model.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and model.flash_attn
    def forward(self, x, position_embaddings, use_cache = False, past_kv = None, attention_mask = None):
        bsz, slen,_ = x.shape
        qx,vx,kx = self.w_q(x), self.w_v(x), self.w_k(x)
        qx = qx.view(bsz, slen, self.q_heads, self.head_dim)
        vx = vx.view(bsz, slen, self.num_key_value_heads, self.head_dim)
        kx = kx.view(bsz, slen, self.num_key_value_heads, self.head_dim)
        qx, kx = self.q_norm(qx), self.k_norm(kx)
        sin, cos = position_embaddings
        qx, kx = apply_pos(qx,kx, sin,cos)
        qx, kx, vx = qx.transpose(1,2), repeat_kv(kx).transpose(1,2), repeat_kv(vx).transpose(1,2)
        if past_kv is not None:
            kx = torch.cat([past_kv[0], kx], dim= 1)
            qx = torch.cat([past_kv[1], vx], dim = 1)
        past_kv = (kx, qx) if use_cache else None
        if self.flash and (slen>1) and (not self.is_causal or past_kv is None) and (attention_mask is None or torch.all(attention_mask) == 1):
            output = F.scaled_dot_product_attention(qx, kx, vx, dropout_p= self.dropout, is_causal= self.is_causal)
        else:
            scores = (qx @ kx.transpose(-2,-1))/math.sqrt(self.head_dim)
            if self.is_causal: scores[:,:,:,-slen:] += torch.full((slen,slen), float("-inf"), device= scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2))*-1e9
            output = self.attn_dropout(F.softmax(scores, dim= -1).type_as(vx)@vx)
        output = output.transpose(1,2).reshape(bsz, slen, -1)
        output = self.reid_dropout(self.w_o(output))
        return output, past_kv



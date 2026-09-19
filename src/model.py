import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta)

    def forward(self, x):
        B, H, T, D = x.shape
        seq = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.outer(seq, self.theta)
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rot = x_complex * freqs[None, None, :, :]
        x_out = torch.view_as_real(x_rot).flatten(-2)
        return x_out.type_as(x)

class KVCache:
    def __init__(self):
        self.k = None
        self.v = None
    def update(self, k, v):
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=-2)
            self.v = torch.cat([self.v, v], dim=-2)
        return self.k, self.v

class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.rep = n_heads // n_kv_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x, cache:KVCache=None):
        B,T,C = x.shape
        q = self.wq(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        k = self.wk(x).view(B,T,self.n_kv_heads,self.head_dim).transpose(1,2)
        v = self.wv(x).view(B,T,self.n_kv_heads,self.head_dim).transpose(1,2)
        q = self.rope(q)
        k = self.rope(k)
        if cache is not None:
            k,v = cache.update(k,v)
        k = torch.repeat_interleave(k, self.rep, dim=1)
        v = torch.repeat_interleave(v, self.rep, dim=1)
        attn = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        attn = attn.transpose(1,2).contiguous().view(B,T,C)
        return self.wo(attn)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.ffn = FeedForward(dim, dim*2)
    def forward(self, x, cache):
        x = x + self.attn(self.norm1(x), cache)
        x = x + self.ffn(self.norm2(x))
        return x

class BaizhuLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["dim"])
        self.blocks = nn.ModuleList([
            Block(cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"], cfg["norm_eps"])
            for _ in range(cfg["n_layers"])
        ])
        self.norm_out = nn.LayerNorm(dim, eps=cfg["norm_eps"])
        self.lm_head = nn.Linear(cfg["dim"], cfg["vocab_size"], bias=False)

    def forward(self, tokens, cache_list=None):
        B, T = tokens.shape
        x = self.emb(tokens)
        if cache_list is None:
            cache_list = [KVCache() for _ in range(len(self.blocks))]
        for i, blk in enumerate(self.blocks):
            x = blk(x, cache_list[i])
        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits, cache_list

def load_model(config_path, weight_path=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = BaizhuLLM(cfg)
    if weight_path is not None:
        sd = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(sd)
    return model

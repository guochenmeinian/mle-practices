import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA_KV_Cache(nn.module):

    def __init__(self, d_model, num_heads):

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    
    def forward(self, x, mask=None, past_kv=None):

        """
        x: 
            in training: [batch, seq_len, dim]
            in inference with kv_cache: [batch, 1, dim]
        mask: [batch, num_heads, seq_len, seq_len]
        past_kv: tuple of (past_k, past_v)
        """
        
        batch, seq_len, _ = x.size()
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # [batch, num_heads, seq_len, d_head]
        Q = Q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        if past_kv:
            past_k, past_v = past_kv # [batch, num_heads, past_len, d_head]
            K = torch.cat([past_k, K], dim=2) # dim=2 is seq_len
            V = torch.cat([past_v, V], dim=2)

        updated_kv = (K, V)

        # [batch, num_heads, q_len, d_head] @ [batch, num_heads, d_head, k_len] -> [batch, num_heads, q_len, k_len]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask:
            scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = scores @ V # [batch, num_heads, q_len, d_head]

        output = attn_weights.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(output)

        return output, updated_kv
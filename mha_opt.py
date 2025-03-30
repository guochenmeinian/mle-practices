import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model / num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # [batch_size, num_heads, seq_len, self.d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # [batch, num_heads, seq_len, seq_len]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # [batch, num_heads, seq_len, self.d_k]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights @ V

        # [batch, num_heads, d_model]
        concat = attn_weights.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        output = self.out_linear(concat)
        output = self.layer_norm(x + output)

        return output
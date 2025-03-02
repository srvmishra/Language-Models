import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d = q.shape[-1]
    # attention_matrix = torch.bmm(q, k.transpose(2, 3))/math.sqrt(d)
    attention_matrix = q @ k.transpose(-2, -1)/math.sqrt(d)
    if mask is not None:
        attention_probs = F.softmax(attention_matrix + mask, dim=-1)
    else:
        attention_probs = F.softmax(attention_matrix, dim=-1)
    if dropout is not None:
        attention_probs = dropout(attention_probs)
    return attention_probs @ v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, context_length=2048, embed_dim=512, num_heads=8, mask=False, dropout=0.5):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_len = context_length
        self.head_dim = self.embed_dim//self.num_heads

        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.is_mask = mask
        # self.register_buffer("mask", torch.triu(torch.ones(self.context_len, self.context_len, diagonal=1)))
        
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(batch_size, self.num_heads, seq_len, self.head_dim)
        k = k.view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.view(batch_size, self.num_heads, seq_len, self.head_dim)

        if self.is_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len, diagonal=1)).to(x.device)
            mask_ = mask.masked_fill_(mask.bool(), -torch.inf)
        else:
            mask_ = torch.zeros((seq_len, seq_len)).to(x.device)
        
        xhat = scaled_dot_product_attention(q, k, v, mask=mask_, dropout=self.dropout)
        xhat = xhat.transpose(1, 2)
        x = xhat.contiguous().view(batch_size, seq_len, self.embed_dim)
        x = self.linear(x)
        return x
    
# implement cross attention -- change forward, masking
    
if __name__=="__main__":
    # debug sdpa
    q = torch.rand(size=(10, 512))
    k = torch.rand(size=(10, 512))
    v = torch.rand(size=(10, 512))
    s = scaled_dot_product_attention(q, k, v)
    print(s.shape)

    # debug mhsa
    MHSA = MultiHeadSelfAttention()
    x = torch.rand(size=(10, 128, 512))
    xh = MHSA(x)
    print(xh.shape)

    # debug masked multi head self attention
    MMHSA = MultiHeadSelfAttention(mask=True)
    xh = MHSA(x)
    print(xh.shape)
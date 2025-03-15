import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
  def __init__(self, d_in, d_out):
    super(SelfAttentionV1, self).__init__()
    self.W_query = nn.Parameter(torch.rand(d_in, d_out))
    self.W_key = nn.Parameter(torch.rand(d_in, d_out))
    self.W_value = nn.Parameter(torch.rand(d_in, d_out))

  def forward(self, x):
    '''
    x -> torch.tensor, shape: [batch_size, d_in]
    '''
    query = x @ self.W_query
    key = x @ self.W_key
    value = x @ self.W_value
    attention_scores = query @ key.T
    attention_weights = torch.softmax(attention_scores/value.shape[-1] ** 0.5,
                                      dim=-1)
    context_vector = attention_weights @ value
    return context_vector

  def set_weights(self, W_query, W_key, W_value):
    self.W_query = nn.Parameter(W_query)
    self.W_key = nn.Parameter(W_key)
    self.W_value = nn.Parameter(W_value)

class SelfAttentionV2(nn.Module):
  def __init__(self, d_in, d_out):
    super(SelfAttentionV2, self).__init__()
    self.W_query = nn.Linear(d_in, d_out, bias=False)
    self.W_key = nn.Linear(d_in, d_out, bias=False)
    self.W_value = nn.Linear(d_in, d_out, bias=False)

  def forward(self, x):
    '''
    x -> torch.tensor, shape: [batch_size, d_in]
    '''
    query = self.W_query(x)
    key = self.W_key(x)
    value = self.W_value(x)
    attention_scores = query @ key.T
    attention_weights = torch.softmax(attention_scores/value.shape[-1] ** 0.5,
                                      dim=-1)
    context_vector = attention_weights @ value
    return context_vector
  
class CausalSelfAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, drop_rate, qkv_bias=False):
    super(CausalSelfAttention, self).__init__()
    self.d_out = d_out
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout = nn.Dropout(drop_rate)

    # we dont have to worry about placing tensors separately on device, so we
    # use register_buffer
    self.register_buffer('mask',
                         torch.triu(torch.ones(context_length, context_length), diagonal=1))

  def forward(self, x):
    '''
    x -> torch.tensor, shape: [num_sequences, num_tokens, d_in]
    '''
    query = self.W_query(x)
    key = self.W_key(x)
    value = self.W_value(x)

    attention_scores = query @ key.transpose(-1, -2)

    # in place operation as function ends with _
    # max length is context length, but sequence only has num_tokens tokens
    attention_scores.masked_fill_(self.mask.bool()[:attention_scores.shape[1], :attention_scores.shape[1]], -torch.inf)
    attention_weights = torch.softmax(attention_scores/value.shape[-1] ** 0.5, dim=-1)
    drop_attention_weights = self.dropout(attention_weights)

    context_vector = drop_attention_weights @ value
    return context_vector
  
class MultiHeadCausalAttentionWrapper(nn.Module):
  def __init__(self, d_in, d_out, context_length, drop_rate, num_heads, qkv_bias=False):
    super(MultiHeadCausalAttentionWrapper, self).__init__()
    self.heads = nn.ModuleList([CausalSelfAttention(d_in, d_out, context_length, drop_rate, qkv_bias=False)
                                for _ in range(num_heads)])

  def forward(self, x):
    '''
    Here we can use torch.bmm
    '''
    return torch.cat([h(x) for h in self.heads], dim=-1)
  
class MultiHeadSelfAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, num_heads, drop_rate, qkv_bias=False):
    super(MultiHeadSelfAttention, self).__init__()

    self.in_dim = d_in
    self.out_dim = d_out
    self.num_heads = num_heads
    self.head_dim = self.out_dim//self.num_heads
    self.context_length = context_length

    self.W_query = nn.Linear(self.in_dim, self.out_dim, bias=qkv_bias)
    self.W_key = nn.Linear(self.in_dim, self.out_dim, bias=qkv_bias)
    self.W_value = nn.Linear(self.in_dim, self.out_dim, bias=qkv_bias)
    # combines the outputs from all heads
    self.out_proj = nn.Linear(self.out_dim, self.out_dim)

    self.dropout = nn.Dropout(drop_rate)
    self.register_buffer('mask',
                         torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1))

  def forward(self, x):
    num_seq, num_tokens, _ = x.shape

    query = self.W_query(x)
    key = self.W_key(x)
    value = self.W_value(x)

    query = query.view(num_seq, num_tokens, self.num_heads, self.head_dim)
    key = query.view(num_seq, num_tokens, self.num_heads, self.head_dim)
    value = query.view(num_seq, num_tokens, self.num_heads, self.head_dim)

    attention_scores = query.transpose(1, 2) @ key.transpose(1, 2).transpose(2, 3)   # --> num_seq, num_heads, num_tokens, num_tokens
    attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
    attention_weights = torch.softmax(attention_scores/self.head_dim ** 0.5, dim=-1)
    drop_attention_weights = self.dropout(attention_weights)

    context_vector = drop_attention_weights @ value.transpose(1, 2) # --> num_seq, num_heads, num_tokens, head_dim
    context_vector = context_vector.transpose(1, 2) # --> num_seq, num_tokens, num_heads, head_dim --> transpose is not the same as view/reshape
    # --> create same memory mapping as if created from scratch
    context_vector = context_vector.contiguous().view(num_seq, num_tokens, self.out_dim) # --> num_seq, num_tokens, out_dim
    context_vector = self.out_proj(context_vector) # --> num_seq, num_tokens, out_dim
    return context_vector
  
if __name__=="__main__":
  print('File containing Multi Head Self Attention Implementations!')
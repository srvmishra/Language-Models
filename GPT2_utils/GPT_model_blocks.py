import torch
import torch.nn as nn
import tiktoken

from GPT2_utils.self_attention_mechanisms import MultiHeadSelfAttention

GPT_CONFIGS = {'small': {'vocab_size': 50257,
                         'context_length': 256,
                         'emb_dim': 768,
                         'n_heads': 12,
                         'n_layers': 12,
                         'drop_rate': 0.1,
                         'qkv_bias': False},

               'medium': {'vocab_size': 50257,
                          'context_length': 1024,
                          'emb_dim': 1024,
                          'n_heads': 16,
                          'n_layers': 24,
                          'drop_rate': 0.1,
                          'qkv_bias': False},

               'large': {'vocab_size': 50257,
                         'context_length': 1024,
                         'emb_dim': 1280,
                         'n_heads': 20,
                         'n_layers': 36,
                         'drop_rate': 0.1,
                         'qkv_bias': False},

               'xl': {'vocab_size': 50257,
                      'context_length': 1024,
                      'emb_dim': 1600,
                      'n_heads': 25,
                      'n_layers': 48,
                      'drop_rate': 0.1,
                      'qkv_bias': False}}

class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super(LayerNorm, self).__init__()
    self.eps = 1e-5

    # learnable scale and shift parameters
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    # calculate mean and variance along emb_dim (feature dimension)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # normalize input --> scale and shift it to return output
    normalized_x = (x - mean)/torch.sqrt(var + self.eps)
    return self.scale * normalized_x + self.shift
  
class GELU(nn.Module):
  def __init__(self):
    super(GELU, self).__init__()

  def forward(self, x):
    return 0.5 * x * (1.0 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715*x**3)
        ))
  
class FeedForward(nn.Module):
  def __init__(self, cfg):
    super(FeedForward, self).__init__()

    self.layers = nn.Sequential(nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
                                GELU(),
                                nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']))

  def forward(self, x):
    return self.layers(x)
  
class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super(TransformerBlock, self).__init__()
    self.att = MultiHeadSelfAttention(cfg['emb_dim'], cfg['emb_dim'],
                                      cfg['context_length'], cfg['n_heads'],
                                      cfg['drop_rate'], cfg['qkv_bias'])

    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg['emb_dim'])
    self.norm2 = LayerNorm(cfg['emb_dim'])
    self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

  def forward(self, x):
    x = x + self.drop_shortcut(self.att(self.norm1(x)))
    x = x + self.drop_shortcut(self.ff(self.norm2(x)))
    return x
  
class GPT2Model(nn.Module):
  def __init__(self, cfg):
    super(GPT2Model, self).__init__()

    self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
    self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
    self.drop_emb = nn.Dropout(cfg['drop_rate'])
    self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
    # * unpacks the list as nn.Sequential expects, and use range while iterating with for loop
    self.final_norm = LayerNorm(cfg['emb_dim'])
    self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

  def forward(self, x): # --> batch_size, num_tokens
    tok_emb = self.tok_emb(x)
    # keep 0 inside torch.arange, add device
    pos_emb = self.pos_emb(torch.arange(0, x.shape[1]).to(x.device))#, device=x.device)
    # device throws error: `embedding forward got an unexpected keyword argument device`
    x = tok_emb + pos_emb

    x = self.drop_emb(x)

    x = self.trf_blocks(x)

    x = self.final_norm(x)
    x = self.out_head(x)
    return x
  
def calculate_num_params(gpt_model, exclude_output=False):
  size_mb = lambda x: (4 * x)/(1024 ** 2)
  params = lambda x: sum(p.numel() for p in x)

  # tok_emb = gpt_model.tok_emb.parameters().numel()
  # pos_emb = gpt_model.pos_emb.parameters().numel()
  # trf_blocks = gpt_model.trf_blocks().parameters().numel()
  # out_head = gpt_model.out_head.parameters().numel()

  tok_emb = params(gpt_model.tok_emb.parameters())
  pos_emb = params(gpt_model.pos_emb.parameters())
  trf_blocks = params(gpt_model.trf_blocks.parameters())
  out_head = params(gpt_model.out_head.parameters())

  if exclude_output:
    total = tok_emb + pos_emb + trf_blocks
  else:
    total = tok_emb + pos_emb + trf_blocks + out_head

  print('Trainable Parameters ... ')
  print(f'Token Embedding layer: {tok_emb}, size: {size_mb(tok_emb)} MB')
  print(f'Position Embedding layer: {pos_emb}, size: {size_mb(pos_emb)} MB')
  print(f'Transformer Blocks: {trf_blocks}, size: {size_mb(trf_blocks)} MB')
  print(f'Output Head: {out_head}, size: {size_mb(out_head)} MB')
  print(f'Total params: {total}, size: {size_mb(total)} MB')
  print('###################################################')

  print('Trainable parameters in attention and feed forward blocks ...')
  trf_blk = gpt_model.trf_blocks[0]
  num_trf_blks = len(gpt_model.trf_blocks)
  # attn = trf_blk.att.parameters().numel() * num_trf_blks
  # ffns = trf_blk.ff.parameters().numel() * num_trf_blks
  attn = params(trf_blk.att.parameters()) * num_trf_blks
  ffns = params(trf_blk.ff.parameters()) * num_trf_blks
  print(f'Attention Layer Parameters: {attn}, size: {size_mb(attn)} MB')
  print(f'Feed Forward Network Parameters: {ffns}, size: {size_mb(ffns)} MB')
  print('###################################################')
  return

def generate_text_simple(gpt_model, input_ids, context_length, num_tokens_to_generate=20):
#   input_ids = torch.tensor(tokenizer.encode(input_text))

  if len(input_ids.shape) == 1:
    input_ids = input_ids.unsqueeze(0)
    # print(input_ids.shape)

  for _ in range(num_tokens_to_generate):
    input_ids_trunc = input_ids[:, -context_length:] # --> we want to keep generating, so we keep the latest tokens only,
    # so we do not use [:, :context_length], as it keeps the same old tokens - to the first `context_length` tokens
    with torch.no_grad():
      model_outs = gpt_model(input_ids)[:, -1, :]
    probs = torch.softmax(model_outs, dim=-1)
    next_token_id = torch.argmax(probs, dim=-1, keepdim=True) # --> keepdim is specified so that concatenation will be easy
    input_ids = torch.cat([input_ids, next_token_id], dim=1)

  input_ids = input_ids.cpu().numpy() # can this tokenizer encode & decode a batch of text sequences?
  # no, so we use a loop over all sequences
#   for tid in input_ids:
#     # print(tid.shape)
#     print(tokenizer.decode(tid.tolist()))

  return input_ids

# encode and decode a single text sequence --> use inside a loop to iterate over multiple sequences
def text_to_token_ids(text, tokenizer):
  encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0)
  return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
  token_ids = token_ids.squeeze(0)
  return tokenizer.decode(token_ids.tolist())

if __name__=="__main__":
  print('File containing building blocks of GPT2 Model and Basic Text Generation!')
import re
from importlib.metadata import version

import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken
print('tiktoken version: ', version('tiktoken'))

class SimpleTokenizer:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str = {i:s for s, i in vocab.items()}

  def encode(self, raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids

  def decode(self, ids):
    text = " ".join([self.int_to_str[id] for id in ids])
    text = re.sub(r'\s+([,.:;?"()\'])', r'\1', text)
    return text
  
class GPTDatasetV1(Dataset):
  def __init__(self, text, tokenizer, max_length, stride):
    self.input_ids, self.target_ids = [], []
    self.max_length = max_length
    self.stride = stride

    tokens = tokenizer.encode(text)

    for i in range(0, len(tokens)-self.max_length, self.stride):
      self.input_ids.append(torch.tensor(tokens[i:i+self.max_length]))
      self.target_ids.append(torch.tensor(tokens[i+1:i+self.max_length+1]))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
  
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
  tokenizer = tiktoken.get_encoding('gpt2')
  dataset = GPTDatasetV1(txt, tokenizer, max_length=max_length, stride=stride)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
  return dataloader

if __name__=="__main__":
  print('File containing Data Processing Functions!')
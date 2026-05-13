import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
torch.manual_seed(60261861)
!wget https://raw.githubusercontent.com/karpathy/char-rnn/m
aster/data/tinyshakespeare/input.txt
batch_size = 1
output_dim = 8
context_length = 4
stride = 4
with open('input.txt', 'r', encoding='utf-8') as f:
text = f.read()
3주차
2
class GPTDatasetV1(Dataset):
def __init__(self, txt, tokenizer, max_length, stride):
self.input
_ids = []
self.target_ids = []
token_ids = tokenizer.encode(txt)
for i in range(0, len(token_ids) - max_length, stri
de):
1]
k))
k))
input_chunk = token_ids[i:i + max_length]
target_chunk = token_ids[i + 1:i + max_length + 
self.input_ids.append(torch.tensor(input_chun
self.target_ids.append(torch.tensor(target_chun
def __len__(self):
return len(self.input_ids)
def __getitem__(self, idx):
return self.input_ids[idx], self.target_ids[idx]
def create_dataloader_v1(txt, batch_size, max_length, strid
e):
tokenizer = tiktoken.get_encoding("gpt2")
dataset = GPTDatasetV1(
txt,
tokenizer,
max_length,
stride
)
dataloader = DataLoader(
dataset,
3주차
3
batch_size=batch_size,
shuffle=False
)
return dataloader
dataloader = create_dataloader_v1(
txt=text,
batch_size=batch_size,
max_length=context_length,
stride=stride
)
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
token_embedding_layer = nn.Embedding(vocab_size, output_di
m)
position_embedding_layer = nn.Embedding(context_length, out
put_dim)
for batch in dataloader:
input_batch, target_batch = batch
break
token_embeddings = token_embedding_layer(input_batch)
position_ids = torch.arange(context_length)
position_embeddings = position_embedding_layer(position_id
s)
input_embeddings = token_embeddings + position_embeddings
print("### 첫 번째 배치 입력 임베딩 결과 ###")
print(f"입력 텐서 형태 (Batch, Context, Embedding): {input_em
3주차
4
beddings.shape}")
print("\n첫 번째 배치의 입력 임베딩 값:")
print(input_embeddings)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

torch.manual_seed(60222535)

with open('/Users/jiyun/Downloads/f1e73199-e483-46b9-a4f1-8a73485a309a/deploy/input.txt', 'r', encoding='utf-8') as f:    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text)

batch_size = 1
output_dim = 8
context_length = 4
stride = 4

class GPTDataset(Dataset):
    def __init__(self, token_ids, context_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i:i + context_length]
            target_chunk = token_ids[i + 1:i + context_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

dataset = GPTDataset(token_ids, context_length, stride)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)

vocab_size = 50257

token_embedding_layer = nn.Embedding(vocab_size, output_dim)
position_embedding_layer = nn.Embedding(context_length, output_dim)

data_iter = iter(dataloader)

inputs, targets = next(data_iter)

token_embeddings = token_embedding_layer(inputs)

positions = torch.arange(context_length)

position_embeddings = position_embedding_layer(positions)

input_embeddings = token_embeddings + position_embeddings

print("### 첫 번째 배치 입력 임베딩 결과 ###")
print(f"입력 텐서 형태 (Batch, Context, Embedding): {input_embeddings.shape}")

print("\n첫 번째 배치의 입력 임베딩 값:")
print(input_embeddings)
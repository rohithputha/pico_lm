
import torch.nn as nn
import torch
import torch.nn.functional as F



n_embed = 128
head_size = 16
num_heads = 8
n_layer = 3
vocab_size = 65
block_size = 16
lr = 1e-3 

eval_interval = 100
device = 'mps' if torch.mps.is_available() else 'cpu'
batch_size = 32


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
    

        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        T = block_size
        self.register_buffer('tril',torch.tril(torch.ones(T, T)))
        self.dropout = nn.Dropout(0.2)


    def forward(self, X):
        B, T, C = X.shape
        k = self.key(X) #(B, T, head_size)
        q = self.query(X) #(B, T, headsize)

        wei = q @ k.transpose(-2, -1) #(B, T, T)

        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1) #(B, T, T)
        wei  = self.dropout(wei)
        v = self.value(X) #(B, T, head_size)

        wei = wei @ v  # (B, T, headsize)
        return wei


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([ Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(0.2)

    def forward(self, X):
        heads = torch.cat([h(X) for h in self.heads], dim = -1)
        out = self.dropout(self.projection(heads))
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self. s = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.2)
        )

    def forward(self, X):
        return self.s(X)
    

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.mha = MultiHeadAttention(head_size,num_heads)
        self.ffn = FeedForward()
        self.dropout = nn.Dropout(0.2)


    def forward(self, X):
        X = X + self.mha(X)
        X = X + self.ffn(X)
    
        return X


class picoGpt(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embed)
        self.pos = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential( *[Block() for _ in range(n_layer)])
        self.l = nn.Linear(n_embed, vocab_size)


    def forward(self, X, targets = None):
        B, T = X.shape
        C = n_embed
        #X -> (B, T, vocab_size) and n_embed is C
        tok_embed = self.emb(X) # (B, T, C)
        pos_enc = self.pos(torch.arange(T, device = device))
        tok_embed = tok_embed + pos_enc

        # print(tok_embed.shape)
        t = self.blocks(tok_embed)
        logits = self.l(t)
        loss = None
        if targets == None:
            loss = None

        else:
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits , loss = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx= torch.cat([idx, idx_next], dim = 1)
        return idx
    
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = { ch: i for i, ch in enumerate(chars)}
itos = { i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)
encode = lambda x: [stoi[ch] for ch in x]  
decode = lambda x: ''.join([itos[i] for i in x])
enc = encode ("hi")
decode(enc)

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train = data[:n]
val = data[n:]
def get_batch(split):

    data =  train if split == 'train' else val
    ix = torch.randint(len(data)- block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x,y


model = picoGpt()

model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M params')
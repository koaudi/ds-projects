import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 16 # how many independent sequences of characters we'll process at once
block_size = 84 #what is the maximum context window length
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cpu'
eval_inters = 200
n_embed = 96
n_head = 6
n_layer = 6
dropout = 0.2
torch.manual_seed(42)
# ---------

with open("input.txt", "r", encoding='utf-8') as file:
    text = file.read()

# here are all the unique characters in the text
chars = sorted(list(set(text))) # gets all the unique characters in the text
vocab_size = len(chars)
# create a mapping of character to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x] # encoder: taking a string and outputs a list of integers
decode = lambda x: "".join([itos[i] for i in x]) # decoder: taking a list of integers and outputs a string

# Split data into training and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9) # first 90% of the data as training, rest validation
train_data = data[:n]
val_data = data[n:]

# load the data
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get random start indices
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack the data
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # stack the data
    return x, y

@torch.no_grad() ## telling pytorch that we will not call .backword on to be more efficient
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_inters)
        for k in range(eval_inters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

## Create one head of self-attention
class Head(nn.Module):
    # One head of self-attention
    def __init__(self,head_size):
        super().__init__()
        self.key  = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B,T,C
        q = self.query(x) # B,T,C
        ## compute attentions scores - affinities
        wei = q @ k.transpose(-2,-1) * C**(-0.5) # B,T,T
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # mask out the upper triangular part
        wei = F.softmax(wei, dim=-1) # B,T,T
        wei = self.dropout(wei) # apply dropout
        #perform the weighted sum of the values
        v = self.value(x) # B,T,C
        out = wei @ v # B,T,T @ B,T,C = B,T,C
        return out

class MultiHeadAttention(nn.Module):
    # Multi-head self-attention 
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # B,T,C
        out = self.dropout(self.proj(out)) # apply dropout
        return out

class FeedFoward(nn.Module):
    # a simple linear layer followed by non-linearity
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout), # apply dropout
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    ## Transformer block: communication followed by computation
    def __init__(self, n_embed,n_head):
        #n_embed embeding dimension , n_head number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

## Create simple bigram model
class BigramLanguageModel(nn.Module):    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # (B,T,C)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed,n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) # (B,T,vocab_size)
        #self.sa_head = MultiHeadAttention(4, n_embed//4) # 4 heads of 8 dimensional self attention
        #self.ffwd = FeedFoward(n_embed)
    
    def forward(self, idx, targets=None):
        B,T = idx.shape
        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # B,T,C
        #x = self.sa_head(x) # apply head of self-attention 
        x = self.blocks(x) # apply multiple heads and feed forward layer
        x = self.ln_f(x) # apply final layer norm
        logits = self.lm_head(x) # B,T,C=vocab_size
        
        if targets is None:
            loss = None
        else:
            # we want to evaluate the model on the targets using loss function
            B, T, C = logits.shape
            # reshape logits to be a 2d tensor specific to pytorch
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)
        return logits, loss   
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B,T) array of indexes in the current context window and this function return B, T+1
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]            
            #get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step, making B,T become B,C
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # Size (B,1)
            # append to the context window
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

## Create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    # every one ince a while evlaute the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']}, val loss: {losses['val']}")
    
    #sample a batch of data
    xb, yb = get_batch('train')
    #evaluate the loss of the model
    logits, loss = model(xb, yb) 
    # zero the gradients from previous steps
    optimizer.zero_grad(set_to_none=True)
    # get new gradients for all parameters
    loss.backward() ## backward propogation
    # update the parameters
    optimizer.step()

##generate text form model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context,max_new_tokens=1000)[0].tolist()))

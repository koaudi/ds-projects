import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32 # how many independent sequences of characters we'll process at once
block_size = 8 #what is the maximum context window length
max_iters = 3000
eval_interval = 30
learning_rate = 1e-2
device = 'cpu'
eval_inters = 200
n_embed = 32
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

## Create simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # (B,T,C)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # (B,T,vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # B,T,C
        logits = self.lm_head(x)
        
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
            #get the predictions
            logits, loss = self(idx)
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
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))



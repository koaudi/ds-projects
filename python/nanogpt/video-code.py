import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)

with open("build-gpt/input.txt", "r", encoding='utf-8') as file:
    text = file.read()
len(text)
print(text[:1000])


# here are all the unique characters in the text
chars = sorted(list(set(text))) # gets all the unique characters in the text
vocab_size = len(chars)
print("".join(chars))
print('Vocab size:', vocab_size)

""" Tokenizing each character in the text to build a character level language model
# We create a mapping form integers to strings 
This is a simple tokenizer - google has Sentence Piece (subword units) & openai has tiktoken (byte pair encoding)
"""
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x] # encoder: taking a string and outputs a list of integers
decode = lambda x: "".join([itos[i] for i in x]) # decoder: taking a list of integers and outputs a string

print(encode('hello'))
print(decode(encode('hello')))

# let's now encode the entire text and store it into a torch tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape,data.dtype)
print(data[:100])

# Split data into training and validation sets
n = int(len(data)*0.9) # first 90% of the data as training, rest validation
train_data = data[:n]
val_data = data[n:]

""" 
We are now going to feed the data into a transformer but only in chunks - computationally too expensive
to do all data at once
"""
## Time dimesion
block_size = 8
train_data[:block_size+1]

# example
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context}, target is  {target}")

# Batch dimension (define) - feed many batches into transformer for efficiency 

block_size = 8 # how many independent characters we will process in parallel
batch_size = 4 # maximum context length for the prediction

def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get random start indices
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack the data
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # stack the data
    return x, y

xb, yb = get_batch('train')
print('inputs')
print(xb)
print('targets')
print(yb) # will help us create loss function 

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"batch {b}, when input is {context.tolist()}, target is  {target}")


## simplest languange model is bigram language model
"""
A bigram language model is a type of statistical language model that predicts the probability of a word occurring given the previous word
The model calculates the probability of a word appearing based on the word that immediately precedes it.
This learned probability allows the model to predict the likelihood of a word following another word in a given sequence.
a bigram language model provides a basic but useful way to understand and predict word sequences by analyzing the
 statistical relationships between consecutive words.
"""
class BigramLanguageModel(nn.Module):    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) 
        
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


m = BigramLanguageModel(vocab_size)
logits,loss = m(xb, yb)
print(logits.shape) # should be (batch_size, block_size, vocab_size)
print(loss) ## use negative log likelihood loss to estimate what loss should be ~4,2 ln(-1/65)

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long),max_new_tokens=100)[0].tolist()))

"""
We are now training the model - use Adam to get training data and update the parameters using context and targets
Optimizers are essential for training neural networks in PyTorch. 
They manage the process of updating model parameters based on gradients, enabling the model to learn from data and improve its performance.
"""
#create a  PyTorch Optimier - takes gradients and updated parameters based on training data
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(1000):
    #get a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss of the model
    logits, loss = m(xb, yb) 
    # zero the gradients from previous steps
    optimizer.zero_grad(set_to_none=True)
    # get new gradients for all parameters
    loss.backward()
    # update the parameters
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long),max_new_tokens=100)[0].tolist()))


"""
We can now build self attention mechanism - the core of the transformer model
The fastest way for tokens to hold context take a token and look at the average of all the tokens before it

To make tokens hold infomation from past context you use a weighted average of all the tokens before it
You can do this manually or by using matrix multiplication with softmax regression

"""
B,T,C = 4,8,2 # batch size, time steps, channels
x = torch.randn(B,T,C)
x.shape

# Version 1 of calculating averages - doing it manually
# we want x[b,t] mean_{i<=t} x[b,i] to be the average of all the tokens before it
xbow = torch.zeros((B,T,C)) # bag of words
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # get all tokens before and including t
        xbow[b,t] = torch.mean(xprev,dim=0) # average over the time dimension

#Version 2 using matrix multiplication
## use matrix multiplaction to make calcualting averages more efficient - essntial for weighted sum 
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1,keepdim=True)
xbow2   = wei @ x 
torch.allclose(xbow, xbow2) # should be true if xbos and xbow2 are the same 

#version 3: use softmax
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros(T,T) ## how many tokens from past to consider
wei = wei.masked_fill(tril==0, float('-inf')) # tokes from past cannot communicate with tokens in the future by setting to negative infinit
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3) # should be true if xbos and xbow3 are the same




'''
build a self attention mechanism - the core of the transformer model
every single token will emmit 2 vectors a query and a query key
query = what am I looking for, key = what do i contain  
we do a dot product between the my query and the key of all previous tokens key to get the attention score (affnity score)
Attentions is a communication mechanism between tokens
'''
B,T,C = 4,8,32 # batch size, time steps, channels
x = torch.randn(B,T,C)

# lets see a single Head perform self attention creating the weights from the query and key
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # B, T, 16(head_size)
q = query(x) # B, T, 16(head_size)
wei = q @ k.transpose(-2,-1) # B, T, T transpose the last two dimensions of the key tensor B,T,16 @ B,16,T = B,T,T
 
tril = torch.tril(torch.ones(T,T))
#wei = torch.zeros(T,T) ## how many tokens from past to consider
wei = wei.masked_fill(tril==0, float('-inf')) # tokes from past cannot communicate with tokens in the future by setting to negative infinity decoder block
wei = F.softmax(wei, dim=-1) # use softmax to normalize to get a nice distribution of probablities to 1

v = value(x)
out = wei @ v # B,T,T @ B,T,16 = B,T,16

''' 
MORE NOTES ON ATTENTION
Attention is a communication mechanism can be seen as nodes in a directed graph looking at each other and aggregating info with a weighted sum from 
all the nods that point to them
There is no notion of space, attention acts over all vectors thats why we need to postionally encode tokens
Each batch is independent of each other and the attention mechanism is applied to each batch
encoder (remove wei.masked_fill(tril==0, float('-inf'))) block is when you allow all the nodes to talk to each other, in decoder blocks nodes can only talk to nodes before them
'self attention' means the keys and values are produced from the same source as the queries
'cross attentions' means the keys and values are produced from a different source as the queries
scaled attention additionally divides weights by 1/sqrt(d_k) to prevent the dot product from getting too large
'''
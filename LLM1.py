from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from jaxtyping import Float, Int
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
@dataclass
class Config: #nodes of the network
    d_model: int # this is the internal language of the network
    d_vocab: int # this is the external language of the network
    d_hidden: int # the number of nodes (neurons) in the hidder layer

class Transformer1(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # imagine that we have a input layer that is d_model wide, because each node puts off a different real number element of the embedding vector
        self.embed = nn.Embedding(config.d_vocab, config.d_model) #this transforms the words into vectors, inititally these vectors are completely random values
        # Learnable lookup table: token_id -> R^{d_model}
        # Weight shape: [d_vocab, d_model]
        # Output: [batch, seq, d_model]

        self.attn = Attention_head(config)
        self.mlp = MLP(config)
        self.unembed = nn.Linear(config.d_model, config.d_vocab)
        # Output projection: R^{d_model} -> scores over vocab
        # Weight shape: [d_vocab, d_model] (transposed internally)
        # Output (logits): [batch, seq, d_vocab]

    def forward(self, tokens):
        # tokens: [batch, seq]  ← THIS is the raw input
        x = self.embed(tokens) # [batch, seq, d_model] ← input layer output
        # this creates the token vectors from the input layer to then be fed into the hidden layer(s) (with the mlp)
        x = self.attn(x)
        x = self.mlp(x)
        logits = self.unembed(x)
        #this takes the vector of size d_model from the mlp and multiplies it by a matrix that is d_vocab rows and d_model columns then makes a bias vector of length d_vocab
        # this represents the output layer and is checking which word our now edited token looks most like
        # returns a vector of size d_vocab
        return logits


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_hidden)
        # the line above creates a weight matrix that has d_model columns by d_hidden rows, then also a bias vector that has d_hidden rows
        # the values for both the bias vector and the weight matrix should initially be very random
        # this is because we need to multiply the embedding vector by a row of this weight matrix
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(config.d_hidden, config.d_model)

    def forward(self, x):
        # x shape: [batch, seq, d_model]
        x = self.linear1(x)   # → [batch, seq, d_hidden]
        #this is the hidden layer acting as there are d_hidden number of rows that the weights for each node are multiplied by the input token, then each neuron
        # from the hidden layer adds its own real number bias from the bias vector
        # the result of this call is a vector of real numbers
        x = self.relu(x)
        # this applies the relu function to the output of the hidden layer
        # this returns a vector of real numbers that has d_hidden number of elements
        x = self.linear2(x)   # → [batch, seq, d_model]
        #this multiplies a vector of size d_hidden with a matrix that is d_hidden (cols) by d_model (rows) and adds a bias vector of length d_model
        # the result of this is a vector of real numbers that has d_model number of elements
        return x # return the vector that is d_model long (per token basis)

class Attention_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        # Learned projection matrices
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        # all of these matricies are d_model by d_model and don't have the bias vector, so when self.W_q(stuff) will multiply the matrix of stuff by the square matrix of W_q
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, causal: bool = True):
        """
        x: [batch, seq, d_model]
        returns: [batch, seq, d_model]

        x = [
        [ token_vector, token_vector, ..., token_vector ],   # sequence 0
        [ token_vector, token_vector, ..., token_vector ],   # sequence 1
        ...] down to sequence # batch
        """
        # T = how many tokens are in ONE sequence
        # B = how many sequences at once
        B, T, D = x.shape # this returns a tuple that is then broken apart into the repsective variables B, T, D like matching in lisp (B=batch_size, T=sequence_length, D=d_model)
        assert D == self.d_model
        # this checks that each token's embedding dimension is d_model
        # 1) Project to queries/keys/values
        # x  →  (B·T) × D when being multiplied by the matrix d_model by d_model
        Q = self.W_q(x)
        # (B·T × D)  @  (D × D)  →  (B·T × D)
        # Then PyTorch reshapes the result back to: B × T × D
        K = self.W_k(x)  
        V = self.W_v(x)  

        # 2) Compute attention scores (dot products)
        # scores[b, t, s] = Q[b,t] · K[b,s]
        # transpose(-2, -1) swaps the last two dimensions [B, D, T]
        scores = Q @ K.transpose(-2, -1)  # [B, T, T]
        # The @ operator does batched matrix multiplication.
        """
        this is basically the same as 
        for each batch b:
            scores[b] = Q[b] @ K[b].T
        """
        # each entry in scores is dot(Q[b,t], K[b,s])
        # How much should token t pay attention to token s? is what is being calculated
        scores = scores / math.sqrt(D) # need to normalize the values before applying the softmax function so that it doesn't struggle with potential very large values
        # after this line every token has compared itself to every other token
        # 3) Optional causal mask (prevent looking ahead)
        if causal:
            # mask future positions (upper triangle) with -inf
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            # needed typically because we don't want the llm to look into the future of the sequence as we just are paying attention to tokens previous to the current one to gain context
        # 4) Softmax over "which source position s to attend to"
        attn = torch.softmax(scores, dim=-1)  # [B, T, T]

        # 5) Weighted sum of values
        out = attn @ V  # [B, T, D]

        # 6) Output projection
        out = self.W_o(out)  # [B, T, D]
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        # This creates a layer that normalizes each token vector independently
        # For each token, LayerNorm subtracts the token’s mean and divides by its
        # standard deviation (computed from the variance), then applies a learned
        # scale (gamma) and shift (beta)
        # these scale and shift parameters are learned through back propogation and gradient descent
        self.attn = Attention_head(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        x: [batch, seq, d_model]
        returns: [batch, seq, d_model]
        """
        # Attention sublayer with residual
        x = x + self.attn(self.ln1(x))

        # MLP sublayer with residual
        x = x + self.mlp(self.ln2(x))

        return x
    
class Transformer(nn.Module):
    def __init__(self, config: Config, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(config.d_vocab, config.d_model)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(config.d_model)   # final norm (common)
        self.unembed = nn.Linear(config.d_model, config.d_vocab)

    def forward(self, tokens):
        # tokens: [B, T]
        x = self.embed(tokens)  # [B, T, D]

        for block in self.blocks:
            x = block(x)        # still [B, T, D] also does basically x = block.forward(x) 

        x = self.ln_f(x) # normalize the output
        logits = self.unembed(x)  # [B, T, V] 
        # for each token position, produce a score for every possible vocabulary token
        return logits
    
@torch.no_grad()
def generate_greedy(model, prompt_tokens, max_new_tokens=50):
    """
    model: your Transformer (returns logits [B, T, V])
    prompt_tokens: LongTensor of shape [B, T] (token ids)
    returns: LongTensor of shape [B, T + max_new_tokens]
    """
    model.eval()
    tokens = prompt_tokens

    for _ in range(max_new_tokens):
        logits = model(tokens)              # [B, T, V]
        next_logits = logits[:, -1, :]      # [B, V] (use last position)
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B, 1]
        tokens = torch.cat([tokens, next_token], dim=1)  # append on T dimension

    return tokens

def top_k_filter(logits, k):
    """
    logits: [B, V]
    keeps only top-k logits per batch row; sets the rest to -inf
    """
    if k is None or k <= 0:
        return logits

    V = logits.size(-1)
    k = min(k, V)  # make sure k is not bigger than vocab size

    topk_vals, _ = torch.topk(logits, k, dim=-1)
    cutoff = topk_vals[:, -1].unsqueeze(-1)  # [B, 1]
    return logits.masked_fill(logits < cutoff, float("-inf"))


@torch.no_grad()
def generate_sample(model, prompt_tokens, max_new_tokens=50, temperature=1.0, top_k=None):
    """
    temperature: >0. Lower -> more deterministic, higher -> more random.
    top_k: if set (e.g. 50), sample only from the top_k tokens.
    """
    model.eval()
    tokens = prompt_tokens

    for _ in range(max_new_tokens):
        logits = model(tokens)                 # [B, T, V]
        next_logits = logits[:, -1, :]         # [B, V]

        # temperature scaling
        next_logits = next_logits / max(temperature, 1e-8)

        # optional top-k
        next_logits = top_k_filter(next_logits, top_k)

        probs = F.softmax(next_logits, dim=-1)          # [B, V]
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens
def train(model, dataloader, epochs=5, lr=3e-4, device="cpu"):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in dataloader:
            # x, y: [B, T]
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # [B, T, V]

            # CrossEntropyLoss expects:
            #   input:  [N, C]
            #   target: [N]
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch+1}/{epochs} | loss {avg_loss:.4f}")

class NextTokenDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        """
        token_ids: the entire corpus encoded into token ids
        block_size: length T of the input sequence
        """
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # we need room for x of length block_size and y of length block_size
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]                 # [T]
        y = self.data[idx + 1 : idx + 1 + self.block_size]         # [T] (next tokens)
        return x, y
class CharTokenizer:
    def __init__(self, text_corpus: str):
        chars = sorted(list(set(text_corpus)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
    

prompt = "Once upon a time"
tokenizer = CharTokenizer(prompt)

config = Config(d_model=256, d_vocab=tokenizer.vocab_size, d_hidden=1024)
model = Transformer(config, n_layers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt_ids = tokenizer.encode(prompt)
tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

out = generate_sample(model, tokens, max_new_tokens=60, temperature=0.8, top_k=50)
print(tokenizer.decode(out[0].tolist()))
from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from jaxtyping import Float, Int
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import re
import json
from pathlib import Path
import os


@dataclass
class Config: #nodes of the network
    d_model: int # this is the internal language of the network
    d_hidden: int # the number of nodes (neurons) in the hidder layer
    d_head: int # Each head looks at the same tokens but from a different learned perspective.
    d_vocab: int
    n_heads: int
    num_blocks: int # the number of transformer blocks to use

class Transformer(nn.Module):
    def __init__(self, config: Config, inputstr: str):
        super().__init__()
        self.tokenizer = WordTokenizer(inputstr)
        d_vocab = self.tokenizer.vocab_size

        self.embed = Embedding(config, d_vocab)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_blocks)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, d_vocab)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T] OR [T]

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # make [1, T] so that a single sequence is a size 1 batch

        x = self.embed(token_ids)  # [B, T, D]

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.unembed(x)  # [B, T, V]

        return logits

class Embedding(nn.Module):
    def __init__(self, config: Config, d_vocab: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_vocab, config.d_model)) # creating a empty matrix of d_vocab rows and d_model columns
        nn.init.normal_(self.weight, mean=0.0, std=0.02) # input random garbage values for the matrix
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        return self.weight[x]  # look up token x in the embedding matrix each id (x translated to ids) will return a vector of real numbers
        #thus after this function call x should be a matrix of n_c by d_model

class WordTokenizer:
    def __init__(self, text: str, vocab_path: str = "LLMVocab.json"):
        self.vocab_path = Path(vocab_path)

        self.stoi: dict[str, int] = self._load_vocab()   # {word: id}
        self.itos: dict[int, str] = {i: w for w, i in self.stoi.items()}

        self.add_text(text)
        self.save()

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^'\w\s]", "", text)
        return text

    def add_text(self, text: str) -> None:
        for w in self.normalize(text).split():
            self.add_word(w)

    def add_word(self, word: str) -> int:
        word = self.normalize(word)
        if word in self.stoi:
            return self.stoi[word]
        new_id = len(self.stoi)
        self.stoi[word] = new_id
        self.itos[new_id] = word
        return new_id

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in self.normalize(text).split():
            if w in self.stoi:
                ids.append(self.stoi[w])
            else:
                ids.append(self.add_word(w))  # grows vocab
        return ids

    def decode(self, ids: list[int]) -> str:
        try:
            return " ".join(self.itos[i] for i in ids)
        except KeyError as e:
            raise ValueError(f"Unknown token id in decode: {e.args[0]}") from e

    def _load_vocab(self) -> dict[str, int]:
        if not self.vocab_path.exists():
            return {}
        data = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Vocab file must contain a JSON object {word: id}")
        return {str(word): int(idx) for word, idx in data.items()}

    def save(self) -> None:
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_path.write_text(
            json.dumps(self.stoi, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model) # this is needed as activations can get very messy after a few layers, so normalize them to keep them usable
        # For each token, LayerNorm subtracts the token’s mean and divides by its
        # standard deviation (computed from the variance), then applies a learned
        # scale (gamma) and shift (beta)
        # these scale and shift parameters are learned through back propogation and gradient descent
        self.attn = Attention_head(config)
        self.ln2 = nn.LayerNorm(config.d_model) # to ensure activation values again don't blow up to crazy values
        # and these are two different layernorms because we want to learn what is best for maintaining normality at each layer
        self.mlp = MLP(config)
    def forward(self, x):
        # Attention sublayer with residual
        x = x + self.attn(self.ln1(x))

        # MLP sublayer with residual
        x = x + self.mlp(self.ln2(x))

        return x
    
class Attention_head(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        ''' for single head use this:
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        '''
        #this is multi head 
        self.W_q = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_k = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_v = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_o = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)
        # what is the difference between single head and multi-head?
        # the multi head allows for much more relationship building between tokens, ex grammar and semantic similarity if n_heads is 2
        # why is n_heads*d_head=d_model?
        # this is because we are basically so that we can compare different learned projections of the tokens to each other, to gain more context
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # allow [T, D] -> [1, T, D]

        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # 1) Compute Q,K,V: [B, T, H*Dh]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) Reshape to heads: [B, H, T, Dh]
        Q = Q.view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        K = K.view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        V = V.view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]

        # 3) Attention scores: [B, H, T, T]
        scores = Q @ K.transpose(-2, -1)  # [B, H, T, T]
        scores = scores / math.sqrt(Dh)   # scale by d_head (not d_model)

        # 4) Causal mask: block attending to future tokens
        # mask shape [T, T] broadcasts to [B, H, T, T]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        # 5) Softmax -> attention weights
        attn = torch.softmax(scores, dim=-1)  # [B, H, T, T]

        # 6) Weighted sum of values -> [B, H, T, Dh]
        out = attn @ V  # [B, H, T, Dh]

        # 7) Merge heads back: [B, T, H*Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)

        # 8) Output projection -> [B, T, d_model]
        out = self.W_o(out)
        return out
    
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

@torch.no_grad()
def generate_greedy(model, prompt_tokens, max_new_tokens=50):
    model.eval()
    tokens = prompt_tokens  # [T]

    for _ in range(max_new_tokens):
        logits = model(tokens)          # [T, V]
        next_logits = logits[-1, :]     # [V]
        next_token = torch.argmax(next_logits, dim=-1)  # scalar
        tokens = torch.cat([tokens, next_token.view(1)], dim=0)  # append to [T]
    return tokens

def top_k_filter(logits, k):
    """
    logits: [V]
    keeps only top-k logits per batch row; sets the rest to -inf
    """
    if k is None or k <= 0:
        return logits

    V = logits.size(-1)
    k = min(k, V)  # make sure k is not bigger than vocab size

    topk_vals, _ = torch.topk(logits, k, dim=-1)
    cutoff = topk_vals[-1].unsqueeze(-1)  # [1]
    return logits.masked_fill(logits < cutoff, float("-inf"))


@torch.no_grad()
def generate_sample(model, prompt_tokens, max_new_tokens=50, temperature=1.0, top_k=None):

    model.eval()
    tokens = prompt_tokens

    for _ in range(max_new_tokens):
        logits = model(tokens)                 # [ T, V]
        next_logits = logits[ -1, :]         # [ V]

        # temperature scaling
        next_logits = next_logits / max(temperature, 1e-8)

        # optional top-k
        next_logits = top_k_filter(next_logits, top_k)

        probs = F.softmax(next_logits, dim=-1)          # [ V]
        next_token = torch.multinomial(probs, num_samples=1)  # [ 1]

        tokens = torch.cat([tokens, next_token], dim=0)

    return tokens
def train(model, dataloader, epochs=5, lr=3e-4, device="cpu", grad_clip=1.0):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            # x, y: [B, T]
            x = x.to(device).long()
            y = y.to(device).long()

            logits = model(x)  # should be [B, T, V] or [B*T, V]

            # Ensure logits are [B, T, V]
            if logits.dim() == 2:
                # If your model returns [T, V] for a single sequence,
                # it won't match batching. You should fix model forward.
                raise ValueError(f"Model returned shape {logits.shape}; expected [B, T, V].")

            B, T, V = logits.shape

            # Cross entropy wants [N, C] and [N]
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                y.reshape(B * T),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch+1}/{epochs} | loss {avg_loss:.4f}")

class NextTokenDataset(Dataset):
    def __init__(self, token_ids, seq_len=32):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]                 # [T]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]         # [T]
        return x, y

prompt = "I have no idea what text I am going to put here to train this model, I really don't fully understand what the training is doing yet"

config = Config(
    d_model=256,
    d_hidden=1024,
    d_head=64,
    n_heads=4,
    num_blocks=4,
    d_vocab=0,   # not used (vocab comes from tokenizer)
)

model = Transformer(config, prompt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt_tokens = torch.tensor(model.tokenizer.encode(prompt), dtype=torch.long, device=device)

model.state_dict()

#out1 = generate_sample(model, prompt_tokens, max_new_tokens=60, temperature=0.8, top_k=50)

#out_tokens = generate_greedy(model, prompt_tokens, max_new_tokens=50)

#print(model.tokenizer.decode(out_tokens.tolist()))

#print(model.tokenizer.decode(out1.tolist()))

if os.path.exists("model_weights.pt"):
    model.load_state_dict(torch.load("model_weights.pt", map_location=device))
    print("Loaded existing weights.")
else:
    print("No saved weights found. Training from scratch.")

text = "I have no idea what text I am going to put here to train this model, I really don't fully understand what the training is doing yet"
token_ids = model.tokenizer.encode(text)
seq_len = 32
if len(token_ids) <= seq_len + 1:
    seq_len = max(2, len(token_ids) - 2)
print("using seq_len:", seq_len)

dataset = NextTokenDataset(token_ids, seq_len)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)

train(model, dataloader, epochs=5, lr=3e-4, device=device)

torch.save(model.state_dict(), "model_weights.pt")
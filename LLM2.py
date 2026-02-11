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
        self.tokenizer = WordTokenizer(inputstr) # tokenize the input string
        d_vocab = self.tokenizer.vocab_size
        self.embed = Embedding(config, d_vocab)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_blocks)])
        self.ln_f = nn.LayerNorm(config.d_model)   # final norm (common)
        self.unembed = nn.Linear(config.d_model, config.d_vocab)

    def forward(self, inputstr: str):
        x = self.embed(self.tokenizer.encode(inputstr))  # is now n_c by d_model

        for block in self.blocks:
            x = block(x)        # still [B, T, D] also does basically x = block.forward(x) 

        x = self.ln_f(x) # normalize the output
        logits = self.unembed(x)  # [B, T, V] 
        # for each token position, produce a score for every possible vocabulary token
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
        text = re.sub(r"[^\w\s]", "", text)
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
        ''' for single head use this:
        self.W_q = nn.Linear(config.d_model, config.d_head, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_head, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_head, bias=False)
        self.W_o = nn.Linear(config.d_head, config.d_model, bias=False)
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
    def forward(self, x: torch.tensor):
        T, D = x.shape # shape of n_c by d_m T=n_c and D=d_model
        Q = self.W_q(x)
        # (B·T × D)  @  (D × D)  →  (B·T × D)
        # Then PyTorch reshapes the result back to: B × T × D
        K = self.W_k(x)  
        V = self.W_v(x)  

        # 2) Compute attention scores (dot products)
        # scores[t, s] = Q[b,t] · K[b,s]
        # transpose(-2, -1) swaps the last two dimensions
        scores = Q @ K.transpose(-2, -1) 

        # each entry in scores is dot(Q[b,t], K[b,s])
        # How much should token t pay attention to token s? is what is being calculated
        scores = scores / math.sqrt(D) # need to normalize the values before applying the softmax function so that it doesn't struggle with potential very large values
        # after this line every token has compared itself to every other token

        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

         # 4) Softmax over "which source position s to attend to"
        attn = torch.softmax(scores, dim=-1)  # [B, T, T]

        # 5) Weighted sum of values
        out = attn @ V  # [B, T, D]

        # 6) Output projection
        out = self.W_o(out)  # [B, T, D]
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


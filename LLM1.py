from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from jaxtyping import Float, Int
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
        # this creates the toke vectors from the input layer to then be fed into the hidden layer(s) (with the mlp)
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

        # Optional output projection (common in practice)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, causal: bool = True):
        """
        x: [batch, seq, d_model]
        returns: [batch, seq, d_model]
        """
        B, T, D = x.shape
        assert D == self.d_model

        # 1) Project to queries/keys/values
        Q = self.W_q(x)  # [B, T, D]
        K = self.W_k(x)  # [B, T, D]
        V = self.W_v(x)  # [B, T, D]

        # 2) Compute attention scores (dot products)
        # scores[b, t, s] = Q[b,t] · K[b,s]
        scores = Q @ K.transpose(-2, -1)  # [B, T, T]
        scores = scores / math.sqrt(D)

        # 3) Optional causal mask (prevent looking ahead)
        if causal:
            # mask future positions (upper triangle) with -inf
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

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
            x = block(x)        # still [B, T, D]

        x = self.ln_f(x)
        logits = self.unembed(x)  # [B, T, V]
        return logits
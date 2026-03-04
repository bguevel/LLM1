# ---------------------------------
# 0. dependencies
# ---------------------------------

from dataclasses import dataclass
import unicodedata
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import requests
import numpy as np
from jaxtyping import Float, Int
from collections import Counter

# ---------------------------------
# 1. tokenizer and text processing
# ---------------------------------

def get_frankenstein(
	id: int|None = 84,
	data_temp: Path|str = "../data/gutenberg_data",
	remove_gutenberg_meta: bool = True,
) -> str:
	
	data_temp = Path(data_temp)
	data_temp.mkdir(parents=True, exist_ok=True)
	
	url: str = f"https://www.gutenberg.org/files/84/84-0.txt"
	data_path: Path = Path(data_temp) / f"{id}.txt"
	data: str
	# read from cache if it exists
	if data_path.exists():
		with open(data_path, 'r', encoding='utf-8') as file:
			data = file.read()
	else:
		# download if it doesn't exist
		response = requests.get(url)
		response.raise_for_status()  # Ensure that the download was successful
		data = response.text

		# save to cache
		with open(data_path, 'w', encoding='utf-8') as file:
			file.write(data)

	# remove header/footer
	if remove_gutenberg_meta:
		data = '***'.join(data.split('***')[2:])
		data = '***'.join(data.split('***')[:-1])
	
	return data

def process_text(
        text: str,
        allowed_punctuation: str = "-.,;:!?()\"" + "".join(str(x) for x in range(10)),
        punctuation_convert: dict[str,str] = {'—': '-'},
    ) -> str:
    
        # replace some special characters
        for char, replacement in punctuation_convert.items():
            text = text.replace(char, replacement)

        text = unicodedata.normalize('NFKD', text)
        # Encode to ASCII bytes, then decode back to string, ignoring errors
        text = text.encode('ascii', 'ignore').decode('ascii')
        # remove newlines and tabs
        text = text.replace('\n', ' ').replace('\t', ' ')

        # put spaces around allowed punctuation
        for char in allowed_punctuation:
            text = text.replace(char, f' {char} ')

        # remove leading and trailing spaces
        text = text.strip()

        # remove multiple spaces
        while '  ' in text:
            text = text.replace('  ', ' ')

        # remove all characters except (alphanumeric, allowed_punctuation, ' ')
        text = ''.join(
        (
                char 
                if (
                    char.isalnum() 
                    or char in allowed_punctuation 
                    or char == ' '
                )
                else ' '
            )
            for char in text 
        )

        # convert to lowercase
        text = text.lower()
        text = text.strip()
        return text

def tokenize(
    text: str,
    process: bool = False,
) -> list[str]:
    if process:
        text = process_text(text)
    return text.split(' ')


data = get_frankenstein()

DATA = process_text(data)
DATA_TOKENIZED = tokenize(DATA)
VOCAB_FREQ = Counter(DATA_TOKENIZED)
VOCAB_ARR: list[str] = [word for word, _ in VOCAB_FREQ.most_common()]
VOCAB_DICT: dict[str, int] = {word: i for i, word in enumerate(VOCAB_ARR)}

def encode(
	text: str|list[str],
) -> Int[np.ndarray, "n_tokens"]:
	if isinstance(text, str):
		text = tokenize(text)
	return np.array([VOCAB_DICT[word] for word in text])

def decode(
	encoded_text: list[int],
) -> str:
	return ' '.join(VOCAB_ARR[i] for i in encoded_text)

# test1 = "i believe i can fly"
# encoded_test1 = encode(test1)
# print("Encoded test:", encoded_test1)
# decoded_test1 = decode(encoded_test1)
# print("Decoded test:", decoded_test1)

DATA_ENCODED: Int[np.ndarray, "n_tokens"] = encode(DATA)
# convert from numpy array to torch tensor of type long (int64)
DATA_ENCODED = torch.from_numpy(DATA_ENCODED).long()
DATA_ENCODED = DATA_ENCODED.long()
# print("Encoded data shape:", DATA_ENCODED.shape)

# ---------------------------------
# 2. transformer architecture
# ---------------------------------

@dataclass
class Config: # nodes of the network
    d_model: int # this is the internal language of the network
    d_vocab: int # this is the external language of the network
    d_hidden: int # the number of nodes (neurons) in the hidder layer
    d_head: int # the dimension of attention heads
    n_context: int # window context size
    n_layers: int # number of transformer blocks


class MLP(nn.Module):
    # activation -> something to translate into hidden layer
    # d_model x d_hidden -> activate this using ReLU
    # d_hidden x d_model

    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.d_hidden, config.d_model)

    def forward(self, x):
        # print("MLP input shape:", x.shape)
        x = self.linear1(x)
        # print("After linear1 shape:", x.shape)
        x = self.relu(x)
        # print("After ReLU shape:", x.shape)
        x = self.linear2(x)
        # print("After linear2 shape:", x.shape)

        return x
    
class Attention(nn.Module):
    # need R^{nc x dm} -> R^{nc x dm}
    def __init__(self, config):
        super().__init__()
        self.Wq = nn.Linear(config.d_model, config.d_head, bias=False)
        self.Wk = nn.Linear(config.d_model, config.d_head, bias=False)
        self.Wo = nn.Linear(config.d_head, config.d_model, bias=False)
        self.Wv = nn.Linear(config.d_model, config.d_head, bias=False)

    def forward(self, x):
        # print("Attention input shape:", x.shape)
        Q = self.Wq(x)
        # print("Q shape:", Q.shape)
        K = self.Wk(x)
        # print("K shape:", K.shape)
        W_qk = Q @ K.transpose(-2, -1) # swap last 2 dimensions of K to match dims
        # print("W_qk shape:", W_qk.shape)
        
        sequence_length = x.shape[1]
        M = torch.triu(torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool), diagonal=1)
        # print("Mask shape:", M.shape)
        
        score = W_qk.masked_fill(M, float("-inf"))
        # print("Score shape:", score.shape)
        score = score / math.sqrt(config.d_head) # normalize before softmax so that we don't potentially have very large values
        A = torch.softmax(score, dim=-1)
        # print("A shape:", A.shape)
        
        # x = A * x * W_{ov}
        # x = (A(xW_v))W_o
        # x = (A V) W_o
        # x = W_o(A @ V)
        V = self.Wv(x)
        # print("V shape:", V.shape)
        attention_x = self.Wo(A @ V)
        # print("attention_x shape:", attention_x.shape)

        return attention_x

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)
    def forward(self, x):
        # attention feeds into mlp
        attention_out = self.attention(x)
        x = x + attention_out
        mlp_out = self.mlp(x)
        transformer_x = x + mlp_out
        return transformer_x
    

class Transformer(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(config.d_vocab, config.d_model)
        self.pos_embedding = nn.Embedding(config.n_context, config.d_model)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.unembedding = nn.Linear(config.d_model, config.d_vocab)

    def forward(self, tokens):
        B, T = tokens.shape

        positions = torch.arange(T, device=tokens.device)
        positions = positions.unsqueeze(0).expand(B, T)

        x = self.embedding(tokens) + self.pos_embedding(positions)

        for block in self.blocks:
            x = block(x)

        logits = self.unembedding(x)
        return logits
    
config = Config(
    d_model=64,
    d_vocab=len(VOCAB_ARR),
    d_hidden=128,
    d_head=32,
    n_context=16,
    n_layers=2
)

# ---------------------------------
# 3. generate text
# ---------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(config).to(device)

@torch.no_grad()
def generate_words(
    model,
    prompt: str,
    max_new_tokens: int,
    context_size: int,
    temperature: float,
):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    prompt_ids = encode(prompt)       
    prompt_ids_list = prompt_ids.tolist()  

    tokens = torch.tensor(
        [prompt_ids_list], 
        dtype=torch.long, 
        device=device
    )

    for step in range(max_new_tokens):

        batch_size, total_length = tokens.shape
        if total_length > context_size:
            starting_index = total_length - context_size
        else:
            starting_index = 0
        ending_index = total_length

        context_tokens = tokens[:, starting_index:ending_index]
        logits = model(context_tokens)

        # take logits from last position so we only predict next token
        last_position_logits = logits[:, -1, :] # (batch_size, vocab_size)

        if temperature != 1.0:
            scaled_logits = last_position_logits / temperature
        else:
            scaled_logits = last_position_logits

        probabilities = torch.softmax(scaled_logits, dim=-1)

        # randomly sample one token from distribution
        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1
        )

        tokens = torch.cat(
            [tokens, next_token_id],
            dim=1
        )

    generated_token_ids = tokens[0].tolist()
    generated_text = decode(generated_token_ids)

    return generated_text

input_prompt = "why is the sky blue"
print(
    generate_words(
        model,
        input_prompt,
        max_new_tokens=80,
        context_size=config.n_context,
        temperature=0.9,
    )
)

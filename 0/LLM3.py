from dataclasses import dataclass
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import re
import os
import time
import requests
from typing import Optional
CHECKPOINT_PATH = "checkpoint.pt"


@dataclass
class Config:
    d_model: int
    d_hidden: int
    d_head: int
    d_vocab: int  # not used directly if you rely on tokenizer.vocab_size
    n_heads: int
    num_blocks: int


class WordTokenizer:
    def __init__(self, initial_text: str = "", stoi: dict[str, int] | None = None):
        # If stoi provided (from checkpoint), use it. Otherwise start empty.
        self.stoi: dict[str, int] = dict(stoi) if stoi is not None else {}
        self.itos: dict[int, str] = {i: w for w, i in self.stoi.items()}

        if initial_text:
            self.add_text(initial_text)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def normalize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r"\w+(?:[-']\w+)*|[.,\"()!?-]", text)

    def add_text(self, text: str) -> None:
        for w in self.normalize(text):
            self.add_word(w)

    def add_word(self, word: str) -> int:
        if word in self.stoi:
            return self.stoi[word]
        new_id = len(self.stoi)
        self.stoi[word] = new_id
        self.itos[new_id] = word
        return new_id

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in self.normalize(text):
            if w in self.stoi:
                ids.append(self.stoi[w])
            else:
                ids.append(self.add_word(w))  # grows vocab
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

class Embedding(nn.Module):
    def __init__(self, config: Config, d_vocab: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_vocab, config.d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        return self.weight[x]  # [B,T] -> [B,T,D]


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.d_hidden, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class Attention_head(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.W_q = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_k = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_v = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_o = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.W_q(x)  # [B,T,H*Dh]
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        K = K.view(B, T, H, Dh).transpose(1, 2)
        V = V.view(B, T, H, Dh).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,T,T]

        if causal_mask is None:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V  # [B,H,T,Dh]

        out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)  # [B,T,H*Dh]
        return self.W_o(out)  # [B,T,D]


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention_head(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal_mask=causal_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config, tokenizer: WordTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        d_vocab = tokenizer.vocab_size

        self.embed = Embedding(config, d_vocab)

        # Positional embedding stuff
        self.max_seq_len = 20000  # raise this if you train on longer sequences
        self.pos_emb = nn.Embedding(self.max_seq_len, config.d_model)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_blocks)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, d_vocab)

        # cache causal mask for current T
        self.register_buffer("_causal_mask", None, persistent=False)

        self._init_std = 0.02  # keep consistent init style

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.size(0) != T or self._causal_mask.device != device:
            self._causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return self._causal_mask

    def resize_vocab(self, new_vocab_size: int) -> None:
        old_vocab_size = self.embed.weight.size(0)
        if new_vocab_size <= old_vocab_size:
            return

        device = self.embed.weight.device
        dtype = self.embed.weight.dtype
        d_model = self.embed.weight.size(1)

        #Resize Embedding
        new_embed = nn.Parameter(torch.empty(new_vocab_size, d_model, device=device, dtype=dtype))
        nn.init.normal_(new_embed, mean=0.0, std=self._init_std)
        new_embed.data[:old_vocab_size] = self.embed.weight.data
        self.embed.weight = new_embed

        #Resize Unembedding
        old_unembed: nn.Linear = self.unembed
        new_unembed = nn.Linear(d_model, new_vocab_size, bias=True).to(device=device, dtype=dtype)

        nn.init.normal_(new_unembed.weight, mean=0.0, std=self._init_std)
        nn.init.zeros_(new_unembed.bias)

        new_unembed.weight.data[:old_vocab_size] = old_unembed.weight.data
        new_unembed.bias.data[:old_vocab_size] = old_unembed.bias.data

        self.unembed = new_unembed

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        B, T = token_ids.shape

        # Positional embeddings 
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len or use RoPE/ALiBi."
            )

        tok = self.embed(token_ids)                          # [B,T,D]
        pos_ids = torch.arange(T, device=token_ids.device)   # [T]
        pos = self.pos_emb(pos_ids).unsqueeze(0)             # [1,T,D]
        x = tok + pos                                        # [B,T,D]

        mask = self._get_causal_mask(T, x.device)

        for block in self.blocks:
            x = block(x, causal_mask=mask)

        x = self.ln_f(x)
        return self.unembed(x)  # [B,T,V]


class NextTokenDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        n = len(self.token_ids) - self.seq_len
        return max(0, n)

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1: idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def top_k_filter(logits: torch.Tensor, k: Optional[int]):
    if k is None or k <= 0:
        return logits
    V = logits.size(-1)
    k = min(k, V)
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    cutoff = topk_vals[-1].unsqueeze(-1)
    return logits.masked_fill(logits < cutoff, float("-inf"))


@torch.no_grad()
def generate_sample(model: Transformer, prompt_tokens: torch.Tensor, max_new_tokens=50, temperature=1.0, top_k=None):
    model.eval()
    tokens = prompt_tokens

    for _ in range(max_new_tokens):
        logits = model(tokens)          # [1, T, V] (because tokens is [T] -> unsqueeze inside model)
        next_logits = logits[0, -1]     # [V]

        next_logits = next_logits / max(float(temperature), 1e-8)
        next_logits = top_k_filter(next_logits, top_k)
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1]
        tokens = torch.cat([tokens, next_token], dim=0)

    return tokens

def train(
    epochs: int,
    lr: float,
    device: str,
    grad_clip,
    prompt: str,
    config: Config,
    model: Transformer,
    batch_size: int = 16,
    save_path: str = "checkpoint.pt",
):
    model.to(device)

    # 1) Encode
    token_ids = model.tokenizer.encode(prompt)
    model.resize_vocab(model.tokenizer.vocab_size)

    # Resize model to match new vocab size
    model.resize_vocab(model.tokenizer.vocab_size)

    # Build dataset
    seq_len = len(token_ids) - 1
    if seq_len <= 0:
        print("[warn] prompt too short to train on.")
        return

    dataset = NextTokenDataset(token_ids, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Optimizer AFTER resize so it sees current parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x = x.to(device).long()
            y = y.to(device).long()

            logits = model(x)  # [B,T,V]
            if logits.dim() != 3:
                raise ValueError(f"Model returned shape {logits.shape}; expected [B, T, V].")

            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"epoch {epoch+1}/{epochs} | loss {avg_loss:.4f}")

    save_checkpoint(model, config, "checkpoint.pt")


def load_weights_if_present(model: Transformer, device: str, path: str = CHECKPOINT_PATH) -> bool:
    loaded = load_checkpoint(path=path, device=device)
    if loaded is None:
        print("No checkpoint found. Training from scratch.")
        return False

    loaded_model, loaded_config = loaded

    # make/load tokenizer first
    model.tokenizer = loaded_model.tokenizer
    model.resize_vocab(model.tokenizer.vocab_size)

    # now load weights
    model.load_state_dict(loaded_model.state_dict(), strict=True)

    print("Loaded existing checkpoint.")
    return True
    
def GenerateResponse(prompt: str, config: Config, new_tokens: int, temp: float, top_k: int, model: Transformer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_checkpoint(path=CHECKPOINT_PATH, device=device)
    if loaded is not None:
        loaded_model, loaded_config = loaded
        model.load_state_dict(loaded_model.state_dict())
        model.tokenizer = loaded_model.tokenizer
        print("Loaded existing checkpoint.")
    else:
        print("No checkpoint found. Using current model weights.")

    model.to(device).eval()

    ids = model.tokenizer.encode(prompt)
    model.resize_vocab(model.tokenizer.vocab_size)
    prompt_tokens = torch.tensor(ids, dtype=torch.long, device=device)

    out = generate_sample(model, prompt_tokens, new_tokens, temp, top_k)
    print(model.tokenizer.decode(out.tolist()))

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
HEADERS = {"User-Agent": "MyScraper/1.0 (contact: you@example.com)"}


def wiki_summary(title: str, timeout: int = 20) -> dict:
    """Fetch Wikipedia REST summary JSON for a page title."""
    url = WIKI_SUMMARY.format(requests.utils.quote(title))
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def read_titles_from_file(path: str) -> list[str]:
    """
    Reads titles from a file.
    Assumes one title per line.
    Ignores blank lines and lines starting with #.
    """
    titles: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            titles.append(t)
    return titles


def train_from_wiki_titles_file( titles_file: str, model: Transformer, config: Config, device: str = "cpu", epochs_per_page: int = 100, lr: float = 5e-4, grad_clip: float | None = 1.0, sleep_s: float = 0.5, skip_disambiguation: bool = True, min_chars: int = 50,
):
    titles = read_titles_from_file(titles_file)
    print(f"Found {len(titles)} titles in {titles_file}")

    for i, title in enumerate(titles, start=1):
        print(f"\n[{i}/{len(titles)}] {title}")

        try:
            data = wiki_summary(title)
        except Exception as e:
            print(f"  skip (fetch error): {e}")
            continue

        page_type = data.get("type")  # "standard" / "disambiguation" / etc.
        if skip_disambiguation and page_type == "disambiguation":
            print("  skip (disambiguation page)")
            continue

        extract = (data.get("extract") or "").strip()
        if len(extract) < min_chars:
            print("  skip (summary too short / empty)")
            continue

        # Add light structure to help next-token training
        prompt = f"Title: {data.get('title', title)}.\nSummary: {extract}\n"

        train(
            epochs=epochs_per_page,
            lr=lr,
            device=device,
            grad_clip=grad_clip,
            prompt=prompt,
            config=config,
            model=model,
        )

        #to ease serve usage
        if sleep_s and sleep_s > 0:
            time.sleep(sleep_s)

    return model


def train_on_plain_text_file(path: str) -> str:
    """Read a text file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def training_menu(model: Transformer, config: Config, device: str):
    """
    Option 1 submenu: (a) Wikipedia titles file, (b) regular text file
    """
    while True:
        print("\n--- Training mode ---")
        print("a) Train on Wikipedia summaries (titles from file)")
        print("b) Train on a plain text file")
        print("x) Back to main menu")

        choice = input("select> ").strip().lower()

        if choice == "x":
            break

        if choice == "a":
            titles_file = input("Enter wikipedia titles filename (one title per line): ").strip()
            if not titles_file:
                print("[warn] No filename entered.")
                continue
            if not os.path.exists(titles_file):
                print(f"[warn] File not found: {titles_file}")
                continue

            #settings for the training of the model
            epochs_per_page = int(input("epochs per page (e.g. 1-3)> ").strip() or "1")
            lr = float(input("learning rate (e.g. 0.0005)> ").strip() or "0.0005")
            grad_clip = float(input("grad clip (e.g. 1.0, blank for none)> ").strip() or "1.0")
            sleep_s = float(input("sleep seconds between requests (e.g. 0.5)> ").strip() or "0.5")

            #do the training on the wiki articles
            train_from_wiki_titles_file(
                titles_file=titles_file,
                model=model,
                config=config,
                device=device,
                epochs_per_page=epochs_per_page,
                lr=lr,
                grad_clip=grad_clip,
                sleep_s=sleep_s,
            )

            print("[ok] Finished training on Wikipedia titles.")
            continue

        if choice == "b":
            text_file = input("Enter plain text filename to train on: ").strip()
            if not text_file:
                print("[warn] No filename entered.")
                continue
            if not os.path.exists(text_file):
                print(f"[warn] File not found: {text_file}")
                continue

            prompt_text = train_on_plain_text_file(text_file)
            if not prompt_text.strip():
                print("[warn] File was empty.")
                continue

            epochs = int(input("epochs (e.g. 10)> ").strip() or "10")
            lr = float(input("learning rate (ex. 0.0005)> ").strip() or "0.0005")
            gc_in = input("grad clip (e.g. 1.0, blank for none)> ").strip()
            grad_clip = None if gc_in == "" else float(gc_in)

            # Call train
            train(
                epochs=epochs,
                lr=lr,
                device=device,
                grad_clip=grad_clip,
                prompt=prompt_text,
                config=config,
                model=model,
            )

            print("[ok] Finished training on plain text.")
            continue

        print("[warn] Invalid choice. Please choose a, b, or x.")

MODEL_PATH = CHECKPOINT_PATH


@torch.no_grad()
def interactive_prompt_loop(
    model: Transformer,
    config: Config,
    device: str,
    temp: float = 0.7,
    top_k: int = 50,
    new_tokens: int = 50,
):
    """
    Continually prompt the user, generate a response.
    Type: /exit to leave, /settings to change params.
    """
    model.to(device)

    loaded = load_weights_if_present(model, device, MODEL_PATH)
    if loaded:
        print(f"[ok] Loaded weights from {MODEL_PATH}")
    else:
        print(f"[warn] No weights found at {MODEL_PATH}. You are sampling from random weights.")

    model.eval()

    print("\n--- Prompting mode ---")
    print("Type /exit to quit prompting mode.")
    print("Type /settings to change generation parameters.\n")

    while True:
        user_prompt = input("prompt> ").strip()
        if not user_prompt:
            continue

        if user_prompt.lower() in ("/exit", "exit", "quit", "q"):
            print("Leaving prompting mode.\n")
            break

        if user_prompt.lower() == "/settings":
            try:
                new_tokens = int(input(f"max_new_tokens (current {new_tokens})> ").strip() or new_tokens)
                temp = float(input(f"temperature (current {temp})> ").strip() or temp)
                top_k_in = input(f"top_k (current {top_k}, blank for none)> ").strip()
                top_k = None if top_k_in == "" else int(top_k_in)
            except ValueError:
                print("[warn] Invalid setting value(s). Keeping previous settings.")
            continue

        # --- IMPORTANT: encode may GROW vocab, so resize model before using token ids ---
        ids = model.tokenizer.encode(user_prompt)
        model.resize_vocab(model.tokenizer.vocab_size)

        prompt_tokens = torch.tensor(ids, dtype=torch.long, device=device)

        out = generate_sample(
            model,
            prompt_tokens,
            max_new_tokens=new_tokens,
            temperature=temp,
            top_k=top_k,
        )

        # Only decode newly generated tokens
        generated_ids = out[len(ids):]   # slice off prompt
        print(model.tokenizer.decode(generated_ids.tolist()))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = load_checkpoint(device=device)
    if loaded is not None:
        model, config = loaded
        print("[ok] Loaded checkpoint.pt")
    else:
        print("[warn] No checkpoint found. Starting fresh.")
        config = Config(
            d_model=256,
            d_hidden=1024,
            d_head=64,
            n_heads=4,
            num_blocks=14,
            d_vocab=0,
        )
        tokenizer = WordTokenizer("seed")
        model = Transformer(config, tokenizer).to(device)
        

    while True:
        print("\n=== Main Menu ===")
        print("1) Training mode")
        print("2) Prompting mode")
        print("q) Quit")

        choice = input("select> ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice == "1":
            training_menu(model, config, device)
        elif choice == "2":
            interactive_prompt_loop(model, config, device)
        else:
            print("[warn] invalid selection")

def save_checkpoint(model: Transformer, config: Config, path: str = CHECKPOINT_PATH):
    ckpt = {
        "config": {
            "d_model": config.d_model,
            "d_hidden": config.d_hidden,
            "d_head": config.d_head,
            "d_vocab": 0,
            "n_heads": config.n_heads,
            "num_blocks": config.num_blocks,
        },
        "stoi": model.tokenizer.stoi,
        "model_state": model.state_dict(),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str = CHECKPOINT_PATH, device: str = "cpu") -> tuple[Transformer, Config] | None:
    if not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location=device)

    # Rebuild config
    cfg_dict = ckpt["config"]
    config = Config(**cfg_dict)

    # Rebuild tokenizer from saved vocab
    tokenizer = WordTokenizer(stoi=ckpt["stoi"])

    # Rebuild model with correct initial vocab size
    model = Transformer(config, tokenizer).to(device)

    # If vocab grew since model init, ensure match
    model.resize_vocab(model.tokenizer.vocab_size)

    # Load weights
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, config

if __name__ == "__main__":
    main()

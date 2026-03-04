# Transformers From Scratch — Autoregressive LLMs

This repository contains **two final “transformers-from-scratch” projects** (two model tracks) implemented in PyTorch:

- **Track A (Ben + Cole): `0/LLM3.py`**
  - Multi-head autoregressive Transformer
  - Trained on **Wikipedia page summaries**
  - Word-level tokenizer with a **dynamic (growing) vocabulary**
  - Interactive CLI for training and generation
  - Checkpoint support via `0/checkpoint.pt`

- **Track B (Luca): `0/luca-gpt1.py` and `0/luca-gpt1-untrained.py`**
  - Single-head autoregressive Transformer
  - Trained on **Project Gutenberg — _Frankenstein_**
  - Includes a trained vs. untrained comparison

> **Important:** All `.py` files plus `wiki` and `checkpoint.pt` live inside the `0/` folder.

---

## Setup (uv)

From the **repo root** (the directory that contains `pyproject.toml`):

```bash
uv sync
```

Then run programs with:

```bash
uv run python 0/<file>.py
```

### If `uv sync` fails
If `uv sync` fails due to a lockfile/environment issue, it may be related to the existing `uv.lock`.  
A common fix is to delete `uv.lock` and rerun:

```bash
rm uv.lock
uv sync
```

*(Only do this if `uv sync` is failing—otherwise keep the lockfile for reproducibility.)*

---

## Running the models

## Track A — LLM3 (Ben + Cole)

Run the interactive program:

```bash
uv run python 0/LLM3.py
```

Follow the terminal prompts. The CLI supports:
- training (Wikipedia titles file like `0/wiki`, or local text)
- prompting / generation (sampling settings adjustable in the menu)
- saving/loading checkpoints (default: `0/checkpoint.pt`)

### Wikipedia training data
For Wikipedia training, the model reads titles from:

- `0/wiki` (one Wikipedia page title per line)

and fetches summaries via Wikipedia’s REST endpoint:
`https://en.wikipedia.org/api/rest_v1/page/summary/{title}`

---

## Track B — luca-gpt1 (Luca)

Run the **trained** single-head model:

```bash
uv run python 0/luca-gpt1.py
```

Run the **untrained** single-head model (same architecture, random weights):

```bash
uv run python 0/luca-gpt1-untrained.py
```

These script-style files print a short generation from a default prompt near the bottom of the file.

---

## Design choices (LLM3)

### Multi-head attention
We chose **multi-head self-attention** (rather than single-head) for the final `LLM3` model to better capture different relationships in parallel.

### Word-level tokenizer + dynamic vocabulary
We use a **word-level tokenizer** and a vocabulary that **expands as new words are encountered**. We intentionally avoided a built-in vocabulary/tokenizer library to keep the project “from scratch” and allow the model to learn new words on the fly.

**Key challenge:** the embedding/unembedding layers depend on the vocabulary size.  
As the vocab grows, we must resize these layers while preserving the learned weights for existing tokens. A significant portion of debugging and development time went into making the training loop and embedding resizing stable.

---

## Example outputs (trained vs. untrained)

### Trained output (LLM3)
**Prompt:** `Hello there how are you`

```
a . is is used a is a and transformer computer of how the . the behavior achieving more structured general signal and .
questions that topics physical of units is the . participated mechanics the and of functional . of supports , intelligence ,
of pursuit topics fundamental summary
```

### Untrained output (LLM3)
**Prompt:** `Hello there how are you`

```
you seed seed there seed you you seed seed there how there seed how are hello are seed there there seed seed seed seed you
seed you how hello how hello there seed seed seed seed seed you seed there there hello there seed seed how are you how are
```

---

## Contributions

These projects were split across two tracks:

- **Luca Burns** — led the single-head track (`0/luca-gpt1.py`, `0/luca-gpt1-untrained.py`) and the Frankenstein pipeline
- **Ben Guevel** — led the multi-head `LLM3` architecture and training integration
- **Cole McGuire** — helped significantly with:
  - debugging training + embedding/vocab resizing issues
  - working through the math and pair-programming the fixes into the code

---

## Future work

If we extended this further, we would:
- move from word-level tokenization to **byte-pair encoding (BPE)** or another subword tokenizer
- standardize logging/plots across both tracks
- speed up generation (e.g., KV caching)
- tune hyperparameters more systematically

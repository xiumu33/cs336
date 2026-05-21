# CS336 Assignment 1 — Transformer Language Model from Scratch

This project implements a GPT-style Transformer language model from scratch using PyTorch, including BPE tokenization, training, and inference. All core components (linear layers, attention, normalization, optimizer) are hand-written without relying on PyTorch's built-in transformer modules.

---

## Module Overview

### 1. `nn.py` — Neural Network Building Blocks

The core neural network components that form the Transformer architecture.

| Class / Function | Description |
|---|---|
| `Linear(in_features, out_features)` | A fully-connected linear layer. Weights are initialized with truncated normal distribution (std = sqrt(2 / (d_in + d_out)), truncated at [-3σ, 3σ]). Forward pass uses `einsum('...i, oi -> ...o', x, self.weight)`. |
| `Embedding(num_embeddings, embedding_dim)` | Token embedding layer. Weight initialized with truncated normal (std=1.0). Forward pass performs direct index lookup: `self.weight[token_ids]`. |
| `RMSNorm(d_model, eps=1e-5)` | Root Mean Square Layer Normalization (used in Llama). Computes `x / sqrt(mean(x²) + ε) * g`, where `g` is a learnable scale parameter initialized to ones. |
| `silu_fn(in_features)` | SiLU (Swish) activation: `x * sigmoid(x)`. |
| `SwiGLU(d_model, d_ff)` | SwiGLU feed-forward network. Contains W1, W3 (up-projection) and W2 (down-projection). Computes `W2(silu(W1(x)) * W3(x))`. |
| `RotaryPositionalEmbedding(theta, d_k, max_seq_len)` | Rotary Positional Embedding (RoPE). Precomputes cos/sin frequency tables for all positions up to `max_seq_len`. Forward applies rotation to even/odd dimension pairs of query/key tensors. |
| `softmax(x, dim=-1)` | Numerically stable softmax: subtracts max before exponentiation to prevent overflow. |
| `scaled_dot_product_attention(Q, K, V, mask=None)` | Core attention mechanism. Computes `softmax(Q·K^T / √d_k + mask) · V`. Supports boolean masks (False = masked to -∞). |
| `CausalSelfAttention(d_model, num_heads, ...)` | Multi-head causal self-attention. Projects input to Q/K/V via `Linear`, splits into heads, optionally applies RoPE, constructs a lower-triangular causal mask, computes SDPA, merges heads, and projects output. |
| `TransformerBlock(d_model, num_heads, d_ff, ...)` | A single Transformer block with residual connections. Supports: pre-norm / post-norm, RMSNorm on/off, SwiGLU / SiLU FFN. |
| `TransformerLM(vocab_size, max_seq_len, ...)` | Full Transformer language model. Stacks token embeddings, N `TransformerBlock` layers, final RMSNorm, and a LM head (Linear → vocab_size). |
| `TransformerLM.generate(prompt_ids, ...)` | Autoregressive text generation with temperature scaling and top-p (nucleus) sampling. Stops early on EOS token. |

---

### 2. `losses.py` — Loss Functions

| Function | Description |
|---|---|
| `cross_entropy(logits, targets)` | Numerically stable cross-entropy loss. Computes `log(∑exp(o - M)) + M - o_y` for each position, then returns the mean across all positions. |

---

### 3. `optimizer.py` — Optimizer & Gradient Utilities

| Class / Function | Description |
|---|---|
| `AdamW(params, lr, betas, eps, weight_decay)` | AdamW optimizer implementation from scratch. Maintains first/second moment estimates (exp_avg, exp_avg_sq), applies bias correction, and performs **decoupled weight decay** (`θ = θ - α·λ·θ`) separately from the adaptive gradient update. |
| `clip_gradient_norm(parameters, max_norm)` | Global L2 gradient norm clipping. Computes `total_norm = sqrt(∑‖g_i‖²)` across all parameters, then scales all gradients by `max_norm / total_norm` if the threshold is exceeded. |

---

### 4. `scheduler.py` — Learning Rate Scheduler

| Function | Description |
|---|---|
| `get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)` | Cosine annealing with linear warmup. Phase 1 (t < warmup_iters): linear increase from 0 to max_lr. Phase 2 (warmup_iters ≤ t ≤ cosine_cycle_iters): cosine decay from max_lr to min_lr. Phase 3 (t > cosine_cycle_iters): constant min_lr. |

---

### 5. `tokenizer.py` — BPE Tokenizer

| Class / Method | Description |
|---|---|
| `BPETokenizer(vocab, merges, special_tokens)` | Byte-level BPE tokenizer. Maintains bidirectional ID↔bytes mapping, a merge-rank dictionary for greedy BPE encoding, and a regex for matching special tokens (longest-match priority). |
| `BPETokenizer.encode(text)` | Encodes a string to token IDs. Splits text on special tokens (matched via regex), processes plain-text segments with BPE merging, and directly maps special tokens to their IDs. |
| `BPETokenizer._encode_text_segment(text)` | Core BPE encoding for plain text: (1) GPT-2 regex pre-tokenization into word chunks, (2) convert each chunk to byte sequence, (3) greedily merge adjacent byte pairs by their rank until no merges remain, (4) map final byte chunks to token IDs. |
| `BPETokenizer.decode(ids)` | Decodes a list of token IDs back to a UTF-8 string. Concatenates byte sequences and decodes with `errors="replace"` for safety. |
| `BPETokenizer.encode_iterable(iterable)` | Memory-efficient streaming encoder. Accepts an iterable of text chunks and yields token IDs one at a time via generator — suitable for processing very large files. |

---

### 6. `data.py` — Data Loading

| Function | Description |
|---|---|
| `get_batch(dataset, batch_size, context_length, device)` | Randomly samples `batch_size` consecutive token sequences of length `context_length` from the dataset (a 1D numpy array of token IDs). Returns input `x` and target `y` (offset by 1 position). |

---

### 7. `checkpointing.py` — Model Persistence

| Function | Description |
|---|---|
| `save_checkpoint(model, optimizer, iteration, out)` | Saves a checkpoint dict containing `model_state_dict`, `optimizer_state_dict`, and `iteration` to disk (path or file-like object). |
| `load_checkpoint(src, model, optimizer)` | Loads a checkpoint from disk, restores model weights and optimizer state into the passed objects, and returns the saved iteration number. Uses `map_location='cpu'` for portability. |

---

### 8. `train_bpe.py` — BPE Training

| Function | Description |
|---|---|
| `train_bpe(input_path, vocab_size, special_tokens)` | Trains a byte-level BPE tokenizer from scratch on the given corpus. Steps: (1) initialize vocab with bytes 0–255, (2) split corpus on special tokens, (3) GPT-2 regex pre-tokenization + word frequency counting, (4) iteratively merge the most frequent byte pair (ties broken by lexicographic order), using an inverted index for efficient frequency updates, (5) add merged tokens and special tokens to the final vocab. |
| `bytes_to_unicode()` | Returns a mapping from bytes (0–255) to visible Unicode characters (GPT-2 standard). Non-printable bytes are mapped to a private Unicode range. |
| `save_tokenizer_files(vocab, merges, out_dir)` | Saves `vocab.json` and `merges.txt` to disk, converting bytes to visible Unicode strings via `bytes_to_unicode()`. |

---

### 9. `preprocess.py` — Corpus Preprocessing

| Function | Description |
|---|---|
| `bytes_to_unicode()` | Same as in `train_bpe.py`: byte-to-visible-Unicode mapping. |
| `load_trained_tokenizer(vocab_path, merges_path, special_tokens)` | Loads a trained BPE tokenizer from `vocab.json` and `merges.txt`, reversing the Unicode mapping back to raw bytes. |
| `process_corpus(input_txt, output_bin, tokenizer)` | Streams a large text corpus through the tokenizer using `encode_iterable`, writing tokens as `uint16` binary to disk in batches of 1M tokens for memory efficiency. |

---

### 10. `inference.py` — Interactive Generation

| Function | Description |
|---|---|
| `bytes_to_unicode()` | Byte-to-Unicode mapping for tokenizer loading. |
| `load_trained_tokenizer(vocab_path, merges_path, special_tokens)` | Loads tokenizer from disk files. |
| `main()` | CLI script for interactive text generation. Loads a trained checkpoint, initializes the model, and runs a REPL loop: user types a prompt → tokenize → generate (with temperature & top-p) → decode → print. Type `q`/`exit` to quit. Supports `--temperature`, `--top_p`, `--max_new_tokens` arguments. |

---

### 11. `sgd.py` — SGD with Decay (Learning Rate Tuning Experiment)

| Class / Function | Description |
|---|---|
| `SGD(params, lr)` | Vanilla SGD with step-dependent decay: `θ = θ - (lr / √(t+1)) · g`. Maintains per-parameter step counter `t`. |
| `run_experiment(learning_rate)` | Runs a 10-iteration experiment minimizing `mean(W²)` to diagnose learning rate quality. LR=10 converges steadily; LR=100 converges very fast; LR=1000 diverges (loss explodes). |

---

### 12. `main_train.py` — Training Entry Point

The main training script that wires everything together. Supports extensive CLI arguments for all hyperparameters and ablation studies:

- **Model**: `--d_model`, `--num_layers`, `--num_heads`, `--d_ff`, `--vocab_size`, `--context_length`
- **Ablations**: `--no_rms_norm` (disable RMSNorm), `--norm_mode` (pre/post), `--no_rope` (disable RoPE), `--ffn_type` (swiglu/silu)
- **Optimization**: `--lr`, `--min_lr`, `--warmup_iters`, `--max_iters`, `--max_norm`
- **Data**: `--train_data_path`, `--valid_data_path` (binary `uint16` files)
- **Logging**: `--wandb_project`, `--run_name`

Training loop: cosine LR schedule → random batch sampling → forward pass → cross-entropy loss → backward → gradient clipping → AdamW step. Logs train/val loss every 100 iters, saves checkpoint every 1000 iters.

---

## Project Structure

```
assignment1/
├── cs336_basics/
│   ├── nn.py              # Transformer neural network modules
│   ├── losses.py           # Cross-entropy loss
│   ├── optimizer.py        # AdamW + gradient clipping
│   ├── scheduler.py        # Cosine LR schedule with warmup
│   ├── tokenizer.py        # BPE tokenizer (encode/decode)
│   ├── train_bpe.py        # BPE training on corpus
│   ├── preprocess.py       # Corpus → binary token preprocessing
│   ├── data.py             # Batch sampling from binary data
│   ├── checkpointing.py    # Model save/load
│   ├── sgd.py              # SGD optimizer (LR tuning experiment)
│   ├── inference.py        # Interactive text generation script
│   ├── main_train.py       # Main training entry point
│   └── learn.ipynb         # Jupyter notebook
├── tests/                  # Test suite
└── cs336_assignment1_basics.pdf  # Assignment description
```

## Usage

```bash
# Train BPE tokenizer
python -m cs336_basics.train_bpe

# Preprocess corpus to binary
python -m cs336_basics.preprocess

# Train the model
python -m cs336_basics.main_train \
    --train_data_path data/train.bin \
    --valid_data_path data/valid.bin \
    --d_model 512 --num_layers 4 --num_heads 8 \
    --max_iters 10000

# Interactive generation
python -m cs336_basics.inference \
    --checkpoint_path out/ckpt.pt \
    --temperature 0.8 --top_p 0.9
```

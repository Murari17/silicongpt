# Small Language Model

A local question-answering system that learns from PDF documents using a **GPT-style transformer** with a hybrid generation + retrieval inference strategy.

## ✨ Features

- **Modern GPT-style decoder architecture** — RMSNorm, Rotary Positional Embeddings (RoPE), SwiGLU FFN, Grouped Query Attention (GQA)
- **Fast inference** — KV-cache for autoregressive generation (5-10x speedup)
- **GPU + CPU support** — Auto-detects CUDA GPU, falls back to CPU
- **Mixed precision training** — FP16 on GPU for faster training
- **Advanced sampling** — Top-k + nucleus (top-p) + repetition penalty
- **TF-IDF retrieval fallback** — Reliable answers when generation is uncertain
- **Gradio web interface** — Beautiful chat UI with tunable parameters
- **Best-model checkpointing** — Saves the best model by validation loss

## Architecture

| Component | Detail |
|---|---|
| Type | Decoder-only transformer (GPT-style) |
| Dimensions | 384 |
| Layers | 6 |
| Attention | Grouped Query Attention (6 heads, 2 KV heads) |
| FFN | SwiGLU |
| Normalization | RMSNorm (pre-norm) |
| Positions | Rotary Positional Embeddings (RoPE) |
| Parameters | ~10M |
| Tokenizer | SentencePiece BPE (8000 vocab) |

## Project Files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Full end-to-end runner with CLI flags |
| `textextraction.py` | Extract text from PDFs in `data/raw/` |
| `clean_text.py` | Clean and normalize extracted text |
| `sentence_split.py` | Split cleaned text into sentences |
| `create_dataset.py` | Create QA-style training samples (6+ variants/sentence) |
| `train_tokenizer.py` | Train SentencePiece tokenizer (8000 vocab + chat tokens) |
| `model_def.py` | Transformer architecture definition |
| `train.py` | Training with cosine LR, AdamW, grad clipping, val split |
| `generate.py` | CLI interactive QA with KV-cache generation |
| `app.py` | Gradio web interface |

## Prerequisites

- Python 3.11+ (tested with 3.13)
- NVIDIA GPU with CUDA (optional, CPU works too)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Add Your PDFs

Place PDF files in `data/raw/`:

```
data/raw/unit1.pdf
data/raw/unit2.pdf
```

## Train and Test

### Full pipeline (recommended)

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 15
```

### With custom settings

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 20 --lr 5e-4 --batch-size 32
```

### Preprocess only (no training)

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --skip-train
```

### Test interactively (CLI)

```powershell
.\.venv\Scripts\python.exe generate.py
```

### Launch web interface

```powershell
.\.venv\Scripts\python.exe app.py
```

Or train + launch UI in one go:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 15 --launch-ui
```

## Pipeline Options

| Flag | Description | Default |
|---|---|---|
| `--epochs N` | Training epochs | 15 |
| `--lr RATE` | Max learning rate | 3e-4 |
| `--batch-size N` | Batch size | 16 |
| `--seq-len N` | Sequence length | 128 |
| `--grad-accum N` | Gradient accumulation steps | 4 |
| `--skip-train` | Skip model training | off |
| `--skip-tokenizer` | Skip tokenizer training | off |
| `--skip-preprocess` | Skip PDF extraction and cleaning | off |
| `--launch-ui` | Launch Gradio web UI after training | off |

## Generated Outputs

After running pipeline, these files are produced:

- `data/extracted/*.txt` — Extracted PDF text
- `data/cleaned/corpus_clean.txt` — Cleaned corpus
- `data/cleaned/sentences.txt` — Sentence-level data
- `dataset/training_data.txt` — QA training records
- `tokenizer.model` / `tokenizer.vocab` — Trained tokenizer
- `model.pth` — Final model checkpoint
- `model_best.pth` — Best model (by validation loss)

## Training Features

- **Cosine LR with warmup** — Smooth convergence
- **AdamW** with proper weight decay groups — Better generalization
- **Gradient clipping** (max_norm=1.0) — Prevents explosion
- **Train/validation split** (90/10) — Detects overfitting
- **Mixed precision (FP16)** on GPU — 2x faster training
- **Gradient accumulation** — Larger effective batch size
- **Best-model checkpointing** — Saves the best model automatically
- **Perplexity logging** — Interpretable training metrics

## Documentation PDF

Project documentation (PDF):

https://drive.google.com/file/d/10cIpMwRrJhPXKvwFGhlByFj74Fl-Kubm/view?usp=sharing

## Contribution Workflow

### 1) Fork and Clone

1. Fork this repository on GitHub.
2. Clone your fork locally.
3. Create a feature branch:

```powershell
git checkout -b add-new-pdfs
```

### 2) Add New PDF Data

1. Add your new PDF files to `data/raw/`.
2. Use meaningful file names (for example: `physics.pdf`).

### 3) Rebuild Dataset and Retrain

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 15
```

### 4) Validate Locally

```powershell
.\.venv\Scripts\python.exe generate.py
```

Test at least 3-5 questions and note improvements.

### 5) Commit and Push

```powershell
git add .
git commit -m "Add new PDFs and regenerate training dataset"
git push origin add-new-pdfs
```

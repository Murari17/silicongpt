"""
Training script -- modern training loop for the GPT-style decoder model.

Features:
  - Auto GPU/CPU detection (CUDA -> CPU fallback)
  - Cosine learning rate schedule with linear warmup
  - AdamW optimizer with weight decay
  - Gradient clipping (max_norm=1.0)
  - Train / validation split (90/10)
  - Best-model checkpointing by validation loss
  - Perplexity logging
  - Mixed precision (float16) on GPU for speed
  - Gradient accumulation for larger effective batch size
  - Reproducible seeding
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch.amp import GradScaler, autocast

from model_def import ModelConfig, TransformerModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_batches(
    data: torch.Tensor, seq_len: int, batch_size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create (input, target) batch pairs with next-token prediction."""
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    stride = seq_len * batch_size
    for i in range(0, len(data) - stride, stride):
        batch_x: list[torch.Tensor] = []
        batch_y: list[torch.Tensor] = []
        for j in range(batch_size):
            start = i + j * seq_len
            end = start + seq_len + 1
            if end > len(data):
                break
            chunk = data[start:end]
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])
        if len(batch_x) == batch_size:  # Only keep full batches
            batches.append((torch.stack(batch_x), torch.stack(batch_y)))
    return batches


def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(42)

    base_dir = Path(__file__).resolve().parent
    tokenizer_path = base_dir / "tokenizer.model"
    dataset_path = base_dir / "dataset" / "training_data.txt"
    model_out_path = base_dir / "model.pth"
    best_model_path = base_dir / "model_best.pth"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    # ---- Load tokenizer ----
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))

    # ---- Tokenize corpus ----
    text = dataset_path.read_text(encoding="utf-8")
    records = [r.strip() for r in text.split("\n\n") if r.strip()]
    eos_id = sp.eos_id()

    tokens: list[int] = []
    for record in records:
        tokens.extend(sp.encode(record))
        if eos_id != -1:
            tokens.append(eos_id)

    data = torch.tensor(tokens, dtype=torch.long)
    print(f"Total tokens: {len(data):,}")

    if len(data) < 500:
        raise ValueError("Training data is too small. Add more source PDFs and re-run the pipeline.")

    # ---- Hyperparameters ----
    vocab_size = sp.get_piece_size()
    seq_len = int(os.getenv("SEQ_LEN", "128"))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    grad_accum_steps = int(os.getenv("GRAD_ACCUM", "4"))  # effective batch = 64
    epochs = int(os.getenv("EPOCHS", "15"))
    max_lr = float(os.getenv("LR", "3e-4"))
    min_lr = max_lr * 0.1
    weight_decay = 0.1
    max_grad_norm = 1.0
    val_split = 0.1

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] Training on: {device_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[CPU] Training on CPU (no CUDA GPU detected)")

    use_amp = device.type == "cuda"

    # ---- Model ----
    config = ModelConfig(
        vocab_size=vocab_size,
        dim=384,
        n_layers=6,
        n_heads=6,
        n_kv_heads=2,
        max_seq_len=1024,
        dropout=0.1,
    )
    model = TransformerModel(config).to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # ---- Data split ----
    n = len(data)
    split_idx = int(n * (1 - val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_batches = build_batches(train_data, seq_len=seq_len, batch_size=batch_size)
    val_batches = build_batches(val_data, seq_len=seq_len, batch_size=batch_size)

    if not train_batches:
        raise ValueError("Could not create training batches. Increase dataset size or lower sequence length.")

    print(f"Train batches: {len(train_batches)} | Val batches: {len(val_batches)}")
    print(f"Seq length: {seq_len} | Batch size: {batch_size} | Grad accum: {grad_accum_steps}")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")
    print(f"Epochs: {epochs} | Max LR: {max_lr} | Weight decay: {weight_decay}")
    print("-" * 70)

    # ---- Optimizer & Scheduler ----
    # Separate weight decay: don't apply to biases, norms, embeddings
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=max_lr, betas=(0.9, 0.95), eps=1e-8)

    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(device.type, enabled=use_amp)

    total_steps = len(train_batches) * epochs // grad_accum_steps
    warmup_steps = min(100, total_steps // 10)

    # ---- Training ----
    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_batches):
            x = x.to(device)
            y = y.to(device)

            # Set learning rate
            lr = get_lr(global_step, warmup_steps, total_steps, max_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(x)
                loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum_steps
            epoch_tokens += y.numel()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

        avg_train_loss = epoch_loss / len(train_batches)
        train_ppl = math.exp(min(avg_train_loss, 20))

        # ---- Validation ----
        val_loss = 0.0
        val_tokens = 0
        if val_batches:
            model.eval()
            with torch.no_grad():
                for x, y in val_batches:
                    x = x.to(device)
                    y = y.to(device)
                    with autocast(device_type=device.type, enabled=use_amp):
                        logits, _ = model(x)
                        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
                    val_loss += loss.item()
                    val_tokens += y.numel()
            avg_val_loss = val_loss / len(val_batches)
            val_ppl = math.exp(min(avg_val_loss, 20))
        else:
            avg_val_loss = avg_train_loss
            val_ppl = train_ppl

        elapsed = time.time() - start_time
        tok_per_sec = epoch_tokens / (time.time() - start_time) * (epoch + 1) if epoch == 0 else epoch_tokens / ((time.time() - start_time) / (epoch + 1))

        print(
            f"Epoch {epoch + 1:>3}/{epochs} | "
            f"train: {avg_train_loss:.4f} (ppl {train_ppl:.1f}) | "
            f"val: {avg_val_loss:.4f} (ppl {val_ppl:.1f}) | "
            f"lr: {lr:.2e} | "
            f"time: {format_time(elapsed)}"
        )

        # ---- Checkpoint best model ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_ppl": val_ppl,
            }, best_model_path)
            print(f"  >> New best model saved (val_loss={avg_val_loss:.4f})")

    # Save final model too
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": epochs,
        "val_loss": avg_val_loss,
        "val_ppl": val_ppl,
    }, model_out_path)

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"Training complete in {format_time(total_time)}")
    print(f"Best val loss: {best_val_loss:.4f} (ppl {math.exp(min(best_val_loss, 20)):.1f})")
    print(f"Models saved: {model_out_path} (final), {best_model_path} (best)")


if __name__ == "__main__":
    main()
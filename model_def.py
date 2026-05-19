"""
Small Language Model — GPT-style Decoder-Only Transformer.

Modern architecture features:
  • RMSNorm (pre-norm) for stable training
  • Rotary Positional Embeddings (RoPE) for length generalisation
  • SwiGLU feed-forward network
  • Grouped Query Attention (GQA) for parameter efficiency
  • KV-cache support for fast autoregressive generation
  • Weight tying between embedding and output projection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 8000
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2          # GQA: fewer KV heads → less memory
    max_seq_len: int = 1024
    ffn_hidden_mult: float = 2.667  # SwiGLU hidden dim ≈ dim * mult
    norm_eps: float = 1e-5
    dropout: float = 0.1
    rope_theta: float = 10000.0


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0,
                          device: Optional[torch.device] = None) -> torch.Tensor:
    """Precompute complex-valued RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _rope_apply_single(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a single tensor (B, H, T, D)."""
    B, H, T, D = x.shape
    x_c = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    # freqs: (T, D//2) → (1, 1, T, D//2)
    f = freqs[:T].unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(x_c * f).reshape(B, H, T, D).type_as(x)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor,
               freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Handles GQA where xq and xk may have different numbers of heads.
    xq: (B, n_heads, T, head_dim)
    xk: (B, n_kv_heads, T, head_dim)
    """
    return _rope_apply_single(xq, freqs), _rope_apply_single(xk, freqs)


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # repetition factor

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        B, H, T, D = x.shape
        return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs)

        # KV cache for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)
        new_cache = (k, v)

        # Expand KV heads to match Q heads (GQA)
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = int(config.dim * config.ffn_hidden_mult)
        # Round to nearest multiple of 64 for GPU efficiency
        hidden = 64 * ((hidden + 63) // 64)
        self.w1 = nn.Linear(config.dim, hidden, bias=False)   # gate
        self.w2 = nn.Linear(hidden, config.dim, bias=False)    # down
        self.w3 = nn.Linear(config.dim, hidden, bias=False)    # up
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.norm1 = RMSNorm(config.dim, config.norm_eps)
        self.norm2 = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm architecture
        h, new_cache = self.attention(self.norm1(x), freqs, mask, kv_cache)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len

        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.output.weight = self.embedding.weight

        # Precompute RoPE frequencies (not a parameter, just a buffer)
        head_dim = config.dim // config.n_heads
        freqs = precompute_rope_freqs(head_dim, config.max_seq_len * 2, config.rope_theta)
        self.register_buffer("rope_freqs", freqs, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor,
                kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
                start_pos: int = 0,
                ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: token ids, shape (B, T)
            kv_caches: list of per-layer KV caches for incremental decoding
            start_pos: position offset for RoPE when using KV cache
        Returns:
            logits (B, T, vocab_size), updated KV caches
        """
        B, T = x.shape
        h = self.dropout(self.embedding(x))

        # RoPE frequencies for current positions
        freqs = self.rope_freqs[start_pos: start_pos + T]

        # Causal mask (only needed for prefill, not single-token decode)
        mask = None
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            # If using KV cache, extend mask to cover cached keys
            if kv_caches is not None and kv_caches[0][0].shape[2] > 0:
                cache_len = kv_caches[0][0].shape[2]
                mask = torch.cat([torch.zeros(T, cache_len, device=x.device), mask], dim=-1)

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, new_cache = layer(h, freqs, mask, cache_i)
            new_caches.append(new_cache)

        h = self.norm(h)
        logits = self.output(h)
        return logits, new_caches

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

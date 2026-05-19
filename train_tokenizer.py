"""
SentencePiece tokenizer trainer.

Improvements:
  • 8000 vocab for better subword coverage of technical terms
  • Special chat-format tokens: <|user|>, <|assistant|>, <|system|>
  • Higher character coverage for scientific/special characters
"""

import sentencepiece as spm
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "dataset" / "training_data.txt"
MODEL_PREFIX = BASE_DIR / "tokenizer"

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Training data not found: {INPUT_PATH}")

spm.SentencePieceTrainer.train(
    input=str(INPUT_PATH),
    model_prefix=str(MODEL_PREFIX),
    vocab_size=8000,
    model_type="bpe",
    hard_vocab_limit=False,
    character_coverage=0.9995,
    # Add special tokens for chat format
    user_defined_symbols=["<|user|>", "<|assistant|>", "<|system|>"],
    # Byte fallback for unknown characters
    byte_fallback=True,
    # Normalization
    normalization_rule_name="identity",
    # Treat whitespace properly
    split_by_whitespace=True,
    max_sentence_length=8192,
)

print("Tokenizer trained (vocab_size=8000, with chat tokens)")
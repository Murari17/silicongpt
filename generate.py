"""
Generation script -- interactive QA with the trained model.

Features:
  - KV-cache-enabled generation (5-10x faster)
  - Nucleus (top-p) + top-k sampling
  - Repetition penalty to prevent loops
  - TF-IDF retrieval fallback for reliability
  - Streaming token output
  - Multi-turn conversation support
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from model_def import ModelConfig, TransformerModel


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "to", "for", "and",
    "or", "on", "with", "by", "do", "you", "mean", "how", "why", "can",
    "tell", "me", "about", "explain", "define", "describe", "does", "it",
}


def normalize_text(text: str) -> str:
    text = text.replace("\u2047", " ")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


# ---------------------------------------------------------------------------
# TF-IDF Retrieval
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """TF-IDF retriever using scikit-learn vectorizer with cleaning, dedup, and reranking."""

    RETRIEVER_LOG = False

    def __init__(self, sentences: list[str]):
        # Clean and deduplicate sentences while keeping originals
        cleaned: list[str] = []
        originals: list[str] = []
        seen: set[str] = set()
        for s in sentences:
            s_norm = normalize_text(s)
            if not s_norm:
                continue
            # filter obvious bad candidates early
            if not self._is_good_answer_candidate(s_norm):
                continue
            key = s_norm.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(s_norm)
            originals.append(s)

        self.sentences = originals
        self.cleaned = cleaned
        self.n = len(self.cleaned)

        # Build vectorizer on cleaned lowercase text
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda txt: tokenize_words(txt),
            lowercase=True,
            norm='l2',
            token_pattern=None,
        )
        if self.cleaned:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned)
        else:
            self.tfidf_matrix = None

    @staticmethod
    def _is_good_answer_candidate(sentence: str) -> bool:
        s = sentence.strip()
        if len(s) < 20:
            return False
        if s.rstrip().endswith("?"):
            return False
        # Filter question-like prompts even if OCR ends them with '.'
        if re.match(r"^\s*(what|why|how|when|where|who|which|whom|whose|explain|define|describe)\b", s, re.IGNORECASE):
            if len(tokenize_words(s)) <= 16:
                return False
        if re.match(r"^\s*(\(?\s*[ivxab]\s*\)?\.?\s|Q\s*\.?\s*\d)", s, re.IGNORECASE):
            return False
        if "Ask:" in s or "Answer:" in s:
            return False
        alpha_ratio = sum(1 for c in s if c.isalpha()) / max(len(s), 1)
        if alpha_ratio < 0.25:
            return False
        words = s.split()
        if len(words) <= 8:
            if any(char.isdigit() for char in s):
                return False
            up_initials = sum(1 for w in words if w and w[0].isupper())
            if up_initials >= max(2, len(words) // 2):
                return False
        head_sample = words[:6]
        caps_count = sum(1 for w in head_sample if w.isalpha() and w.isupper())
        if caps_count >= 2:
            return False
        return True

    def _rerank(self, question: str, candidates: list[tuple[int, float]]) -> list[tuple[int, float]]:
        q_words = [w for w in tokenize_words(question) if w not in STOPWORDS]
        reranked: list[tuple[int, float]] = []
        for idx, base_score in candidates:
            sent = self.cleaned[idx]
            score = float(base_score)
            # Lexical overlap boost
            overlap = sum(1 for w in q_words if re.search(rf"\b{re.escape(w)}\b", sent.lower()))
            score += 0.05 * overlap
            # Definition boost
            s_lower = sent.lower()
            for qw in q_words:
                if re.search(rf"\b{re.escape(qw)}\b\s+(is|are|refers|means)\b", s_lower):
                    score *= 1.25
                    break
            # Penalize very short candidates
            if len(tokenize_words(sent)) < 6:
                score *= 0.8

            reranked.append((idx, score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def query(self, question: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self.tfidf_matrix is None:
            return []
        q_text = normalize_text(question)
        q_vec = self.vectorizer.transform([q_text])
        sims = linear_kernel(q_vec, self.tfidf_matrix).flatten()
        # Collect top candidates indices
        top_idx = np.argsort(-sims)[: max(50, top_k * 5)]
        candidates = []
        for i in top_idx:
            if sims[i] <= 0:
                continue
            candidates.append((int(i), float(sims[i])))

        # Rerank with simple heuristics
        reranked = self._rerank(question, candidates)

        results: list[tuple[str, float]] = []
        seen_texts: set[str] = set()
        for idx, score in reranked:
            orig = self.sentences[idx]
            if orig in seen_texts:
                continue
            seen_texts.add(orig)
            results.append((orig, float(score)))
            if len(results) >= top_k:
                break

        if self.RETRIEVER_LOG:
            print("[retriever] question:", question)
            for i, (t, s) in enumerate(results, 1):
                print(f"  {i}. {s:.3f} {t[:200]}")

        return results


class BM25Retriever:
    """BM25 retriever wrapper using rank_bm25 (if available)."""

    def __init__(self, sentences: list[str]):
        from rank_bm25 import BM25Okapi

        cleaned: list[str] = []
        originals: list[str] = []
        seen: set[str] = set()
        for s in sentences:
            s_norm = normalize_text(s)
            if not s_norm:
                continue
            if not TFIDFRetriever._is_good_answer_candidate(s_norm):
                continue
            key = s_norm.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(s_norm)
            originals.append(s)

        self.sentences = originals
        self.cleaned = cleaned
        self.tokenized = [tokenize_words(t) for t in cleaned]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def query(self, question: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.bm25:
            return []
        q = tokenize_words(question)
        scores = self.bm25.get_scores(q)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: top_k * 5]
        results = []
        for i in idxs:
            if scores[i] <= 0:
                continue
            results.append((self.sentences[i], float(scores[i])))
            if len(results) >= top_k:
                break
        return results


class HybridRetriever:
    """Hybrid retriever combining TF-IDF and BM25 with score fusion."""

    def __init__(self, sentences: list[str], alpha: float = 0.6):
        self.alpha = alpha
        self.tfidf = TFIDFRetriever(sentences)
        self.bm25 = None
        try:
            self.bm25 = BM25Retriever(sentences)
        except Exception:
            self.bm25 = None

    @staticmethod
    def _normalize_scores(results: list[tuple[str, float]]) -> dict[str, float]:
        if not results:
            return {}
        vals = [s for _, s in results]
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return {t: 1.0 for t, _ in results}
        return {t: (s - lo) / (hi - lo) for t, s in results}

    def query(self, question: str, top_k: int = 5) -> list[tuple[str, float]]:
        tfidf_results = self.tfidf.query(question, top_k=max(12, top_k * 3))
        bm25_results = self.bm25.query(question, top_k=max(12, top_k * 3)) if self.bm25 else []

        tfidf_norm = self._normalize_scores(tfidf_results)
        bm25_norm = self._normalize_scores(bm25_results)

        all_texts = set(tfidf_norm) | set(bm25_norm)
        fused: list[tuple[str, float]] = []
        for text in all_texts:
            score = self.alpha * tfidf_norm.get(text, 0.0) + (1.0 - self.alpha) * bm25_norm.get(text, 0.0)
            fused.append((text, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]


def create_retriever(sentences: list[str]) -> object:
    """Create the best available retriever (hybrid BM25+TF-IDF when possible)."""
    try:
        return HybridRetriever(sentences)
    except Exception:
        return TFIDFRetriever(sentences)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Path, sp: spm.SentencePieceProcessor,
               device: torch.device) -> TransformerModel:
    """Load model from checkpoint, auto-detecting config."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # New-style checkpoint with config
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
        # Override vocab size to match current tokenizer
        config.vocab_size = sp.get_piece_size()
        model = TransformerModel(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Loaded model: {model.count_parameters():,} params "
              f"(val_loss={checkpoint.get('val_loss', '?')}, "
              f"epoch={checkpoint.get('epoch', '?')})")
    else:
        # Legacy checkpoint (old model format) -- try to load with defaults
        state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        # Try to detect old-style model
        if "embedding.weight" in state_dict and "encoder.layers.0.norm1.weight" in state_dict:
            print("[!] Detected old encoder-style model. Please retrain with the new architecture.")
            print("   Run: python run_pipeline.py --epochs 15")
            sys.exit(1)

        # Detect config from weights
        dim = state_dict["embedding.weight"].shape[1]
        n_layers = sum(1 for k in state_dict if "layers." in k and k.endswith(".norm1.weight"))
        config = ModelConfig(
            vocab_size=sp.get_piece_size(),
            dim=dim,
            n_layers=n_layers,
        )
        model = TransformerModel(config)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded legacy model: {model.count_parameters():,} params")

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Generation with KV cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_answer(
    model: TransformerModel,
    sp: spm.SentencePieceProcessor,
    question: str,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    stream: bool = True,
) -> str:
    """Generate answer using KV cache with top-p + top-k sampling."""

    # Format prompt with chat tokens (or legacy format)
    user_tag = "<|user|>"
    asst_tag = "<|assistant|>"

    # Check if tokenizer knows chat tokens
    user_id = sp.piece_to_id(user_tag)
    if user_id == sp.unk_id():
        # Fall back to legacy format
        prompt = f"Ask: {question}\nAnswer:"
    else:
        prompt = f"{user_tag} {question}\n{asst_tag}"

    input_ids = sp.encode(prompt)
    generated_ids: list[int] = []
    eos_id = sp.eos_id()

    # Prefill: process entire prompt at once
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits, kv_caches = model(x)

    # Track generated token IDs for repetition penalty
    all_token_ids = list(input_ids)

    for step in range(max_new_tokens):
        # Get logits for the last position
        next_logits = logits[0, -1].float()

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(all_token_ids[-64:]):  # Look back 64 tokens
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        # Temperature
        next_logits = next_logits / max(temperature, 0.05)

        # Top-k filtering
        if top_k > 0:
            k = min(top_k, next_logits.size(0))
            top_k_vals, _ = torch.topk(next_logits, k)
            threshold = top_k_vals[-1]
            next_logits[next_logits < threshold] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            # Scatter back
            next_logits.scatter_(0, sorted_indices, sorted_logits)

        # Sample
        probs = torch.softmax(next_logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())

        # Stop conditions
        if next_id == eos_id:
            break

        generated_ids.append(next_id)
        all_token_ids.append(next_id)

        # Check for stop patterns
        decoded_so_far = sp.decode(generated_ids)
        stop_patterns = ["\nAsk:", "\n<|user|>", "\n<|system|>"]
        should_stop = any(p in decoded_so_far for p in stop_patterns)
        if should_stop:
            # Trim to before the stop pattern
            for p in stop_patterns:
                if p in decoded_so_far:
                    decoded_so_far = decoded_so_far.split(p, 1)[0]
            break

        # Stream output
        if stream and step % 3 == 0:
            text = sp.decode(generated_ids)
            sys.stdout.write(f"\r{normalize_text(text)}")
            sys.stdout.flush()

        # Single-token decode with KV cache
        start_pos = len(input_ids) + step
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, kv_caches = model(x, kv_caches=kv_caches, start_pos=start_pos)

    if stream:
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()

    answer = normalize_text(sp.decode(generated_ids))

    # Clean up any remaining markers
    for marker in ("\nAsk:", "Ask:", "\nAnswer:", "Answer:",
                   "<|user|>", "<|assistant|>", "<|system|>"):
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    return answer if answer else "I need more training data to answer this clearly."


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def is_weak_answer(question: str, answer: str) -> bool:
    """Check if the generated answer is truly broken/useless."""
    answer = normalize_text(answer)
    # Empty or extremely short
    if not answer or len(answer) < 15:
        return True
    # Contains training artifacts
    if "Ask:" in answer or "Answer:" in answer:
        return True
    # Is just a repeat of the question
    if answer.lower().strip("?.! ") == question.lower().strip("?.! "):
        return True
    # Generated a question instead of an answer
    if answer.strip().endswith("?"):
        return True
    # Starts with weird punctuation
    if re.match(r"^[^\w\s\(\[]", answer.strip()):
        return True
    # Repetitive gibberish (same 3-word phrase repeated 3+ times)
    words = answer.split()
    if len(words) >= 9:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        tri_counts = Counter(trigrams)
        most_common = tri_counts.most_common(1)
        if most_common and most_common[0][1] >= 3:
            return True
    return False


def _answer_relevance(question: str, answer: str) -> float:
    """Score how relevant an answer is to the question (0.0 to 1.0)."""
    q_keywords = {w for w in tokenize_words(question) if w not in STOPWORDS}
    if not q_keywords:
        return 1.0  # Can't judge, assume relevant
    a_words = set(tokenize_words(answer))
    overlap = len(q_keywords & a_words)
    return overlap / len(q_keywords)


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    tokenizer_path = base_dir / "tokenizer.model"
    best_model_path = base_dir / "model_best.pth"
    model_path = base_dir / "model.pth"
    sentences_path = base_dir / "data" / "cleaned" / "sentences.txt"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")

    # Prefer best model, fall back to final model
    if best_model_path.exists():
        active_model_path = best_model_path
    elif model_path.exists():
        active_model_path = model_path
    else:
        raise FileNotFoundError("No model checkpoint found. Run training first.")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] Inference on: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[CPU] Inference on CPU")

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))

    # Load model
    model = load_model(active_model_path, sp, device)

    # Load retrieval fallback
    retriever: Optional[object] = None
    if sentences_path.exists():
        sentences = [
            line.strip()
            for line in sentences_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if sentences:
            retriever = create_retriever(sentences)
            print(f"Loaded {len(sentences)} sentences for retrieval fallback")

    print("\n" + "=" * 60)
    print("  Small Language Model -- Interactive QA")
    print("  Type 'quit' or 'exit' to stop. 'clear' for new conversation.")
    print("=" * 60 + "\n")

    conversation_history: list[str] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if question.lower() == "clear":
            conversation_history.clear()
            print("Conversation cleared.\n")
            continue

        # Generate answer
        generated = generate_answer(model, sp, question, device, stream=False)

        # TF-IDF retrieval fallback
        final_answer = generated
        retrieval_used = False

        if retriever:
            results = retriever.query(question, top_k=3)
            best_retrieved = results[0] if results else ("", 0.0)
            retrieved_text, retrieved_score = best_retrieved

            gen_is_weak = is_weak_answer(question, generated)
            gen_relevance = _answer_relevance(question, generated)

            if gen_is_weak:
                # Model produced garbage -- use retrieval if reasonably confident
                if retrieved_score >= 0.30 and not is_weak_answer(question, retrieved_text):
                    final_answer = normalize_text(retrieved_text)
                    retrieval_used = True
                else:
                    final_answer = "I could not find a clear answer in the loaded notes."
            elif gen_relevance < 0.4 and retrieved_score >= 0.45:
                # Model answer has poor keyword overlap but retrieval is strong
                if not is_weak_answer(question, retrieved_text):
                    final_answer = normalize_text(retrieved_text)
                    retrieval_used = True

        source = " [retrieved]" if retrieval_used else " [generated]"
        print(f"AI{source}: {final_answer}\n")

        conversation_history.append(f"Q: {question}")
        conversation_history.append(f"A: {final_answer}")


if __name__ == "__main__":
    main()

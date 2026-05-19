"""
Dataset creator — builds QA-style training samples from sentence data.

Improvements over original:
  • 6+ question variants per sentence (definition, explain, describe, etc.)
  • Multi-sentence context windows for richer answers
  • Chat-format tokens for structured prompting
  • Reproducible shuffle
"""

from __future__ import annotations

import random
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "cleaned" / "sentences.txt"
OUTPUT_DIR = BASE_DIR / "dataset"
OUTPUT_PATH = OUTPUT_DIR / "training_data.txt"

SEED = 42

# Chat format tokens (matched in tokenizer special tokens)
USER_TAG = "<|user|>"
ASST_TAG = "<|assistant|>"


def normalize_question(text: str) -> str:
    return text.strip().rstrip(".?!")


def extract_subject_verb(sentence: str) -> tuple[str, str]:
    """Try to extract subject and verb from a definition-style sentence."""
    m = re.match(r"^([A-Za-z][A-Za-z0-9\-\s]{2,60}?)\s+(is|are|was|were|refers to|means)\s+", sentence)
    if m:
        return normalize_question(m.group(1)), m.group(2)
    return "", ""


def build_question_variants(sentence: str) -> list[str]:
    """Generate multiple question phrasings from a single sentence."""
    subject, verb = extract_subject_verb(sentence)
    variants: list[str] = []

    if subject:
        # Definition-style questions
        if verb in ("is", "refers to", "means"):
            variants.append(f"What is {subject}?")
            variants.append(f"Define {subject}.")
            variants.append(f"Can you explain what {subject} is?")
        elif verb in ("are", "were"):
            variants.append(f"What are {subject}?")
            variants.append(f"Define {subject}.")
            variants.append(f"Can you explain what {subject} are?")
        else:
            variants.append(f"What does {subject} refer to?")

        # Common follow-up styles
        variants.append(f"Tell me about {subject}.")
        variants.append(f"Describe {subject}.")
    else:
        # Non-definition sentences — use generic question forms
        words = sentence.split()
        short = " ".join(words[:10]).strip()
        short = normalize_question(short)
        variants.append(f"Explain: {short}?")
        variants.append(f"What do you know about: {short}?")
        variants.append(f"Describe: {short}.")

    return variants


def build_context_answers(sentences: list[str], idx: int, window: int = 2) -> list[str]:
    """Build answer variants: single sentence + multi-sentence context."""
    answers = [sentences[idx]]

    # Multi-sentence context (current + next 1-2 sentences)
    end = min(idx + window + 1, len(sentences))
    if end > idx + 1:
        context = " ".join(sentences[idx:end])
        if len(context) < 800:  # Don't create absurdly long answers
            answers.append(context)

    return answers


def format_record(question: str, answer: str) -> str:
    """Format a single QA pair using chat-style tokens."""
    return f"{USER_TAG} {question}\n{ASST_TAG} {answer}\n"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Sentence file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        raw_sentences = f.readlines()

    sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= 20]

    if not sentences:
        raise ValueError("No usable sentences found (all too short).")

    dataset: list[str] = []

    for idx, sentence in enumerate(sentences):
        questions = build_question_variants(sentence)
        answers = build_context_answers(sentences, idx)

        for question in questions:
            for answer in answers:
                dataset.append(format_record(question, answer))

    # Add the legacy Ask/Answer format for backward compat during transition
    for idx, sentence in enumerate(sentences):
        subject, verb = extract_subject_verb(sentence)
        if subject and verb in ("is", "are"):
            q = f"What {verb} {subject}?"
            dataset.append(f"Ask: {q}\nAnswer: {sentence}\n")

    # Shuffle for better training dynamics
    random.seed(SEED)
    random.shuffle(dataset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in dataset:
            f.write(record + "\n")

    print(f"Dataset created: {len(dataset)} records -> {OUTPUT_PATH}")
    print(f"  Sentences used: {len(sentences)}")
    print(f"  Avg records/sentence: {len(dataset) / len(sentences):.1f}")


if __name__ == "__main__":
    main()
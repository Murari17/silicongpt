from pathlib import Path
import sys

from generate import (
    load_model,
    generate_answer,
    create_retriever,
    is_weak_answer,
    _answer_relevance,
)
import sentencepiece as spm
import torch


def load_resources(base_dir: Path):
    tokenizer_path = base_dir / "tokenizer.model"
    best_model_path = base_dir / "model_best.pth"
    model_path = base_dir / "model.pth"
    sentences_path = base_dir / "data" / "cleaned" / "sentences.txt"

    sp = None
    model = None
    retriever = None

    if tokenizer_path.exists():
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_path))
    else:
        print("Tokenizer not found; generation will be skipped.")

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load model if available
    active = best_model_path if best_model_path.exists() else (model_path if model_path.exists() else None)
    if active and sp is not None:
        try:
            model = load_model(active, sp, device)
        except Exception as e:
            print("Failed to load model:", e)
            model = None

    # retriever
    if sentences_path.exists():
        sentences = [l.strip() for l in sentences_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if sentences:
            retriever = create_retriever(sentences)

    return sp, model, retriever, device


def compare_questions(questions):
    base_dir = Path(__file__).resolve().parent
    sp, model, retriever, device = load_resources(base_dir)

    for q in questions:
        print("\n" + "-" * 80)
        print("Question:", q)

        # Generated answer
        if model and sp:
            try:
                gen = generate_answer(model, sp, q, device, stream=False)
            except Exception as e:
                gen = f"[generation failed: {e}]"
        else:
            gen = "[model not available]"

        weak = is_weak_answer(q, gen) if isinstance(gen, str) else True
        rel = _answer_relevance(q, gen) if isinstance(gen, str) else 0.0

        print("Generated:", gen)
        print(f"  weak:{weak}  relevance:{rel:.2f}")

        # Retrieval
        if retriever:
            results = retriever.query(q, top_k=5)
            if results:
                print("Top retrievals:")
                for i, (s, score) in enumerate(results, 1):
                    print(f"  {i}. ({score:.3f}) {s[:200]}")
            else:
                print("  [no retrieval results]")
        else:
            print("  [retriever not available]")


if __name__ == '__main__':
    sample_qs = [
        "What is a wave?",
        "What is hardness of water?",
        "Explain environmental engineering",
    ]
    compare_questions(sample_qs)

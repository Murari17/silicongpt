from pathlib import Path
import argparse
from generate import TFIDFRetriever, normalize_text, create_retriever
try:
    from generate import BM25Retriever
except Exception:
    BM25Retriever = None


def load_sentences(base_dir: Path):
    path = base_dir / "data" / "cleaned" / "sentences.txt"
    if not path.exists():
        raise FileNotFoundError(f"Sentences file not found: {path}")
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines


def evaluate(retriever: TFIDFRetriever, sentences: list[str], top_ks=(1, 3, 5), limit: int | None = None):
    totals = {k: 0 for k in top_ks}
    n = 0
    dataset = sentences[:limit] if limit else sentences
    for s in dataset:
        q = normalize_text(s)
        if not q:
            continue
        n += 1
        results = retriever.query(q, top_k=max(top_ks))
        found = [normalize_text(r[0]) for r in results]
        for k in top_ks:
            topk = found[:k]
            if normalize_text(s) in topk:
                totals[k] += 1

    print(f"Evaluated {n} queries")
    for k in top_ks:
        recall = totals[k] / n if n else 0.0
        print(f"Recall@{k}: {recall:.3f} ({totals[k]}/{n})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate retrievers on sentence self-recall")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation queries")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    sents = load_sentences(base)
    print("Evaluating TF-IDF retriever...")
    tfidf_ret = TFIDFRetriever(sents)
    evaluate(tfidf_ret, sents, top_ks=(1, 3, 5), limit=args.limit)

    if BM25Retriever is not None:
        print("\nEvaluating BM25 retriever...")
        bm25_ret = BM25Retriever(sents)
        evaluate(bm25_ret, sents, top_ks=(1, 3, 5), limit=args.limit)
    else:
        print("BM25 retriever not available.")

    print("\nEvaluating Hybrid retriever...")
    hybrid_ret = create_retriever(sents)
    evaluate(hybrid_ret, sents, top_ks=(1, 3, 5), limit=args.limit)

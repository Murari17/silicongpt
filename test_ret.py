import math
import re
from collections import Counter
from pathlib import Path
from generate import TFIDFRetriever, tokenize_words, STOPWORDS, normalize_text

sents = [l.strip() for l in Path('data/cleaned/sentences.txt').read_text(encoding='utf-8').splitlines() if l.strip()]

ret = TFIDFRetriever(sents)

def my_query(question):
    q_words = [w for w in tokenize_words(question) if w not in STOPWORDS]
    if not q_words: return []
    q_tf = Counter(q_words)
    q_tfidf = {w: tf * ret.idf(w) for w, tf in q_tf.items()}
    q_norm = math.sqrt(sum(v*v for v in q_tfidf.values())) or 1.0
    
    scores = []
    for idx, doc_words in enumerate(ret.doc_words):
        sent = ret.sentences[idx]
        if not ret._is_good_answer_candidate(sent):
            continue
            
        d_tf = Counter(doc_words)
        d_tfidf = {w: tf * ret.idf(w) for w, tf in d_tf.items()}
        d_norm = math.sqrt(sum(v*v for v in d_tfidf.values())) or 1.0
        
        dot = sum(q_tfidf.get(w,0) * d_tfidf.get(w,0) for w in set(q_words))
        sim = dot / (q_norm * d_norm)
        
        # Penalize short sentences (likely fragments/headings)
        if len(doc_words) < 10:
            sim *= 0.7
            
        # Boost definitions
        s_lower = sent.lower()
        for qw in q_words:
            if f"{qw} is" in s_lower or f"{qw} are" in s_lower or f"{qw} refers" in s_lower or f"{qw} means" in s_lower:
                sim *= 1.3
                break
                
        scores.append((sent, sim))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:5]

print("Hardness of water:")
for t, s in my_query("what is hardness of water"):
    print(f"[{s:.2f}] {t}")

print("\nEnvironment:")
for t, s in my_query("what is environment"):
    print(f"[{s:.2f}] {t}")

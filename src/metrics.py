"""Answer metrics (SQuAD-style EM/F1 adapted to Vietnamese) and retrieval metrics."""
import re
import string
import unicodedata
from collections import Counter

_PUNCT = set(string.punctuation) | set("“”‘’–—…«»")


def normalize(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred, golds):
    p = normalize(pred)
    return float(any(p == normalize(g) for g in golds))


def f1(pred, golds):
    """Token F1 over Vietnamese syllables (space-separated), best over gold answers."""
    p = normalize(pred).split()
    best = 0.0
    for g in golds:
        g = normalize(g).split()
        common = sum((Counter(p) & Counter(g)).values())
        if common:
            pr, rc = common / len(p), common / len(g)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def contains(pred, golds):
    """1 if a gold answer appears inside the prediction (LLMs often add a few words)."""
    p = normalize(pred)
    return float(any(normalize(g) in p for g in golds))


def hit_at_k(retrieved_ctx, gold_ctx, k):
    return float(gold_ctx in retrieved_ctx[:k])


def reciprocal_rank(retrieved_ctx, gold_ctx, k):
    for rank, c in enumerate(retrieved_ctx[:k], 1):
        if c == gold_ctx:
            return 1.0 / rank
    return 0.0

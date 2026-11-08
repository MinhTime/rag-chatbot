"""Retrievers: dense (FAISS) and hybrid (BM25 + FAISS fused with Reciprocal Rank Fusion)."""
import numpy as np
from rank_bm25 import BM25Okapi

from .metrics import normalize


def tokenize(text):
    """Vietnamese syllables, lower-cased, punctuation removed."""
    return normalize(text).split()


class Retriever:
    """search(question, k) -> chunk Documents; search_batch(questions, k) -> chunk index lists.

    mode="dense":  FAISS only (V0).
    mode="hybrid": FAISS top-`candidates` + BM25 top-`candidates`, fused with RRF (V1).
    reranker="<cross-encoder>": re-score the top-`candidates` with a cross-encoder (V2).
    """

    def __init__(self, index, embeddings, chunks, mode="dense", candidates=20, rrf_k=60,
                 reranker="", device="cuda"):
        self.index, self.emb, self.chunks = index, embeddings, chunks
        self.mode, self.candidates, self.rrf_k = mode, candidates, rrf_k
        self.reranker = None
        if reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker, max_length=512, device=device)
            if device == "cuda":
                self.reranker.model.half()  # fp16 on GPU: ~2x faster, same ranking
        # FAISS position -> position in `chunks` (same order as from_documents, mapped to be safe)
        store, id_map = index.docstore, index.index_to_docstore_id
        self.pos2chunk = np.array([store.search(id_map[i]).metadata["chunk_id"]
                                   for i in range(index.index.ntotal)])
        self.bm25 = None
        if mode == "hybrid":
            self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def _dense(self, questions, k):
        q = np.asarray(self.emb.embed_documents(questions), dtype="float32")
        _, pos = self.index.index.search(q, k)
        return [[int(self.pos2chunk[p]) for p in row if p >= 0] for row in pos]

    def _bm25(self, question, k):
        scores = self.bm25.get_scores(tokenize(question))
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        return [int(i) for i in top[np.argsort(-scores[top])]]

    def _rrf(self, rankings, k):
        fused = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return [cid for cid, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]

    def _first_stage(self, questions, k):
        if self.mode == "dense":
            return self._dense(questions, k)
        n = max(self.candidates, k)
        dense = self._dense(questions, n)
        return [self._rrf([d, self._bm25(q, n)], k) for q, d in zip(questions, dense)]

    def _rerank(self, questions, cand, k):
        pairs = [(q, self.chunks[i].page_content) for q, row in zip(questions, cand) for i in row]
        scores = self.reranker.predict(pairs, batch_size=64, show_progress_bar=False)
        out, j = [], 0
        for row in cand:
            s = scores[j:j + len(row)]
            j += len(row)
            out.append([row[i] for i in np.argsort(-s)[:k]])
        return out

    def search_batch(self, questions, k):
        if self.reranker is None:
            return self._first_stage(questions, k)
        return self._rerank(questions, self._first_stage(questions, max(self.candidates, k)), k)

    def search(self, question, k):
        return [self.chunks[i] for i in self.search_batch([question], k)[0]]

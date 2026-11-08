import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.data import split_docs
from src.rag import build_index
from src.retrieval import Retriever

DOCS = [
    Document(page_content="Bức tường Berlin được xây dựng năm 1961 để chia cắt Đông và Tây Berlin.",
             metadata={"ctx_id": 0, "title": "Bức tường Berlin"}),
    Document(page_content="Tokyo là thủ đô của Nhật Bản, nằm trên đảo Honshu.",
             metadata={"ctx_id": 1, "title": "Nhật Bản"}),
    Document(page_content="Quảng Châu là thủ phủ của tỉnh Quảng Đông, Trung Quốc.",
             metadata={"ctx_id": 2, "title": "Quảng Châu"}),
]


@pytest.fixture(scope="module")
def setup():
    chunks = split_docs(DOCS, 500, 100)
    emb = DeterministicFakeEmbedding(size=32)
    return build_index(chunks, emb), emb, chunks


def test_chunks_keep_metadata_and_title(setup):
    _, _, chunks = setup
    assert [c.metadata["chunk_id"] for c in chunks] == [0, 1, 2]
    assert chunks[1].page_content.startswith("Nhật Bản: ")


def test_dense_returns_k_unique(setup):
    index, emb, chunks = setup
    r = Retriever(index, emb, chunks, mode="dense")
    got = r.search_batch(["Thủ đô Nhật Bản?"], 2)[0]
    assert len(got) == 2 and len(set(got)) == 2


def test_hybrid_bm25_finds_keyword_match(setup):
    # fake embeddings are random, so a correct top-1 must come from BM25 via RRF
    index, emb, chunks = setup
    r = Retriever(index, emb, chunks, mode="hybrid", candidates=3)
    assert r._bm25("Bức tường Berlin xây năm nào", 1) == [0]
    top = r.search("Bức tường Berlin xây năm nào", 3)
    assert 0 in [d.metadata["ctx_id"] for d in top]


def test_rrf_prefers_items_ranked_by_both(setup):
    index, emb, chunks = setup
    r = Retriever(index, emb, chunks, mode="hybrid")
    assert r._rrf([[1, 2, 0], [2, 1, 0]], 3)[2] == 0
    assert r._rrf([[1, 2], [2, 0]], 1) == [2]

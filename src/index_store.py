"""Build the FAISS index once (offline, GPU) and load it at serving time.

    python -m src.index_store --viquad_dev data/dev.json --out artifacts/index
"""
import argparse
import json
import os
import time

from langchain_community.vectorstores import FAISS

from .config import Config
from .data import build_corpus, load_viquad, split_docs
from .rag import build_index, get_embeddings


def save(index, cfg, path, extra=None):
    index.save_local(path)
    meta = {"embed_model": cfg.embed_model, "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap, "add_title": cfg.add_title,
            "chunks": index.index.ntotal, **(extra or {})}
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load(path, embeddings):
    """Returns (index, chunks ordered by chunk_id, meta)."""
    index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)  # own file
    docs = [index.docstore.search(i) for i in index.index_to_docstore_id.values()]
    chunks = sorted(docs, key=lambda d: d.metadata["chunk_id"])
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    return index, chunks, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viquad_dev", default=Config.viquad_dev)
    ap.add_argument("--out", default="artifacts/index")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = Config(viquad_dev=a.viquad_dev)
    t = time.time()
    docs, _ = build_corpus(load_viquad(cfg.viquad_dev))
    chunks = split_docs(docs, cfg.chunk_size, cfg.chunk_overlap, cfg.add_title)
    index = build_index(chunks, get_embeddings(cfg.embed_model, a.device))
    save(index, cfg, a.out, {"build_sec": round(time.time() - t, 1)})
    print(f"saved {index.index.ntotal} chunks to {a.out} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()

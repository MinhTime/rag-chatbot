"""All settings in one place. Change a value here, not inside the code."""
from dataclasses import dataclass


@dataclass
class Config:
    version: str = "v0_dense"  # results go to results/<version>/

    # data
    viquad_dev: str = "/kaggle/input/uit-viquad-2-0/dev.json"  # Kaggle: gamind/uit-viquad-2-0
    n_questions: int = 200  # answer test set; the first 50 are the original v0 questions
    n_core: int = 50
    seed: int = 42

    # chunking
    chunk_size: int = 500
    chunk_overlap: int = 100
    add_title: bool = True  # prepend the Wikipedia article title to every chunk

    # retrieval
    embed_model: str = "BAAI/bge-m3"
    retriever: str = "dense"  # "dense" (V0) | "hybrid" (V1: BM25 + dense, RRF)
    candidates: int = 20  # hybrid: candidates taken from each retriever before fusion
    rrf_k: int = 60  # RRF constant (Cormack et al. 2009)
    reranker: str = ""  # V2: cross-encoder, e.g. "BAAI/bge-reranker-v2-m3" ("" = off)
    top_k: int = 4
    max_k_curve: int = 10  # Hit@k curve is computed for k = 1..max_k_curve

    # generation (Ollama)
    llm_model: str = "qwen2.5:7b"  # Q4_K_M, ~4.7 GB
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: int = 128
    think: bool = False  # Qwen3 "thinking" mode; off = short direct answers

    # judges: the main judge is from a DIFFERENT model family than the answer model
    # (avoids self-preference bias); the second judge is used to measure agreement
    judge_model: str = "gemma2:9b"
    second_judge: str = "qwen2.5:7b"

    out_dir: str = "results"

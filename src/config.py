"""All settings in one place. Change a value here, not inside the code."""
from dataclasses import dataclass


@dataclass
class Config:
    # data
    viquad_dev: str = "/kaggle/input/uit-viquad-2-0/dev.json"  # Kaggle dataset gamind/uit-viquad-2-0
    n_questions: int = 50
    seed: int = 42

    # chunking
    chunk_size: int = 500
    chunk_overlap: int = 100
    add_title: bool = True  # prepend the Wikipedia article title to every chunk

    # retrieval
    embed_model: str = "BAAI/bge-m3"
    top_k: int = 4
    max_k_curve: int = 10  # Hit@k curve is computed for k = 1..max_k_curve

    # generation (Ollama)
    llm_model: str = "qwen2.5:7b"  # Q4_K_M, ~4.7 GB
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: int = 128

    # judge (same local model, see README "Limitations")
    judge_model: str = "qwen2.5:7b"

    out_dir: str = "results"

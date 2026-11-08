"""Build one Kaggle notebook per experiment version from src/*.py.

The notebook writes src/ with %%writefile, so Kaggle runs exactly the code in this repo.
Run:  python scripts/build_notebook.py            # all versions
      python scripts/build_notebook.py v1_hybrid  # one version
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ["__init__.py", "config.py", "data.py", "metrics.py", "retrieval.py", "rag.py", "judge.py",
       "ollama_utils.py", "index_store.py", "evaluate.py", "plots.py"]

# Each version changes ONE factor vs the previous one. `why` goes into the notebook header.
VERSIONS = {
    "v0_dense": {
        "overrides": {},
        "why": "Baseline: dense retrieval only (bge-m3 + FAISS, top-4), Qwen2.5 7B.",
    },
    "v1_hybrid": {
        "overrides": {"retriever": "hybrid"},
        "why": ("Change vs V0: **hybrid retrieval** = BM25 + dense, fused with Reciprocal Rank "
                "Fusion (k=60, 20 candidates each). Why: V0 misses 9.6% of gold paragraphs in "
                "the top-4, and ViQuAD questions are full of names, dates and numbers that "
                "exact keyword matching (BM25) catches better than embeddings. Hybrid "
                "sparse+dense retrieval beats dense alone in Ma et al. 2021 (DPR replication) "
                "and RRF is a simple, tuning-free fusion (Cormack et al. 2009)."),
    },
    # V2/V3 build on the best version so far (set by the scheduled runner after each result)
    "v2_rerank": {
        "overrides": {"retriever": "hybrid", "reranker": "BAAI/bge-reranker-v2-m3"},
        "why": ("Change vs best so far: **cross-encoder reranker** `BAAI/bge-reranker-v2-m3` "
                "re-scores the top-20 candidates and keeps the top-4. Why: in V0, Hit@1 is "
                "75.9% but Hit@10 is 95.3%, so the gold paragraph is usually retrieved but "
                "ranked too low. A cross-encoder reads question and passage together and ranks "
                "much more precisely than a bi-encoder (Nogueira & Cho 2019, monoBERT)."),
    },
    "v3_qwen3": {
        "overrides": {"retriever": "hybrid", "reranker": "BAAI/bge-reranker-v2-m3",
                      "llm_model": "qwen3:8b"},
        "why": ("Change vs best so far: **answer LLM** Qwen2.5 7B -> **Qwen3 8B** "
                "(thinking off, Q4_K_M ~5.2 GB, still fits a T4). The judges stay Gemma 2 9B + "
                "Qwen2.5 7B so scores stay comparable. Why: once retrieval is fixed, the remaining errors are "
                "reading errors (right paragraph, wrong span); Qwen3 reports better multilingual "
                "reading comprehension than Qwen2.5 at the same size (Qwen3 Technical Report, "
                "2025)."),
    },
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


def build(version):
    spec = VERSIONS[version]
    cells = [
        md(f"""
# RAG chatbot on Vietnamese Wikipedia (UIT-ViQuAD 2.0) — `{version}`

**Stack:** LangChain + FAISS + `BAAI/bge-m3` embeddings + **Qwen2.5 7B** served by **Ollama**.

**This version:** {spec['why']}

Overrides vs `src/config.py` defaults: `{json.dumps(spec['overrides'])}`

**What this notebook does**
1. Installs Ollama and pulls the LLM on the Kaggle T4 GPU.
2. Builds a corpus from the 3,171 unique paragraphs of the ViQuAD 2.0 dev set, splits them into chunks and indexes them.
3. Measures retrieval (Hit@k, MRR) on **all 3,852** answerable dev questions.
4. Answers **200 questions** (the original 50 with seed 42, plus 150 more with seed 43) **with RAG** and **without RAG**.
5. Scores answers with EM, F1, Contains and two LLM judges from different families: **Gemma 2 9B** (main: correctness 1-5, faithfulness 1-5) and Qwen2.5 7B (second, to measure judge agreement with Cohen's kappa).

Settings: Kaggle GPU **T4**, Internet **on**, dataset `gamind/uit-viquad-2-0` attached.
"""),
        code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q langchain-core langchain-community langchain-text-splitters langchain-huggingface langchain-ollama faiss-cpu rank_bm25 2>&1 | grep -v -i warning | tail -2
"""),
        md("## 1. Install Ollama"),
        code("""
!apt-get update -qq > /dev/null && apt-get install -y -qq zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh > /tmp/ollama_install.log 2>&1; tail -2 /tmp/ollama_install.log
!ollama --version
"""),
        md("## 2. Project code (`src/`)\nThese cells are generated from the repo's `src/` folder."),
        code("import os\nos.makedirs('src', exist_ok=True)"),
    ]
    for name in SRC:
        body = "# package\n" if name == "__init__.py" else (ROOT / "src" / name).read_text()
        cells.append(code(f"%%writefile src/{name}\n" + body))

    cells += [
        md("## 3. Config, start Ollama, pull the LLM"),
        code(f"""
import time, json
from src.config import Config
from src.ollama_utils import start_server, pull, model_info

cfg = Config(version="{version}", **{spec['overrides']!r})
OUT = os.path.join(cfg.out_dir, cfg.version); os.makedirs(OUT, exist_ok=True)
print(cfg)
server = start_server(os.path.join(OUT, "ollama.log"))
for m in dict.fromkeys([cfg.llm_model, cfg.judge_model, cfg.second_judge]):  # judges fixed across versions
    t = time.time(); pull(m)
    print(f"pulled {{m}} in {{time.time() - t:.0f}}s:", model_info(m))
"""),
        md("## 4. Data: corpus, chunks and the 50 test questions"),
        code("""
import pandas as pd
from src.evaluate import prepare

docs, chunks, all_q, test_q, stats = prepare(cfg)
print(stats)
json.dump(test_q, open(f"{OUT}/questions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("example chunk:\\n", chunks[0].page_content[:300])
"""),
        md("## 5. Embeddings (bge-m3) + FAISS index + retriever"),
        code("""
from src.rag import get_embeddings, build_index
from src.retrieval import Retriever

t = time.time()
emb = get_embeddings(cfg.embed_model)
index = build_index(chunks, emb)
retriever = Retriever(index, emb, chunks, cfg.retriever, cfg.candidates, cfg.rrf_k, cfg.reranker)
stats["index_sec"] = round(time.time() - t, 1)
print(f"indexed {index.index.ntotal} chunks in {stats['index_sec']}s, retriever={cfg.retriever}")

# export the index for serving (upload index.tar.gz to a GitHub Release, see README)
from src.index_store import save
save(index, cfg, "index", {"built_by": f"kaggle {cfg.version}"})
!tar -czf {OUT}/index.tar.gz index && rm -rf index && du -h {OUT}/index.tar.gz
"""),
        md("## 6. Retrieval quality on all answerable dev questions\n"
           "A hit means a retrieved chunk comes from the paragraph that holds the gold answer."),
        code("""
from src.evaluate import retrieval_eval

curve, mrr, sec = retrieval_eval(retriever, chunks, all_q, cfg.max_k_curve)
curve.to_csv(f"{OUT}/retrieval_curve.csv", index=False)
print(f"{len(all_q)} questions in {sec:.0f}s", {k: round(v, 4) for k, v in mrr.items()})
curve.round(4)
"""),
        md("## 7. Try the chatbot"),
        code("""
from src.rag import RAGChatbot, get_llm

bot = RAGChatbot(retriever, get_llm(cfg), k=cfg.top_k)
for q in ["Thủ đô của Nhật Bản là thành phố nào?",
          "Bức tường Berlin được xây dựng vào năm nào?",
          test_q[0]["question"]]:
    out = bot.ask(q)
    print("Q:", q)
    print("A:", out["answer"])
    print("sources:", [d.metadata["title"] for d in out["docs"]], "\\n")
"""),
        md("## 8. Evaluate the test questions: RAG vs no RAG"),
        code("""
from src.evaluate import answer_eval, make_judges, summarize_all

judges = make_judges(cfg)  # main: gemma2 (other family), second: qwen2.5 (agreement check)
print("judges:", list(judges))
t = time.time()
df = answer_eval(cfg, bot, judges, test_q)
print(f"done in {time.time() - t:.0f}s")
df.to_csv(f"{OUT}/predictions.csv", index=False)
df[["question", "gold", "rag_answer", "norag_answer", "hit@4", "rag_judge", "norag_judge"]].head(15)
"""),
        md("## 9. Results"),
        code("""
from src.plots import plot_rag_vs_norag, plot_hit_curve

summary = {"config": vars(cfg), "data": stats,
           "retrieval_all_dev": {**{f"Hit@{r.k}": r.hit for r in curve.itertuples()}, **mrr,
                                 "n": len(all_q), "sec": round(sec, 1)},
           **summarize_all(df, cfg.n_core)}
json.dump(summary, open(f"{OUT}/metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
plot_rag_vs_norag(summary, OUT); plot_hit_curve(curve, len(all_q), OUT)

for key in [k for k in summary if k.startswith("test")]:
    t = summary[key]
    print(f"{key}: Hit@4={t['hit@4']:.3f}  MRR@4={t['MRR@4']:.3f}")
    display(pd.DataFrame({m: t[m] for m in ("rag", "norag")}).round(3))
"""),
        md("## 10. Save environment info and print results to the log"),
        code("""
import subprocess, sys, importlib.metadata as md_
pk = ["langchain-core", "langchain-community", "langchain-huggingface", "langchain-ollama",
      "faiss-cpu", "rank-bm25", "sentence-transformers", "torch", "transformers"]
env = {p: md_.version(p) for p in pk}
env["python"] = sys.version.split()[0]
env["ollama"] = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
json.dump(env, open(f"{OUT}/environment.json", "w"), indent=2)
print("ENV_JSON", json.dumps(env))
print("METRICS_JSON", json.dumps(summary, ensure_ascii=False))
print("PREDICTIONS_JSON", df.to_json(orient="records", force_ascii=False))
"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for c in nb["cells"]:  # store source as a list of lines (standard .ipynb layout)
        lines = c["source"].split("\n")
        c["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    out = ROOT / "notebooks" / f"{version}_kaggle.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    for v in sys.argv[1:] or VERSIONS:
        build(v)

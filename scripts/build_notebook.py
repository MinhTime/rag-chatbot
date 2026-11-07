"""Build notebooks/rag_viquad_qwen_kaggle.ipynb from src/*.py.

The notebook writes src/ with %%writefile, so Kaggle runs exactly the code in this repo.
Run:  python scripts/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ["__init__.py", "config.py", "data.py", "rag.py", "metrics.py", "judge.py",
       "ollama_utils.py", "evaluate.py", "plots.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


cells = [
    md("""
# RAG chatbot on Vietnamese Wikipedia (UIT-ViQuAD 2.0)

**Stack:** LangChain + FAISS + `BAAI/bge-m3` embeddings + **Qwen2.5 7B** served by **Ollama**.

**What this notebook does**
1. Installs Ollama and pulls `qwen2.5:7b` (Q4_K_M, ~4.7 GB) on the Kaggle T4 GPU.
2. Builds a corpus from the 3,171 unique paragraphs of the ViQuAD 2.0 dev set, splits them into chunks and indexes them in FAISS.
3. Measures retrieval (Hit@k, MRR) on **all** answerable dev questions.
4. Answers **50 random questions** (seed 42) twice: **with RAG** and **without RAG** (same LLM, same prompt style, only the context changes).
5. Scores answers with EM, F1, Contains and an LLM judge (correctness 1-5, faithfulness 1-5).

Settings: Kaggle GPU **T4**, Internet **on**, dataset `gamind/uit-viquad-2-0` attached.
"""),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q langchain-core langchain-community langchain-text-splitters langchain-huggingface langchain-ollama faiss-cpu 2>&1 | grep -v -i warning | tail -2
"""),
    md("## 1. Install Ollama"),
    code("""
!apt-get update -qq > /dev/null && apt-get install -y -qq zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh > /tmp/ollama_install.log 2>&1; tail -2 /tmp/ollama_install.log
!ollama --version
"""),
    md("## 2. Project code (`src/`)\nThese cells are generated from the repo's `src/` folder."),
]
for name in SRC:
    cells.append(code(f"%%writefile src/{name}\n" + (ROOT / "src" / name).read_text()
                      if name != "__init__.py" else "%%writefile src/__init__.py\n# package\n"))
cells.insert(len(cells) - len(SRC), code("import os\nos.makedirs('src', exist_ok=True)\nos.makedirs('results', exist_ok=True)"))

cells += [
    md("## 3. Start Ollama and pull Qwen2.5 7B"),
    code("""
import time
from src.config import Config
from src.ollama_utils import start_server, pull, model_info

cfg = Config()
server = start_server("results/ollama.log")
t = time.time(); pull(cfg.llm_model)
print(f"pulled {cfg.llm_model} in {time.time() - t:.0f}s:", model_info(cfg.llm_model))
"""),
    md("## 4. Data: corpus, chunks and the 50 test questions"),
    code("""
import json, pandas as pd
from src.evaluate import prepare

docs, chunks, all_q, test_q, stats = prepare(cfg)
print(stats)
json.dump(test_q, open("results/questions_50.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("example chunk:\\n", chunks[0].page_content[:300])
pd.DataFrame(test_q)[["title", "question", "answers"]].head(10)
"""),
    md("## 5. Embeddings (bge-m3) + FAISS index"),
    code("""
from src.rag import get_embeddings, build_index

t = time.time()
emb = get_embeddings(cfg.embed_model)
index = build_index(chunks, emb)
stats["index_sec"] = round(time.time() - t, 1)
print(f"indexed {index.index.ntotal} chunks in {stats['index_sec']}s")
"""),
    md("## 6. Retrieval quality on all answerable dev questions\n"
       "A hit means a retrieved chunk comes from the paragraph that holds the gold answer."),
    code("""
from src.evaluate import retrieval_eval

curve, mrr, sec = retrieval_eval(index, emb, all_q, cfg.max_k_curve)
curve.to_csv("results/retrieval_curve.csv", index=False)
print(f"{len(all_q)} questions in {sec:.0f}s", {k: round(v, 4) for k, v in mrr.items()})
curve.round(4)
"""),
    md("## 7. Try the chatbot"),
    code("""
from src.rag import RAGChatbot, get_llm

bot = RAGChatbot(index, get_llm(cfg), k=cfg.top_k)
for q in ["Thủ đô của Nhật Bản là thành phố nào?",
          "Bức tường Berlin được xây dựng vào năm nào?",
          test_q[0]["question"]]:
    out = bot.ask(q)
    print("Q:", q)
    print("A:", out["answer"])
    print("sources:", [d.metadata["title"] for d in out["docs"]], "\\n")
"""),
    md("## 8. Evaluate 50 questions: RAG vs no RAG"),
    code("""
from src.judge import Judge
from src.evaluate import answer_eval, summarize

judge = Judge(get_llm(cfg, model=cfg.judge_model, json_mode=True))
t = time.time()
df = answer_eval(cfg, bot, judge, test_q)
print(f"done in {time.time() - t:.0f}s")
df.to_csv("results/predictions.csv", index=False)
df[["question", "gold", "rag_answer", "norag_answer", "hit@4", "rag_judge", "norag_judge"]].head(15)
"""),
    md("## 9. Results"),
    code("""
from src.plots import plot_rag_vs_norag, plot_hit_curve

summary = {"config": vars(cfg), "data": stats,
           "retrieval_all_dev": {**{f"Hit@{r.k}": r.hit for r in curve.itertuples()}, **mrr,
                                 "n": len(all_q), "sec": round(sec, 1)},
           "test50": summarize(df)}
json.dump(summary, open("results/metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
plot_rag_vs_norag(summary, "results"); plot_hit_curve(curve, len(all_q), "results")

t50 = summary["test50"]
table = pd.DataFrame({m: t50[m] for m in ("rag", "norag")}).round(3)
print(f"Retrieval on 50: Hit@4={t50['hit@4']:.2f}  MRR@4={t50['MRR@4']:.3f}")
table
"""),
    code("""
from IPython.display import Image, display
display(Image("results/rag_vs_norag.png")); display(Image("results/retrieval_hit_at_k.png"))
"""),
    md("## 10. Save environment info and print results to the log"),
    code("""
import subprocess, sys, importlib.metadata as md_
pk = ["langchain-core", "langchain-community", "langchain-huggingface", "langchain-ollama",
      "faiss-cpu", "sentence-transformers", "torch", "transformers"]
env = {p: md_.version(p) for p in pk}
env["python"] = sys.version.split()[0]
env["ollama"] = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
json.dump(env, open("results/environment.json", "w"), indent=2)
print("ENV_JSON", json.dumps(env))
print("METRICS_JSON", json.dumps(summary, ensure_ascii=False))
print("PREDICTIONS_JSON", df.to_json(orient="records", force_ascii=False))
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:  # store source as list of lines (standard .ipynb layout)
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
out = ROOT / "notebooks" / "rag_viquad_qwen_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out)

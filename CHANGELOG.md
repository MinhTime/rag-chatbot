# Changelog

Every experiment version changes **one factor** versus the best version before it, says **why** (previous result + idea + paper), and is evaluated the same way (Evaluation v2): retrieval on all 3,852 answerable ViQuAD 2.0 dev questions; answers on the same 200 questions, judged by Gemma 2 9B (+ Qwen2.5 7B as a second judge); gate with a paired bootstrap.

## v0.3.0 (2026-11-08): best experiment in production

- Production defaults = v3_qwen3: hybrid retrieval (BM25 + FAISS, RRF) + `bge-reranker-v2-m3` + **Qwen3 8B** (thinking off).
- Helm chart and image version 0.3.0; docker-compose pulls `qwen3:8b`.
- Judge is fault-tolerant (JSON → plain retry, failures counted as `judge_missing`); notebooks now ship every `src/` module they import (regression test).

## Evaluation v2 (2026-11-07): more questions, independent judge, statistics

The 50-question test could not separate small differences (one question = 2 points; v1 looked worse on F1 while its retrieval was clearly better). From now on, all versions are re-run with the same harness:
- **200 test questions**: the original 50 (seed 42) plus 150 more (seed 43). Results are reported on both `test50` and `test200`.
- **Independent judge**: the main judge is **Gemma 2 9B** (Google), a different family from the answer model, which avoids self-preference bias. Qwen2.5 7B stays as a second judge, and **Cohen's kappa** between the two judges is reported to show how reliable the judge scores are.
- **Eval gate with statistics**: a paired bootstrap (2,000 resamples) gives a 95% CI for ΔF1. A version fails only when F1 drops more than 2 points **and** the CI is entirely below 0.
- Judge prompts use a single user message, so every model family reads them the same way.

## v0.2.0 (2026-11-07): production stack + experiment pipeline

- **Serving**: FastAPI API (`/ask`, `/healthz`, `/readyz`, `/metrics`), Dockerfile, docker-compose (Ollama on GPU + API).
- **Kubernetes**: Helm chart with an Ollama GPU deployment, the API, an initContainer that downloads the versioned FAISS index, an optional Ingress or Cloudflare Tunnel, and an optional HPA.
- **CI/CD**: GitHub Actions runs lint + tests + `helm lint` on every push and builds and pushes the image to GHCR on `v*` tags. A manual `eval` workflow runs a version on Kaggle and fails if it is worse than the baseline (eval gate).
- **Experiments**: one notebook per version (`notebooks/<version>_kaggle.ipynb`), results per version (`results/<version>/`), `scripts/kaggle_eval.py` to push, collect and gate.
- Retrieval code moved to `src/retrieval.py` (dense, hybrid BM25 + dense with RRF, optional cross-encoder reranker).
- Unit tests (`tests/`), ruff config, Makefile.

## Experiments

### v1_hybrid: hybrid retrieval (BM25 + dense, RRF)
- **Change**: retriever `dense` → `hybrid` (20 candidates from each, Reciprocal Rank Fusion with k=60).
- **Why**: V0 misses the gold paragraph in the top-4 for 9.6% of questions, and when it misses, F1 drops to 7%. ViQuAD questions contain many names, dates and numbers that exact keyword matching catches better than embeddings. Sparse + dense hybrids beat dense alone (Ma et al. 2021, *A Replication Study of Dense Passage Retriever*), and RRF needs no tuning (Cormack et al. 2009).
- **Result** (2026-11-07): retrieval improves on all 3,852 questions: Hit@1 75.9 → **79.9**, Hit@4 90.4 → **93.1**, Hit@10 95.3 → **96.9**, MRR@10 0.828 → **0.862**. On the 50 test questions, answer F1 falls 69.3 → 63.4 and EM 48 → 40 (judge ≥ 4: 74 → 70). Per question: 3 questions lost the gold paragraph from the top-4 and 1 gained it; the remaining differences come from the LLM reacting to a different context order (e.g. longer or shorter lists). **Eval gate (50 questions): FAIL** at first (F1 −5.9 > 3-point tolerance). With the paired bootstrap the ΔF1 95% CI is −14.7 .. +1.7 points, so the drop is **not significant**. This led to Evaluation v2 above, and v1 is being re-run on 200 questions.
- **Re-run with Evaluation v2** (2026-11-08, 200 questions, Gemma judge): v0 EM 39.0 / F1 65.7 / judge ≥ 4 75.5%; v1 EM 38.0 / F1 63.6 / judge ≥ 4 74.0%. Paired bootstrap ΔF1 = −2.1 points, 95% CI −6.0 .. +1.8 → no significant difference. Judge agreement Gemma vs Qwen: kappa 0.85 (v0) and 0.82 (v1). **Gate: PASS** → v1 becomes the best version and the production default (`RETRIEVER=hybrid`).
- **Decision**: keep hybrid as the first stage for v2, because a reranker re-orders candidates and hybrid gives it the best candidate set (Hit@10 96.9% vs 95.3%). The serving default stays `hybrid` only if v2 passes the gate.

### v2_rerank: cross-encoder reranker
- **Change**: re-score the top-20 hybrid candidates with `BAAI/bge-reranker-v2-m3` and keep the top-4.
- **Why**: in both v0 and v1 the gold paragraph is usually retrieved but ranked too low (v1: Hit@1 79.9% vs Hit@10 96.9%). A cross-encoder reads the question and the passage together and ranks much more precisely than bi-encoder or BM25 scores (Nogueira & Cho 2019, *Passage Re-ranking with BERT*).
- **Result** (2026-11-08, Evaluation v2): retrieval jumps on 3,852 questions: Hit@1 79.9 → **90.2**, Hit@4 93.1 → **97.1**, Hit@10 96.9 → **98.2**, MRR@10 0.862 → **0.934**. Answers (200 questions): EM 39.0, F1 64.7, judge ≥ 4 **76.5%** (best so far), faithfulness 4.43/5; judge kappa 0.84. Gate vs v1: ΔF1 +1.2, 95% CI −3.7 .. +5.9 → **PASS** (also passes vs v0: ΔF1 −0.9, CI −4.9 .. +3.4). Retrieval-only eval cost: 549 s for 3,852 questions on a T4 (~0.14 s per question).
- **Insight**: the gold paragraph is now in the top-4 for 196/200 questions, but F1 on those is 65.6%. Retrieval is no longer the bottleneck; the reader LLM is. This motivates v3.
- **Decision**: v2 is the new best and production default (`RETRIEVER=hybrid`, `RERANKER=BAAI/bge-reranker-v2-m3`).

### v3_qwen3: stronger reader LLM
- **Change**: answer LLM Qwen2.5 7B → **Qwen3 8B** (thinking off). Retrieval = v2. Judges unchanged.
- **Why**: after v2, 98% of questions have the gold paragraph in the context, yet F1 is ~65%: the errors are reading errors (partial spans, paraphrases). A newer model of the same size with better multilingual reading comprehension should read the same context better (Qwen3 Technical Report, 2025).
- **Result** (2026-11-08, Evaluation v2, 200 questions): EM 39.0 → 39.5, F1 64.7 → **68.5**, judge ≥ 4 76.5% → **82.5%**, judge mean 4.02 → 4.28, faithfulness 4.43 → **4.57**. Second judge (Qwen2.5) also sees the gain: 72.5% → 80.5%; kappa 0.83. Answers "Không biết" with the answer in context: 8 → 0. On the gold-in-context questions F1 rises 65.6 → 69.4. Latency 0.73 → 1.05 s per answer on a T4. Gate vs v2: ΔF1 +3.7, 95% CI −0.7 .. +8.1 → **PASS**.
- **Decision**: v3 is the best version and the production default (`LLM_MODEL=qwen3:8b`), released as **v0.3.0**.

### v0_dense: baseline (2026-11-06)
- bge-m3 + FAISS (top-4) + Qwen2.5 7B via Ollama, 500/100 chunks with the title prepended.
- Evaluation v2 (200 questions, Gemma judge): EM 39.0%, F1 65.7%, judge ≥ 4 75.5%, faithfulness 4.34/5; no RAG: EM 3.0%, F1 12.6%, judge ≥ 4 17.0%.
- Retrieval (3,852 questions): Hit@1 75.9%, Hit@4 90.4%, Hit@10 95.3%, MRR@10 0.828.
- First run, 50 questions, Qwen judge, RAG vs no RAG: EM 48% vs 4%, F1 69.3% vs 14.9%, judge ≥ 4: 74% vs 18%.

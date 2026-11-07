"""Full pipeline: corpus -> FAISS -> retrieval eval -> RAG vs no-RAG answers -> scores.

Usage (needs a running Ollama server with the model pulled):
    python -m src.evaluate --viquad_dev path/to/dev.json --out_dir results
"""
import argparse
import json
import os
import time

import pandas as pd
from tqdm.auto import tqdm

from .config import Config
from .data import answerable_questions, build_corpus, load_viquad, sample_questions, split_docs
from .judge import Judge
from .metrics import contains, exact_match, f1, hit_at_k, reciprocal_rank
from .rag import (RAGChatbot, batch_retrieve_ctx_ids, build_index, format_context,
                  get_embeddings, get_llm)


def prepare(cfg):
    records = load_viquad(cfg.viquad_dev)
    docs, ctx_ids = build_corpus(records)
    all_q = answerable_questions(records, ctx_ids)
    test_q = sample_questions(all_q, cfg.n_questions, cfg.seed)
    chunks = split_docs(docs, cfg.chunk_size, cfg.chunk_overlap, cfg.add_title)
    stats = {"paragraphs": len(docs), "chunks": len(chunks),
             "answerable_questions": len(all_q), "test_questions": len(test_q)}
    return docs, chunks, all_q, test_q, stats


def retrieval_eval(index, emb, questions, max_k):
    """Hit@k and MRR on ALL answerable dev questions (cheap: no LLM)."""
    t = time.time()
    got = batch_retrieve_ctx_ids(index, emb, [q["question"] for q in questions], max_k)
    curve = [{"k": k, "hit": sum(hit_at_k(r, q["ctx_id"], k) for r, q in zip(got, questions))
              / len(questions)} for k in range(1, max_k + 1)]
    mrr = {f"MRR@{k}": sum(reciprocal_rank(r, q["ctx_id"], k) for r, q in zip(got, questions))
           / len(questions) for k in (4, max_k)}
    return pd.DataFrame(curve), mrr, time.time() - t


def answer_eval(cfg, bot, judge, test_q):
    rows = []
    for q in tqdm(test_q, desc="answering"):
        docs = bot.retrieve(q["question"])
        ctx = format_context(docs)
        retrieved = [d.metadata["ctx_id"] for d in docs]
        t0 = time.time()
        rag = bot.rag_chain.invoke({"context": ctx, "question": q["question"]}).strip()
        t1 = time.time()
        plain = bot.plain_chain.invoke({"question": q["question"]}).strip()
        t2 = time.time()
        rows.append({
            "id": q["id"], "title": q["title"], "question": q["question"],
            "gold": " | ".join(q["answers"]),
            "hit@4": hit_at_k(retrieved, q["ctx_id"], cfg.top_k),
            "rr@4": reciprocal_rank(retrieved, q["ctx_id"], cfg.top_k),
            "rag_answer": rag, "norag_answer": plain,
            "rag_em": exact_match(rag, q["answers"]), "norag_em": exact_match(plain, q["answers"]),
            "rag_f1": f1(rag, q["answers"]), "norag_f1": f1(plain, q["answers"]),
            "rag_contains": contains(rag, q["answers"]),
            "norag_contains": contains(plain, q["answers"]),
            "rag_judge": judge.correctness(q["question"], q["answers"], rag),
            "norag_judge": judge.correctness(q["question"], q["answers"], plain),
            "rag_faithfulness": judge.faithfulness(q["question"], ctx, rag),
            "rag_sec": round(t1 - t0, 2), "norag_sec": round(t2 - t1, 2),
            "sources": "; ".join(f"{d.metadata['title']}#{d.metadata['ctx_id']}" for d in docs),
        })
    return pd.DataFrame(rows)


def summarize(df):
    s = {"hit@4": df["hit@4"].mean(), "MRR@4": df["rr@4"].mean()}
    for mode in ("rag", "norag"):
        s[mode] = {"EM": df[f"{mode}_em"].mean(), "F1": df[f"{mode}_f1"].mean(),
                   "Contains": df[f"{mode}_contains"].mean(),
                   "Judge(1-5)": df[f"{mode}_judge"].mean(),
                   "Judge>=4": (df[f"{mode}_judge"] >= 4).mean(),
                   "sec/answer": df[f"{mode}_sec"].mean()}
    s["rag"]["Faithfulness(1-5)"] = df["rag_faithfulness"].mean()
    # RAG answers split by whether retrieval found the gold paragraph
    for hit, name in ((1.0, "rag_when_hit"), (0.0, "rag_when_miss")):
        sub = df[df["hit@4"] == hit]
        s[name] = {"n": len(sub), "F1": sub["rag_f1"].mean() if len(sub) else None,
                   "Judge(1-5)": sub["rag_judge"].mean() if len(sub) else None}
    return json.loads(json.dumps(s, default=float))


def run(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    docs, chunks, all_q, test_q, stats = prepare(cfg)
    print("data:", stats)
    with open(os.path.join(cfg.out_dir, "questions_50.json"), "w", encoding="utf-8") as f:
        json.dump(test_q, f, ensure_ascii=False, indent=1)

    t = time.time()
    emb = get_embeddings(cfg.embed_model)
    index = build_index(chunks, emb)
    stats["index_sec"] = round(time.time() - t, 1)

    curve, mrr, sec = retrieval_eval(index, emb, all_q, cfg.max_k_curve)
    curve.to_csv(os.path.join(cfg.out_dir, "retrieval_curve.csv"), index=False)

    bot = RAGChatbot(index, get_llm(cfg), k=cfg.top_k)
    judge = Judge(get_llm(cfg, model=cfg.judge_model, json_mode=True))
    df = answer_eval(cfg, bot, judge, test_q)
    df.to_csv(os.path.join(cfg.out_dir, "predictions.csv"), index=False)

    summary = {"config": vars(cfg), "data": stats,
               "retrieval_all_dev": {**{f"Hit@{r.k}": r.hit for r in curve.itertuples()}, **mrr,
                                     "n": len(all_q), "sec": round(sec, 1)},
               "test50": summarize(df)}
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return index, emb, bot, df, curve, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in vars(Config()).items():
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else lambda x: x == "1",
                        default=v)
    run(Config(**vars(ap.parse_args())))

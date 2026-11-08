"""Figures for the README."""
import os

import matplotlib.pyplot as plt

BLUE, GREY = "#2a6fdb", "#9aa5b1"


def plot_rag_vs_norag(summary, out_dir):
    key = max((k for k in summary if k.startswith("test")), key=lambda k: int(k[4:]))
    s = summary[key]
    names = ["EM", "F1", "Contains", "Judge>=4"]
    rag = [s["rag"][n] * 100 for n in names]
    plain = [s["norag"][n] * 100 for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    c = summary.get("config", {})
    llm = c.get("llm_model", "LLM")
    retr = c.get("retriever", "dense") + (" + rerank" if c.get("reranker") else "")
    b1 = ax.bar([i - 0.2 for i in x], plain, 0.4, label=f"No RAG ({llm} only)", color=GREY)
    b2 = ax.bar([i + 0.2 for i in x], rag, 0.4, label=f"RAG ({retr}, top-{c.get('top_k', 4)})",
                color=BLUE)
    ax.bar_label(b1, fmt="%.0f", fontsize=9)
    ax.bar_label(b2, fmt="%.0f", fontsize=9)
    ax.set_xticks(list(x), names)
    ax.set_ylabel(f"% of {key[4:]} questions")
    ax.set_ylim(0, 105)
    ax.set_title(f"Answer quality on {key[4:]} UIT-ViQuAD 2.0 questions")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "rag_vs_norag.png")
    fig.savefig(path, dpi=150)
    return path


def plot_hit_curve(curve, n, out_dir):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(curve["k"], curve["hit"] * 100, marker="o", color=BLUE)
    for k, h in zip(curve["k"], curve["hit"]):
        if k in (1, 4, curve["k"].max()):
            ax.annotate(f"{h * 100:.1f}", (k, h * 100), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=9)
    ax.set_xlabel("k (retrieved chunks)")
    ax.set_ylabel("Hit@k (%)")
    ax.set_title(f"Retrieval: gold paragraph in top-k ({n} dev questions)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "retrieval_hit_at_k.png")
    fig.savefig(path, dpi=150)
    return path

"""Load UIT-ViQuAD 2.0, build the document corpus and the 50-question test set."""
import glob
import json
import os
import random

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def find_dev_json(path):
    """Use `path` if it exists, else search /kaggle/input (mount paths differ between images)."""
    if os.path.exists(path):
        return path
    hits = sorted(glob.glob("/kaggle/input/**/dev.json", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{path} not found and no dev.json under /kaggle/input")
    return hits[0]


def load_viquad(path):
    """dev.json is a flat list of {id, title, context, question, answers{text, answer_start}}."""
    path = find_dev_json(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


def build_corpus(records):
    """One Document per unique paragraph. Returns (docs, context -> ctx_id map)."""
    ctx_ids, docs = {}, []
    for r in records:
        if r["context"] not in ctx_ids:
            ctx_ids[r["context"]] = len(docs)
            docs.append(Document(page_content=r["context"],
                                 metadata={"ctx_id": len(docs), "title": r["title"]}))
    return docs, ctx_ids


def answerable_questions(records, ctx_ids):
    """Questions that have a gold answer (ViQuAD 2.0 also has unanswerable ones)."""
    out = []
    for r in records:
        if r["answers"]["text"]:
            out.append({"id": r["id"], "title": r["title"], "question": r["question"],
                        "answers": sorted(set(r["answers"]["text"])),
                        "ctx_id": ctx_ids[r["context"]]})
    return out


def sample_questions(questions, n, seed):
    qs = sorted(questions, key=lambda q: q["id"])
    return random.Random(seed).sample(qs, n)


def split_docs(docs, chunk_size, chunk_overlap, add_title=True):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""])
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        if add_title:
            c.page_content = f"{c.metadata['title']}: {c.page_content}"
    return chunks

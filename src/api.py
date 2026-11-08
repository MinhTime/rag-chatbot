"""Production API: FastAPI around the RAG chatbot.

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Settings come from environment variables (see Settings). Swagger UI at /docs,
Prometheus metrics at /metrics, health at /healthz (liveness) and /readyz (readiness).
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from .config import Config

log = logging.getLogger("rag-api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

REQS = Counter("rag_requests_total", "Questions answered", ["status"])
LAT = Histogram("rag_latency_seconds", "Latency per stage", ["stage"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
IDK = Counter("rag_dont_know_total", "Answers that were 'Không biết'")


class Settings:
    """All runtime settings, overridable by env vars (12-factor)."""

    def __init__(self):
        d = Config()
        self.index_dir = os.getenv("INDEX_DIR", "artifacts/index")
        self.ollama_url = os.getenv("OLLAMA_URL", d.ollama_url)
        self.llm_model = os.getenv("LLM_MODEL", d.llm_model)
        self.embed_model = os.getenv("EMBED_MODEL", d.embed_model)
        self.device = os.getenv("DEVICE", "cpu")
        self.retriever = os.getenv("RETRIEVER", d.retriever)
        self.reranker = os.getenv("RERANKER", d.reranker)
        self.top_k = int(os.getenv("TOP_K", d.top_k))
        self.candidates = int(os.getenv("CANDIDATES", d.candidates))
        self.version = os.getenv("APP_VERSION", "dev")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["Bức tường Berlin được xây dựng vào năm nào?"])
    top_k: int | None = Field(None, ge=1, le=10)


class Source(BaseModel):
    title: str
    ctx_id: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: dict[str, float]
    version: str


STATE = {}


def build_bot(s: Settings):
    """Heavy imports live here so tests can inject a fake bot without torch/ollama."""
    from .index_store import load
    from .rag import RAGChatbot, get_embeddings, get_llm
    from .retrieval import Retriever

    emb = get_embeddings(s.embed_model, s.device)
    index, chunks, meta = load(s.index_dir, emb)
    if meta.get("embed_model") != s.embed_model:
        raise RuntimeError(f"index built with {meta.get('embed_model')}, not {s.embed_model}")
    retriever = Retriever(index, emb, chunks, s.retriever, s.candidates, reranker=s.reranker,
                          device=s.device)
    cfg = Config(ollama_url=s.ollama_url, llm_model=s.llm_model, top_k=s.top_k)
    log.info("loaded index: %s chunks, retriever=%s reranker=%s llm=%s",
             len(chunks), s.retriever, s.reranker or "-", s.llm_model)
    return RAGChatbot(retriever, get_llm(cfg), k=s.top_k)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = Settings()
    STATE["settings"] = s
    if "bot" not in STATE:  # tests pre-inject a fake bot
        STATE["bot"] = await run_in_threadpool(build_bot, s)
    yield


app = FastAPI(title="RAG chatbot (UIT-ViQuAD)", version=os.getenv("APP_VERSION", "dev"),
              lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if "bot" not in STATE:
        raise HTTPException(503, "loading")
    return {"status": "ready", "version": STATE["settings"].version}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    bot, s = STATE["bot"], STATE["settings"]
    try:
        t0 = time.perf_counter()
        docs = bot.retrieve(req.question, req.top_k or s.top_k)
        t1 = time.perf_counter()
        answer = bot.answer_with(req.question, docs)
        t2 = time.perf_counter()
    except Exception as e:  # LLM down, timeout, ...
        REQS.labels("error").inc()
        log.exception("ask failed")
        raise HTTPException(502, f"backend error: {type(e).__name__}") from e
    LAT.labels("retrieve").observe(t1 - t0)
    LAT.labels("generate").observe(t2 - t1)
    REQS.labels("ok").inc()
    if "không biết" in answer.lower():
        IDK.inc()
    log.info("ask q_len=%d retrieve_ms=%.0f generate_ms=%.0f", len(req.question),
             (t1 - t0) * 1e3, (t2 - t1) * 1e3)
    return AskResponse(
        answer=answer,
        sources=[Source(title=d.metadata["title"], ctx_id=d.metadata["ctx_id"],
                        text=d.page_content) for d in docs],
        latency_ms={"retrieve": round((t1 - t0) * 1e3, 1), "generate": round((t2 - t1) * 1e3, 1)},
        version=s.version)

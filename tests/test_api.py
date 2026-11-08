from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src import api


class FakeBot:
    def retrieve(self, question, k=None):
        return [Document(page_content="Nhật Bản: Tokyo là thủ đô của Nhật Bản.",
                         metadata={"title": "Nhật Bản", "ctx_id": 1, "chunk_id": 7})][:k]

    def answer_with(self, question, docs):
        return "Tokyo"


def client():
    api.STATE["bot"] = FakeBot()
    return TestClient(api.app)


def test_ask_returns_answer_and_sources():
    with client() as c:
        r = c.post("/ask", json={"question": "Thủ đô của Nhật Bản là gì?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Tokyo"
        assert body["sources"][0]["title"] == "Nhật Bản"
        assert set(body["latency_ms"]) == {"retrieve", "generate"}


def test_validation_and_health():
    with client() as c:
        assert c.post("/ask", json={"question": "a"}).status_code == 422
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").status_code == 200
        assert "rag_requests_total" in c.get("/metrics").text

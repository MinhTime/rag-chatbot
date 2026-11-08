# RAG chatbot API. The FAISS index is NOT baked in: it is mounted at INDEX_DIR
# (compose: ./artifacts/index, k8s: downloaded by an initContainer from INDEX_URL).
#   docker build -f docker/Dockerfile -t rag-chatbot:dev .
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models INDEX_DIR=/app/artifacts/index DEVICE=cpu

WORKDIR /app
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

RUN useradd -m -u 10001 app && mkdir -p /models && chown app /models
COPY src/ src/

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=300s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Common tasks. `make help` lists them.
IMAGE ?= ghcr.io/minhtime/rag-chatbot
VERSION ?= dev
DEV_JSON ?= data/dev.json

help:  ## show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## dev dependencies
	pip install -r requirements-dev.txt

lint:  ## ruff
	ruff check .

test:  ## unit tests (no GPU / Ollama needed)
	pytest -q

data:  ## download UIT-ViQuAD 2.0 (needs Kaggle API token)
	kaggle datasets download gamind/uit-viquad-2-0 -p data --unzip

index:  ## build FAISS index into artifacts/index (GPU recommended)
	python -m src.index_store --viquad_dev $(DEV_JSON) --out artifacts/index

eval:  ## full offline evaluation (needs Ollama running)
	python -m src.evaluate --viquad_dev $(DEV_JSON) --version local --retriever hybrid

notebooks:  ## regenerate Kaggle notebooks from src/
	python scripts/build_notebook.py

serve:  ## run the API locally
	uvicorn src.api:app --reload --port 8000

image:  ## build the API image
	docker build -f docker/Dockerfile --build-arg APP_VERSION=$(VERSION) -t $(IMAGE):$(VERSION) .

up:  ## docker compose stack (Ollama GPU + API)
	APP_VERSION=$(VERSION) docker compose up -d --build

deploy:  ## helm upgrade on the current kube context
	helm upgrade --install rag deploy/helm/rag-chatbot -n rag --create-namespace \
	  --set image.repository=$(IMAGE) --set image.tag=$(VERSION)

.PHONY: help install lint test data index eval notebooks serve image up deploy

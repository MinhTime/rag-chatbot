#!/usr/bin/env bash
# Install Ollama (Linux), start the server and pull the model.
set -euo pipefail
MODEL="${1:-qwen2.5:7b}"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  nohup ollama serve > ollama.log 2>&1 &
  for _ in $(seq 60); do curl -sf http://localhost:11434/api/tags >/dev/null && break; sleep 1; done
fi
ollama pull "$MODEL"

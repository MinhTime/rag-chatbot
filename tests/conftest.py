import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# CI has no torch: stub the HuggingFace embeddings import (tests use fake embeddings)
if "langchain_huggingface" not in sys.modules:
    try:
        import langchain_huggingface  # noqa: F401
    except ImportError:
        m = types.ModuleType("langchain_huggingface")
        m.HuggingFaceEmbeddings = object
        sys.modules["langchain_huggingface"] = m

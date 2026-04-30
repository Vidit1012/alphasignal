"""Stub heavy ML/AI packages before any backend module is imported.

pytest loads conftest.py first, so sys.modules entries registered here are
already in place when test_api.py does `from backend.main import app`.
In local dev the real packages take precedence (setdefault is a no-op if
the package is already imported). In CI, where torch / transformers /
llama-index are not installed, the stubs let pytest collect and run all
tests without any GPU dependencies.
"""
import sys
from unittest.mock import MagicMock


# ── torch / transformers ────────────────────────────────────────────────────
for _mod in ["torch", "transformers"]:
    sys.modules.setdefault(_mod, MagicMock())

# ── LangChain core ──────────────────────────────────────────────────────────
# @tool must act as an identity decorator so the decorated functions in
# agent.py remain plain callables after import.
_tool_stub = MagicMock()
_tool_stub.side_effect = lambda f: f

_lc_tools = MagicMock()
_lc_tools.tool = _tool_stub

for _mod, _obj in [
    ("langchain_core", MagicMock()),
    ("langchain_core.messages", MagicMock()),
    ("langchain_core.tools", _lc_tools),
    ("langchain_community", MagicMock()),
]:
    sys.modules.setdefault(_mod, _obj)

# ── langchain-ollama / LangGraph ────────────────────────────────────────────
for _mod in ["langchain_ollama", "langgraph", "langgraph.prebuilt"]:
    sys.modules.setdefault(_mod, MagicMock())

# ── LlamaIndex ──────────────────────────────────────────────────────────────
for _mod in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.schema",
    "llama_index.embeddings",
    "llama_index.embeddings.huggingface",
    "llama_index.vector_stores",
    "llama_index.vector_stores.chroma",
]:
    sys.modules.setdefault(_mod, MagicMock())

# ── ChromaDB / sentence-transformers ────────────────────────────────────────
for _mod in ["chromadb", "sentence_transformers"]:
    sys.modules.setdefault(_mod, MagicMock())

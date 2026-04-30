"""RAG pipeline: chunk and index financial news into ChromaDB via LlamaIndex."""
from __future__ import annotations

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "financial_news"


def _get_chroma_collection(path: str = CHROMA_PATH):
    client = chromadb.PersistentClient(path=path)
    return client, client.get_or_create_collection(COLLECTION_NAME)


def _configure_settings() -> None:
    """Apply embedding model to LlamaIndex global settings; disable built-in LLM."""
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None  # LLM not needed for indexing or retrieval


def index_headlines(headlines: list[str], ticker: str) -> None:
    """Embed *headlines* and upsert them into ChromaDB under *ticker* metadata."""
    if not headlines:
        return
    _configure_settings()
    _, collection = _get_chroma_collection()

    docs = [
        Document(text=h, metadata={"ticker": ticker.upper(), "source": "yfinance"})
        for h in headlines
    ]
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(docs, storage_context=storage_context)


def semantic_search(query: str, top_k: int = 5) -> list[str]:
    """Return the top-*k* news chunks most semantically relevant to *query*."""
    _configure_settings()
    _, collection = _get_chroma_collection()

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [n.get_content() for n in nodes]


if __name__ == "__main__":
    # Run as: python -m backend.rag
    sample_headlines = [
        "NVIDIA beats earnings estimates on surging AI chip demand.",
        "Federal Reserve signals potential rate cuts in Q2 2025.",
        "Apple unveils new Vision Pro features at WWDC.",
        "NVDA data-center revenue grows 400% year-over-year.",
    ]
    index_headlines(sample_headlines, "NVDA")
    results = semantic_search("AI chip revenue growth", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")

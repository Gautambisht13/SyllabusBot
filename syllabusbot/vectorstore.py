"""Embedding factory + persistent ChromaDB handle.

ChromaDB runs embedded (no server): vectors, documents and metadata live in
`storage/chroma/chroma.sqlite3`, so the whole index is a folder you can delete
or copy between machines.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.embeddings import Embeddings

from syllabusbot.config import Settings, get_settings
from syllabusbot.errors import MissingCredentialsError, MissingDependencyError

logger = logging.getLogger(__name__)


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the configured embedding model.

    huggingface: all-MiniLM-L6-v2 — 384 dims, ~90 MB, runs locally, free.
                 Normalised vectors so cosine similarity is well behaved.
    openai:      text-embedding-3-small — 1536 dims, stronger retrieval,
                 costs money and sends document text to the API.
    """
    settings = settings or get_settings()

    if settings.embedding_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings  # lazy import
        except ModuleNotFoundError as exc:
            raise MissingDependencyError(
                "EMBEDDING_PROVIDER=openai needs the OpenAI integration:\n"
                "    pip install langchain-openai"
            ) from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingCredentialsError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY. Add it to "
                ".env, or set EMBEDDING_PROVIDER=huggingface to embed locally."
            )

        logger.info("Embeddings: OpenAI %s", settings.openai_embedding_model)
        return OpenAIEmbeddings(model=settings.openai_embedding_model)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # lazy import
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "EMBEDDING_PROVIDER=huggingface needs the local embedding stack:\n"
            "    pip install langchain-huggingface sentence-transformers\n"
            "(or set EMBEDDING_PROVIDER=openai in .env to embed via the API)"
        ) from exc

    logger.info("Embeddings: HuggingFace %s (local)", settings.hf_embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
        model_kwargs={"device": "cpu"},  # set "cuda" if you have a GPU
        # MiniLM is trained for cosine similarity; normalising makes the
        # cosine distance Chroma stores directly comparable across queries.
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
) -> Any:
    """Open (or create) the persistent Chroma collection.

    The collection name encodes the embedding model (see Settings.collection_name)
    so two providers can coexist on disk without a dimension clash.
    """
    try:
        from langchain_chroma import Chroma  # lazy import (chromadb is heavy)
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "The vector store needs Chroma:\n    pip install langchain-chroma chromadb"
        ) from exc

    settings = settings or get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or build_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        # Default HNSW space is L2; cosine is the right metric for text
        # embeddings and makes relevance scores interpretable (1 - distance).
        # Only applied when the collection is first created.
        collection_metadata={"hnsw:space": "cosine"},
    )


def count_documents(store: Any) -> int:
    """Number of chunks in the collection (0 for a fresh index)."""
    try:
        return int(store._collection.count())  # fast path: direct Chroma count
    except Exception:  # noqa: BLE001 — fall back to the public LangChain API
        try:
            return len(store.get(include=[]).get("ids", []))
        except Exception:  # noqa: BLE001
            return 0


def collection_summary(store: Any) -> dict[str, Any]:
    """Aggregate what is indexed: chunk count, files, doc types, courses.

    Used by `--stats`, the CLI `/stats` command and the Streamlit sidebar.
    """
    summary: dict[str, Any] = {
        "chunks": count_documents(store),
        "files": {},
        "doc_types": {},
        "courses": {},
    }
    if summary["chunks"] == 0:
        return summary

    try:
        records = store.get(include=["metadatas"])
    except Exception:  # noqa: BLE001
        return summary

    for meta in records.get("metadatas") or []:
        if not meta:
            continue
        for key, field in (
            ("files", "source"),
            ("doc_types", "doc_type"),
            ("courses", "course_code"),
        ):
            value = meta.get(field) or ""
            if value:
                summary[key][value] = summary[key].get(value, 0) + 1
    return summary

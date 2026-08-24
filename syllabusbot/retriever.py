"""Retrieval layer: top-k similarity search, course pre-filter, formatting.

Two things happen here beyond a plain `as_retriever()`:

1. Course-aware pre-filtering. "What is the late policy in CS101?" should not
   surface CS102's syllabus just because the wording is near-identical across
   courses — the classic RAG failure on a homogeneous document set. When a
   course code appears in the question we push a metadata filter down into
   Chroma, and fall back to an unfiltered search if that course is not indexed.

2. Context formatting. Each chunk is rendered with an explicit `CITE AS:` line,
   so producing a correct citation is copying, not composing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from syllabusbot.config import Settings, get_settings
from syllabusbot.loaders import iter_course_codes
from syllabusbot.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def detect_course_code(question: str) -> str:
    """Pull a course code out of the question ("...in cs 101?" -> "CS101")."""
    for code in iter_course_codes(question.upper()):
        return code
    return ""


def format_docs(documents: list[Document]) -> str:
    """Render retrieved chunks as the <context> block the prompt consumes."""
    if not documents:
        return "(no documents retrieved)"

    blocks: list[str] = []
    for position, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        header = [
            f"document: {meta.get('source', 'unknown')}",
            f"type: {meta.get('doc_type', 'other')}",
        ]
        if meta.get("course_code"):
            header.append(f"course: {meta['course_code']}")
        if meta.get("page"):
            header.append(f"page: {meta['page']} of {meta.get('total_pages', '?')}")
        if meta.get("section"):
            header.append(f"section: {meta['section']}")
        if meta.get("relevance") is not None:
            header.append(f"relevance: {meta['relevance']:.3f}")

        blocks.append(
            f"--- CHUNK {position} of {len(documents)} ---\n"
            f"CITE AS: {meta.get('citation', '[' + str(meta.get('source', 'unknown')) + ']')}\n"
            f"{' | '.join(header)}\n"
            f"content:\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def build_citations(documents: list[Document]) -> list[dict[str, Any]]:
    """De-duplicated, ordered source list for the UI's "Sources" panel."""
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for doc in documents:
        meta = doc.metadata or {}
        key = (str(meta.get("source", "")), meta.get("page"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "label": meta.get("citation", f"[{meta.get('source', 'unknown')}]"),
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "section": meta.get("section", ""),
                "doc_type": meta.get("doc_type", "other"),
                "course_code": meta.get("course_code", ""),
                "relevance": meta.get("relevance"),
                "excerpt": doc.page_content,
            }
        )
    return citations


@dataclass
class SyllabusRetriever:
    """Thin, explicit wrapper around the Chroma store.

    Kept as a plain object (not a BaseRetriever subclass) so the filter-and-
    fallback logic stays readable; `as_runnable()` drops it into any LCEL chain.
    """

    settings: Settings = field(default_factory=get_settings)
    store: Any = None

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = get_vectorstore(self.settings)

    # -- internals ---------------------------------------------------------
    def _search(self, question: str, k: int, where: dict | None) -> list[Document]:
        """Similarity search that degrades gracefully if scores are unavailable."""
        try:
            scored = self.store.similarity_search_with_relevance_scores(
                question, k=k, filter=where
            )
            documents = []
            for doc, score in scored:
                doc.metadata = {**(doc.metadata or {}), "relevance": round(float(score), 4)}
                documents.append(doc)
            return documents
        except Exception as exc:  # noqa: BLE001 — older chromadb, odd distance fn
            logger.debug("Scored search unavailable (%s); using plain search", exc)
            return self.store.similarity_search(question, k=k, filter=where)

    # -- public API --------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        """Top-k chunks for a question (k defaults to TOP_K = 3)."""
        question = (question or "").strip()
        if not question:
            return []
        k = k or self.settings.top_k

        where: dict | None = None
        course = detect_course_code(question) if self.settings.use_course_filter else ""
        if course:
            where = {"course_code": course}

        documents = self._search(question, k, where)

        # The course was mentioned but isn't in the index (or isn't tagged):
        # retry unfiltered rather than answering "I don't know" on a technicality.
        if course and not documents:
            logger.info("No chunks tagged course_code=%s; retrying unfiltered", course)
            documents = self._search(question, k, None)

        if self.settings.min_relevance > 0:
            before = len(documents)
            documents = [
                doc
                for doc in documents
                if doc.metadata.get("relevance") is None
                or doc.metadata["relevance"] >= self.settings.min_relevance
            ]
            if len(documents) < before:
                logger.info(
                    "Dropped %d chunk(s) below MIN_RELEVANCE=%.2f",
                    before - len(documents),
                    self.settings.min_relevance,
                )
        return documents

    def as_runnable(self) -> Runnable:
        """`str -> list[Document]`, for use inside the LCEL chain."""
        return RunnableLambda(self.retrieve, name="retrieve_top_k")

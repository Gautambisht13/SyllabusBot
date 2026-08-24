"""The RAG pipeline itself: a hand-built LCEL chain plus a small facade.

Chain shape (`str | {"question", "chat_history"}` in, dict out):

    normalize_input
        -> assign documents = itemgetter("question") | retriever   # top-k = 3
        -> assign context   = format_docs(documents)               # CITE AS: blocks
        -> assign answer    = branch:
                                 no documents -> canned refusal (no LLM call)
                                 otherwise    -> RAG_PROMPT | llm | StrOutputParser

Everything is a Runnable, so the whole thing composes, streams, batches, and is
traceable in LangSmith without extra plumbing.

The `SyllabusBot` facade adds the two things a UI needs and a bare chain can't
express cleanly: follow-up condensing (multi-turn chat) and a token-streaming
path that retrieves exactly once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from syllabusbot.config import Settings, get_settings
from syllabusbot.llm import build_llm
from syllabusbot.prompts import (
    CONDENSE_PROMPT,
    INSUFFICIENT_CONTEXT,
    NO_CONTEXT_ANSWER,
    RAG_PROMPT,
)
from syllabusbot.retriever import SyllabusRetriever, build_citations, format_docs
from syllabusbot.vectorstore import collection_summary

logger = logging.getLogger(__name__)

# A turn is ("student question", "bot answer").
History = Sequence[tuple[str, str]]

_HISTORY_HEADER = (
    "Conversation so far (for resolving references only — never a source of facts):\n"
)


def format_history(history: History | None, max_turns: int = 3) -> str:
    """Render the last few turns, or "" when there is nothing to render.

    Bot answers are truncated: the condenser and the reference-resolution step
    only need the gist, and a full previous answer would compete with <context>
    for the model's attention.
    """
    if not history:
        return ""
    lines = [
        f"Student: {question}\nSyllabusBot: {answer[:300]}"
        for question, answer in list(history)[-max_turns:]
        if question
    ]
    return f"{_HISTORY_HEADER}{chr(10).join(lines)}\n\n" if lines else ""


def _normalize_input(payload: Any) -> dict[str, str]:
    """Accept a bare question string or a dict; always emit both keys."""
    if isinstance(payload, str):
        return {"question": payload.strip(), "chat_history": ""}
    if isinstance(payload, dict):
        history = payload.get("chat_history") or ""
        if not isinstance(history, str):  # a list of turns was passed in
            history = format_history(history)
        return {"question": str(payload.get("question", "")).strip(), "chat_history": history}
    raise TypeError(f"Expected str or dict input, got {type(payload).__name__}")


def build_rag_chain(
    settings: Settings | None = None,
    *,
    retriever: SyllabusRetriever | None = None,
    llm: BaseChatModel | None = None,
) -> Runnable:
    """Compose the LCEL retrieval-augmented generation chain."""
    settings = settings or get_settings()
    retriever = retriever or SyllabusRetriever(settings=settings)
    llm = llm or build_llm(settings)

    generate: Runnable = RAG_PROMPT | llm | StrOutputParser()

    # Empty retrieval => refuse locally. Cheaper than a round-trip, and it makes
    # "no context" a property of the pipeline rather than a hope about the model.
    answer_step = RunnableBranch(
        (
            lambda payload: not payload.get("documents"),
            RunnableLambda(lambda _: NO_CONTEXT_ANSWER, name="no_context_refusal"),
        ),
        generate,
    )

    return (
        RunnableLambda(_normalize_input, name="normalize_input")
        | RunnablePassthrough.assign(
            documents=itemgetter("question") | retriever.as_runnable()
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda payload: format_docs(payload["documents"]), name="format_context"
            )
        )
        | RunnablePassthrough.assign(answer=answer_step)
    ).with_config(run_name="syllabusbot_rag")


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass
class Retrieval:
    """Output of the retrieval half — reused by both the sync and stream paths."""

    question: str
    standalone_question: str
    chat_history: str
    documents: list[Document]
    context: str


@dataclass
class Answer:
    question: str
    answer: str
    documents: list[Document] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    standalone_question: str = ""

    @property
    def grounded(self) -> bool:
        """False when the bot declined for lack of context."""
        return bool(self.documents) and INSUFFICIENT_CONTEXT not in self.answer

    def pretty_sources(self) -> str:
        if not self.citations:
            return "(no sources)"
        return "\n".join(
            f"  {index}. {cite['label']}"
            + (f"  ({cite['relevance']:.2f})" if cite.get("relevance") is not None else "")
            for index, cite in enumerate(self.citations, start=1)
        )


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
class SyllabusBot:
    """What the CLI and Streamlit app talk to.

        bot = SyllabusBot()
        print(bot.ask("Late submission policy for CS101?").answer)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: Any = None,
        llm: BaseChatModel | None = None,
        streaming: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = SyllabusRetriever(settings=self.settings, store=store)
        self.llm = llm or build_llm(self.settings, streaming=streaming)
        # Full pipeline (requirement #3) — used by ask() and any batch job.
        self.chain = build_rag_chain(
            self.settings, retriever=self.retriever, llm=self.llm
        )
        # Generation-only tail, used by the streaming path so that prepare()
        # can retrieve once and stream tokens without a second search.
        self.answer_chain: Runnable = RAG_PROMPT | self.llm | StrOutputParser()
        self.condense_chain: Runnable = CONDENSE_PROMPT | self.llm | StrOutputParser()

    # -- multi-turn support ------------------------------------------------
    def condense(self, question: str, history: History | None = None) -> str:
        """Make a follow-up self-contained so vector search can work on it."""
        if not history or not self.settings.rewrite_followups:
            return question
        try:
            rewritten = self.condense_chain.invoke(
                {"question": question, "chat_history": format_history(history)}
            ).strip()
        except Exception as exc:  # noqa: BLE001 — never fail a turn over this
            logger.warning("Follow-up rewrite failed (%s); using original", exc)
            return question
        # Reject a degenerate rewrite (empty, or the model started explaining).
        if not rewritten or len(rewritten) > max(200, len(question) * 4):
            return question
        return rewritten

    # -- retrieval ---------------------------------------------------------
    def prepare(self, question: str, history: History | None = None) -> Retrieval:
        standalone = self.condense(question, history)
        documents = self.retriever.retrieve(standalone)
        return Retrieval(
            question=question,
            standalone_question=standalone,
            chat_history=format_history(history),
            documents=documents,
            context=format_docs(documents),
        )

    # -- generation --------------------------------------------------------
    def ask(self, question: str, history: History | None = None) -> Answer:
        """One-shot answer through the full LCEL chain."""
        standalone = self.condense(question, history)
        result = self.chain.invoke(
            {"question": standalone, "chat_history": format_history(history)}
        )
        documents = result.get("documents", [])
        return Answer(
            question=question,
            answer=result.get("answer", "").strip(),
            documents=documents,
            citations=build_citations(documents),
            standalone_question=standalone,
        )

    def stream_answer(self, retrieval: Retrieval) -> Iterator[str]:
        """Token stream for an already-retrieved question."""
        if not retrieval.documents:
            yield NO_CONTEXT_ANSWER
            return
        yield from self.answer_chain.stream(
            {
                "question": retrieval.standalone_question,
                "context": retrieval.context,
                "chat_history": retrieval.chat_history,
            }
        )

    def stream(self, question: str, history: History | None = None) -> Iterator[str]:
        """Convenience wrapper: retrieve, then stream tokens."""
        yield from self.stream_answer(self.prepare(question, history))

    # -- introspection -----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return collection_summary(self.retriever.store)

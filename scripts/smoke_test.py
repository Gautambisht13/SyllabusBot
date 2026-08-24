"""End-to-end wiring check — no API key, no model download, no network.

    python scripts/smoke_test.py

Runs the real pipeline (pypdf -> RecursiveCharacterTextSplitter -> ChromaDB ->
LCEL chain) against the demo PDFs, substituting two test doubles:

  * HashingEmbeddings — a deterministic hashed bag-of-words embedder. Not
    semantic, but lexical enough to prove retrieval ranks the right chunk.
  * a fake chat model — so the chain, the prompt rendering and the refusal
    branch can be asserted without spending a token.

Exit code 0 = every check passed. Useful as a CI gate.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)

from syllabusbot.chain import SyllabusBot  # noqa: E402
from syllabusbot.config import Settings, enable_utf8_stdout  # noqa: E402
from syllabusbot.ingest import ingest  # noqa: E402
from syllabusbot.loaders import discover_pdfs  # noqa: E402
from syllabusbot.prompts import INSUFFICIENT_CONTEXT, NO_CONTEXT_ANSWER, RAG_PROMPT  # noqa: E402
from syllabusbot.vectorstore import collection_summary, get_vectorstore  # noqa: E402

CANNED_ANSWER = "Late work loses 10% per 24 hours [CS101_Syllabus.pdf, p.2]."
QUESTION = "What is the policy for late assignment submissions in CS101?"

# Every prompt the fake model receives, so we can assert what was actually sent.
SENT_PROMPTS: list[str] = []


class HashingEmbeddings(Embeddings):
    """Deterministic hashed term-frequency vectors (cosine-comparable)."""

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims

    def _vector(self, text: str) -> list[float]:
        counts = [0.0] * self.dims
        for token in re.findall(r"[a-z0-9%]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims
            counts[bucket] += 1.0
        weighted = [math.log1p(value) for value in counts]
        norm = math.sqrt(sum(value * value for value in weighted)) or 1.0
        return [value / norm for value in weighted]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class RecordingFakeChat(GenericFakeChatModel):
    """Fake chat model that remembers the prompts it was handed."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        SENT_PROMPTS.append("\n\n".join(str(message.content) for message in messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class Checks:
    def __init__(self) -> None:
        self.failures = 0

    def ok(self, condition: bool, label: str, detail: str = "") -> None:
        if condition:
            print(f"  PASS  {label}")
        else:
            self.failures += 1
            print(f"  FAIL  {label}{f' -> {detail}' if detail else ''}")


def ensure_sample_pdfs() -> None:
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists() and list(data_dir.rglob("*.pdf")):
        return
    print("No PDFs found — generating the demo set ...")
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import make_sample_pdfs

    if make_sample_pdfs.main() != 0:
        raise SystemExit("could not generate sample PDFs (pip install reportlab)")


def main() -> int:
    enable_utf8_stdout()
    ensure_sample_pdfs()
    checks = Checks()
    chroma_dir = Path(tempfile.mkdtemp(prefix="syllabusbot_smoke_"))

    try:
        settings = Settings(
            data_dir=PROJECT_ROOT / "data",
            chroma_dir=chroma_dir,
            collection_prefix="smoke",
            rewrite_followups=False,  # no history in this test; skip the LLM call
        )
        embeddings: Any = HashingEmbeddings()
        store = get_vectorstore(settings, embeddings=embeddings)

        print("\n[1] Ingestion")
        pdfs = discover_pdfs(settings.data_dir)
        report = ingest(settings, store=store)
        checks.ok(report["chunks"] > 0, "chunks were indexed", str(report))
        checks.ok(report["files"] == len(pdfs), f"all {len(pdfs)} PDFs indexed", str(report))

        summary = collection_summary(store)
        checks.ok(summary["chunks"] == report["chunks"], "collection count matches report")
        checks.ok("CS101" in summary["courses"], "CS101 tagged from filename", str(summary["courses"]))
        checks.ok(
            {"syllabus", "calendar", "handbook"} <= set(summary["doc_types"]),
            "doc types inferred from folders",
            str(summary["doc_types"]),
        )

        print("\n[2] Idempotent re-ingest")
        second = ingest(settings, store=store)
        checks.ok(second["skipped"] == len(pdfs), "unchanged files skipped", str(second))
        checks.ok(second["chunks"] == 0, "no duplicate chunks written", str(second))
        checks.ok(
            collection_summary(store)["chunks"] == summary["chunks"],
            "collection size unchanged after re-run",
        )

        print("\n[3] Retrieval")
        bot = SyllabusBot(
            settings,
            store=store,
            llm=RecordingFakeChat(messages=iter([CANNED_ANSWER] * 10)),
        )
        retrieval = bot.prepare(QUESTION)
        sources = {doc.metadata["source"] for doc in retrieval.documents}
        checks.ok(len(retrieval.documents) == settings.top_k, f"top-{settings.top_k} chunks returned")
        checks.ok(sources == {"CS101_Syllabus.pdf"}, "course filter kept CS101 only", str(sources))
        checks.ok("10%" in retrieval.context, "late-penalty text was retrieved")
        checks.ok("CITE AS:" in retrieval.context, "context carries CITE AS tags")
        checks.ok(
            all(doc.metadata.get("relevance") is not None for doc in retrieval.documents),
            "cosine relevance scores present",
        )
        top = retrieval.documents[0].metadata
        print(f"        top hit: {top['citation']} (relevance {top.get('relevance')})")

        calendar = bot.prepare("When does the add/drop period end?")
        calendar_sources = {doc.metadata["source"] for doc in calendar.documents}
        checks.ok(
            "Academic_Calendar_2026_2027.pdf" in calendar_sources,
            "calendar question hits the calendar",
            str(calendar_sources),
        )

        print("\n[4] Prompt construction")
        rendered = RAG_PROMPT.invoke(
            {"context": retrieval.context, "question": QUESTION, "chat_history": ""}
        ).to_string()
        checks.ok(INSUFFICIENT_CONTEXT in rendered, "refusal sentence is baked into the system prompt")
        checks.ok("CITE AS" in rendered, "citation instruction present")
        checks.ok(QUESTION in rendered, "question reached the prompt")

        print("\n[5] LCEL chain")
        answer = bot.ask(QUESTION)
        checks.ok(answer.answer == CANNED_ANSWER, "chain returned the model's answer", answer.answer)
        checks.ok(len(answer.citations) == len(retrieval.documents), "citations attached")
        checks.ok(answer.grounded, "answer marked grounded")
        checks.ok(bool(SENT_PROMPTS), "model was actually invoked")
        checks.ok(
            SENT_PROMPTS and "CITE AS:" in SENT_PROMPTS[-1] and "10%" in SENT_PROMPTS[-1],
            "model received the retrieved context",
        )
        streamed = "".join(bot.stream(QUESTION))
        checks.ok(streamed == CANNED_ANSWER, "streaming path yields the same answer", streamed)

        print("\n[6] Empty-index refusal (no LLM call)")
        empty_settings = replace(settings, collection_prefix="smoke_empty")
        empty_bot = SyllabusBot(
            empty_settings,
            store=get_vectorstore(empty_settings, embeddings=embeddings),
            llm=RecordingFakeChat(messages=iter([CANNED_ANSWER] * 5)),
        )
        before = len(SENT_PROMPTS)
        refusal = empty_bot.ask("Is there a policy about hoverboards?")
        checks.ok(refusal.answer == NO_CONTEXT_ANSWER, "canned refusal returned", refusal.answer[:60])
        checks.ok(not refusal.grounded, "refusal is not marked grounded")
        checks.ok(len(SENT_PROMPTS) == before, "no tokens spent on an empty retrieval")

    finally:
        shutil.rmtree(chroma_dir, ignore_errors=True)

    print(f"\n{'ALL CHECKS PASSED' if not checks.failures else f'{checks.failures} CHECK(S) FAILED'}")
    return 1 if checks.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

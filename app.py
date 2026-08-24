"""Streamlit UI for SyllabusBot.

    streamlit run app.py

Same pipeline as the CLI — this file only handles presentation, so the two
interfaces can never drift in behaviour.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace

import streamlit as st

from syllabusbot.chain import SyllabusBot
from syllabusbot.config import get_settings
from syllabusbot.errors import SyllabusBotError
from syllabusbot.retriever import build_citations

# --- Auto-generate and ingest sample PDFs if vector store is empty (for cloud deployment) ---
if not os.path.exists("storage/chroma") or not os.listdir("storage/chroma"):
    with st.spinner("Generating sample PDFs and building vector index..."):
        # Run sample script and ingestion as subprocesses to match command line execution
        subprocess.run(["python", "scripts/make_sample_pdfs.py"], check=True)
        subprocess.run(["python", "-m", "syllabusbot.ingest"], check=True)

EXAMPLES = [
    "What is the policy for late assignment submissions in CS101?",
    "When does the add/drop period end?",
    "How is the final grade in CS101 calculated?",
    "What are the quiet hours in the residence halls?",
]

st.set_page_config(page_title="SyllabusBot", page_icon="🎓", layout="centered")


# --- one bot per server process (model load + Chroma handle are expensive) ---
@st.cache_resource(show_spinner="Loading embeddings and vector index...")
def load_bot() -> SyllabusBot:
    return SyllabusBot(get_settings())


def render_sources(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for index, cite in enumerate(citations, start=1):
            score = (
                f" · relevance {cite['relevance']:.3f}"
                if cite.get("relevance") is not None
                else ""
            )
            st.markdown(f"**{index}. {cite['label']}**{score}")
            st.caption(
                f"type: {cite['doc_type']}"
                + (f" · course: {cite['course_code']}" if cite["course_code"] else "")
            )
            st.text(cite["excerpt"][:1500])
            if index < len(citations):
                st.divider()


def main() -> None:
    st.title("🎓 SyllabusBot")
    st.caption(
        "Answers come only from the indexed university PDFs, with a citation on "
        "every fact. If it isn't in the documents, the bot says so."
    )

    try:
        bot = load_bot()
    except SyllabusBotError as exc:
        st.error(str(exc))
        st.stop()

    settings = bot.settings
    stats = bot.stats()

    # ---------------- sidebar ------------------------------------------------
    with st.sidebar:
        st.subheader("Index")
        st.metric("Chunks", stats["chunks"])
        st.metric("Documents", len(stats["files"]))
        if stats["files"]:
            st.caption("Indexed files")
            for name, count in sorted(stats["files"].items()):
                st.write(f"- {name} ({count})")
        if stats["courses"]:
            st.caption("Courses: " + ", ".join(sorted(stats["courses"])))

        st.subheader("Configuration")
        st.write(f"**Chat model:** `{settings.llm_provider}/{settings.chat_model}`")
        st.write(f"**Embeddings:** `{settings.embedding_provider}`")
        st.write(f"**Chunking:** {settings.chunk_size} / {settings.chunk_overlap}")

        top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, settings.top_k)
        if top_k != bot.settings.top_k:
            bot.settings = replace(bot.settings, top_k=top_k)
            bot.retriever.settings = bot.settings

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption("Re-index after changing PDFs:\n\n`python -m syllabusbot.ingest`")

    if stats["chunks"] == 0:
        st.warning(
            f"The index is empty. Put PDFs in `{settings.data_dir}` and run "
            "`python -m syllabusbot.ingest` (or `python scripts/make_sample_pdfs.py` "
            "first for demo data), then reload this page."
        )
        st.stop()

    # ---------------- transcript --------------------------------------------
    st.session_state.setdefault("messages", [])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("citations", []))

    # Example buttons only while the conversation is empty.
    prefill: str | None = None
    if not st.session_state.messages:
        st.write("**Try one of these:**")
        columns = st.columns(2)
        for index, example in enumerate(EXAMPLES):
            if columns[index % 2].button(example, use_container_width=True):
                prefill = example

    question = st.chat_input("Ask about a syllabus, deadline, or campus policy...") or prefill

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Previous turns, as the (question, answer) pairs the chain expects.
    history = [
        (
            st.session_state.messages[i]["content"],
            st.session_state.messages[i + 1]["content"],
        )
        for i in range(0, len(st.session_state.messages) - 1, 2)
        if st.session_state.messages[i]["role"] == "user"
        and st.session_state.messages[i + 1]["role"] == "assistant"
    ]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                retrieval = bot.prepare(question, history)
            answer_text = st.write_stream(bot.stream_answer(retrieval))
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the app
            st.error(f"{type(exc).__name__}: {exc}")
            st.session_state.messages.pop()  # drop the unanswered question
            return

        citations = build_citations(retrieval.documents)
        render_sources(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "citations": citations}
    )


if __name__ == "__main__":
    main()

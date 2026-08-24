"""SyllabusBot — a grounded RAG assistant for university course documents.

Architecture (each concern lives in exactly one module):

    config.py       typed settings loaded from .env / environment
    loaders.py      PDF discovery + metadata enrichment (course, doc type)
    ingest.py       chunking (RecursiveCharacterTextSplitter) + idempotent upsert
    vectorstore.py  embedding factory + persistent ChromaDB handle
    llm.py          chat model factory (Anthropic / OpenAI)
    prompts.py      the strict, citation-enforcing system prompt
    retriever.py    top-k retrieval, optional course pre-filter, doc formatting
    chain.py        the LCEL chain + the SyllabusBot facade
    cli.py          command-line chat loop
    ../app.py       optional Streamlit UI

Typical use:

    from syllabusbot import SyllabusBot
    bot = SyllabusBot()
    print(bot.ask("What is the late submission policy in CS101?").answer)
"""

from syllabusbot.chain import Answer, SyllabusBot, build_rag_chain
from syllabusbot.config import Settings, get_settings
from syllabusbot.errors import (
    MissingCredentialsError,
    MissingDependencyError,
    SyllabusBotError,
)

__all__ = [
    "Answer",
    "MissingCredentialsError",
    "MissingDependencyError",
    "Settings",
    "SyllabusBot",
    "SyllabusBotError",
    "build_rag_chain",
    "get_settings",
]

__version__ = "1.0.0"

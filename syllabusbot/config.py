"""Central configuration.

One frozen dataclass, populated from environment variables (via `.env`), is
passed explicitly to every component. No module reads `os.environ` on its own,
which keeps the pipeline testable — a test just builds `Settings(...)`.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at runtime; env vars work either way.
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover
    pass

# Repository root = parent of the `syllabusbot` package. Relative paths in the
# environment are resolved against it so the CLI works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_TRUE = {"1", "true", "t", "yes", "y", "on"}


def _s(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or not value.strip() else value.strip()


def _i(key: str, default: int) -> int:
    try:
        return int(_s(key, str(default)))
    except ValueError:
        return default


def _f(key: str, default: float) -> float:
    try:
        return float(_s(key, str(default)))
    except ValueError:
        return default


def _b(key: str, default: bool) -> bool:
    return _s(key, str(default)).lower() in _TRUE


def _path(key: str, default: str) -> Path:
    raw = Path(_s(key, default)).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def _slug(text: str) -> str:
    """Chroma collection names allow [a-zA-Z0-9._-] only."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")


@dataclass(frozen=True)
class Settings:
    # --- paths -------------------------------------------------------------
    data_dir: Path
    chroma_dir: Path

    # --- chunking (requirement: 1000 / 150) --------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- retrieval ---------------------------------------------------------
    top_k: int = 3
    min_relevance: float = 0.0  # 0.0 = keep all top-k; 0.2-0.4 = drop weak hits
    use_course_filter: bool = True
    rewrite_followups: bool = True

    # --- embeddings --------------------------------------------------------
    embedding_provider: str = "huggingface"  # huggingface | openai
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- chat model --------------------------------------------------------
    llm_provider: str = "anthropic"  # anthropic | openai
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024

    collection_prefix: str = "syllabusbot"

    # ----------------------------------------------------------------------
    @property
    def embedding_model(self) -> str:
        return (
            self.openai_embedding_model
            if self.embedding_provider == "openai"
            else self.hf_embedding_model
        )

    @property
    def chat_model(self) -> str:
        return (
            self.openai_model if self.llm_provider == "openai" else self.anthropic_model
        )

    @property
    def collection_name(self) -> str:
        """Embedding model is part of the collection name.

        Vector dimensionality is a property of the embedding model (384 for
        MiniLM, 1536 for text-embedding-3-small). Encoding it here means
        flipping EMBEDDING_PROVIDER creates a fresh, correctly-sized index
        instead of blowing up on a dimension mismatch at query time.
        """
        return f"{self.collection_prefix}_{_slug(self.embedding_provider)}_{_slug(self.embedding_model.split('/')[-1])}"

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k < 1:
            raise ValueError("TOP_K must be >= 1")
        if self.embedding_provider not in {"huggingface", "openai"}:
            raise ValueError(
                f"EMBEDDING_PROVIDER must be 'huggingface' or 'openai', got {self.embedding_provider!r}"
            )
        if self.llm_provider not in {"anthropic", "openai"}:
            raise ValueError(
                f"LLM_PROVIDER must be 'anthropic' or 'openai', got {self.llm_provider!r}"
            )


def load_settings() -> Settings:
    """Build Settings from the environment (no caching — handy in tests)."""
    settings = Settings(
        data_dir=_path("DATA_DIR", "data"),
        chroma_dir=_path("CHROMA_DIR", "storage/chroma"),
        chunk_size=_i("CHUNK_SIZE", 1000),
        chunk_overlap=_i("CHUNK_OVERLAP", 150),
        top_k=_i("TOP_K", 3),
        min_relevance=_f("MIN_RELEVANCE", 0.0),
        use_course_filter=_b("USE_COURSE_FILTER", True),
        rewrite_followups=_b("REWRITE_FOLLOWUPS", True),
        embedding_provider=_s("EMBEDDING_PROVIDER", "huggingface").lower(),
        hf_embedding_model=_s(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        openai_embedding_model=_s("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_provider=_s("LLM_PROVIDER", "anthropic").lower(),
        anthropic_model=_s("ANTHROPIC_MODEL", "claude-sonnet-5"),
        openai_model=_s("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=_f("LLM_TEMPERATURE", 0.0),
        max_tokens=_i("LLM_MAX_TOKENS", 1024),
    )
    settings.validate()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton used by the CLI / Streamlit app."""
    return load_settings()


def enable_utf8_stdout() -> None:
    """Make stdout tolerant of the '·' in citations.

    Windows consoles still default to a legacy code page; without this, printing
    a citation can raise UnicodeEncodeError on cp437 or mangle it in a redirected
    log. Called from every entry point.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 — not a real text stream (pipes, tests)
            pass

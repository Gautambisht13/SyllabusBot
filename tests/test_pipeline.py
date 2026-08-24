"""Fast unit tests for the deterministic parts of the pipeline.

These need no API key, no ChromaDB and no embedding model — they cover the logic
most likely to silently rot: metadata extraction, chunking, context formatting
and input normalisation.

    pytest tests/            (or)     python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from syllabusbot.chain import _normalize_input, format_history  # noqa: E402
from syllabusbot.config import Settings  # noqa: E402
from syllabusbot.ingest import build_splitter, chunk_documents, chunk_id  # noqa: E402
from syllabusbot.loaders import (  # noqa: E402
    extract_course_code,
    guess_section,
    infer_doc_type,
    make_citation,
)
from syllabusbot.retriever import detect_course_code, format_docs  # noqa: E402

SETTINGS = Settings(data_dir=Path("data"), chroma_dir=Path("storage/chroma"))


def test_course_code_from_filename() -> None:
    assert extract_course_code("CS101_Syllabus") == "CS101"
    assert extract_course_code("CS 101 Syllabus") == "CS101"
    assert extract_course_code("MATH-201A_outline") == "MATH201A"
    assert extract_course_code("Campus_Handbook") == ""


def test_course_code_ignores_rooms_and_years() -> None:
    """"Turing Hall 105" and "Fall 2026" must not become course codes."""
    body = "Turing Hall 105. Turing Hall 105. Fall 2026 term. Fall 2026 term."
    assert extract_course_code("Campus_Handbook", body) == ""
    assert detect_course_code("where is Room 214?") == ""


def test_course_code_needs_two_mentions_in_body() -> None:
    assert extract_course_code("handbook", "See CS101 for details.") == ""
    assert extract_course_code("handbook", "CS101 grading. CS101 labs.") == "CS101"


def test_detect_course_code_in_question() -> None:
    assert detect_course_code("late policy in cs 101?") == "CS101"
    assert detect_course_code("when is add/drop?") == ""


def test_doc_type_from_folder_then_filename() -> None:
    data = Path("data")
    assert infer_doc_type(data / "syllabi" / "CS101.pdf", data) == "syllabus"
    assert infer_doc_type(data / "misc" / "Academic_Calendar.pdf", data) == "calendar"
    assert infer_doc_type(data / "handbook" / "Guide.pdf", data) == "handbook"
    assert infer_doc_type(data / "misc" / "Notes.pdf", data) == "other"


def test_guess_section_finds_headings() -> None:
    assert guess_section("LATE SUBMISSION POLICY\nWork is due at 23:59.") == "LATE SUBMISSION POLICY"
    assert guess_section("3.2 Late Work\nPenalty applies.") == "3.2 Late Work"
    assert guess_section("work is due at 23:59 on the posted date.") == ""


def test_chunking_respects_size_and_carries_metadata() -> None:
    page = Document(
        page_content="LATE SUBMISSION POLICY\n" + ("Late work loses 10% per day. " * 120),
        metadata={
            "source": "CS101_Syllabus.pdf",
            "source_path": "syllabi/CS101_Syllabus.pdf",
            "page": 4,
            "total_pages": 12,
            "doc_type": "syllabus",
            "course_code": "CS101",
            "citation": make_citation("CS101_Syllabus.pdf", 4),
            "file_hash": "deadbeef",
        },
    )
    chunks = chunk_documents([page], build_splitter(SETTINGS))

    assert len(chunks) > 1, "a long page must split"
    # Chunks may exceed chunk_size by the length of a re-attached section
    # heading; nothing may exceed it by more than that.
    limit = SETTINGS.chunk_size + len("LATE SUBMISSION POLICY") + 1
    assert all(len(chunk.page_content) <= limit for chunk in chunks)
    first = chunks[0].metadata
    assert first["page"] == 4 and first["course_code"] == "CS101"
    assert first["section"] == "LATE SUBMISSION POLICY"
    assert first["citation"] == "[CS101_Syllabus.pdf, p.4 · LATE SUBMISSION POLICY]"
    # Chroma rejects non-primitive metadata values.
    assert all(
        isinstance(value, (str, int, float, bool))
        for chunk in chunks
        for value in chunk.metadata.values()
    )


def test_heading_only_fragment_is_inherited_by_following_chunks() -> None:
    """The heading is its own split here — every chunk must still carry it."""
    page = Document(
        page_content="LATE SUBMISSION POLICY\n" + ("Penalty is 10% per day. " * 150),
        metadata={
            "source": "CS101_Syllabus.pdf",
            "source_path": "syllabi/CS101_Syllabus.pdf",
            "page": 4,
            "citation": make_citation("CS101_Syllabus.pdf", 4),
        },
    )
    chunks = chunk_documents([page], build_splitter(SETTINGS))
    assert len(chunks) >= 3
    assert all(chunk.metadata["section"] == "LATE SUBMISSION POLICY" for chunk in chunks)
    assert all(chunk.page_content.startswith("LATE SUBMISSION POLICY") for chunk in chunks)


def test_section_does_not_leak_across_pages() -> None:
    pages = [
        Document(
            page_content="GRADING\n" + ("Assignments are 30% of the grade. " * 40),
            metadata={"source": "s.pdf", "source_path": "s.pdf", "page": 1, "citation": "[s.pdf, p.1]"},
        ),
        Document(
            page_content="Contact the registrar for enrolment questions. " * 20,
            metadata={"source": "s.pdf", "source_path": "s.pdf", "page": 2, "citation": "[s.pdf, p.2]"},
        ),
    ]
    chunks = chunk_documents(pages, build_splitter(SETTINGS))
    page_two = [chunk for chunk in chunks if chunk.metadata["page"] == 2]
    assert page_two, "page 2 should produce chunks"
    assert all(chunk.metadata["section"] == "" for chunk in page_two)


def test_chunk_ids_are_stable_and_content_sensitive() -> None:
    meta = {"source_path": "a.pdf", "page": 1, "start_index": 0}
    assert chunk_id(meta, "hello") == chunk_id(dict(meta), "hello")
    assert chunk_id(meta, "hello") != chunk_id(meta, "hello!")
    assert chunk_id(meta, "hello") != chunk_id({**meta, "page": 2}, "hello")


def test_format_docs_emits_a_cite_as_line_per_chunk() -> None:
    docs = [
        Document(
            page_content="Late work loses 10% per day.",
            metadata={
                "source": "CS101_Syllabus.pdf",
                "page": 4,
                "total_pages": 12,
                "doc_type": "syllabus",
                "course_code": "CS101",
                "citation": "[CS101_Syllabus.pdf, p.4]",
                "relevance": 0.7123,
            },
        )
    ]
    block = format_docs(docs)
    assert "CITE AS: [CS101_Syllabus.pdf, p.4]" in block
    assert "CHUNK 1 of 1" in block
    assert "relevance: 0.712" in block
    assert format_docs([]) == "(no documents retrieved)"


def test_normalize_input_accepts_str_and_dict() -> None:
    assert _normalize_input("  hi  ") == {"question": "hi", "chat_history": ""}
    normalized = _normalize_input({"question": "hi", "chat_history": [("q", "a")]})
    assert normalized["question"] == "hi"
    assert "Student: q" in normalized["chat_history"]


def test_format_history_is_empty_when_no_turns() -> None:
    assert format_history(None) == ""
    assert format_history([]) == ""
    assert "SyllabusBot: a" in format_history([("q", "a")])


def test_settings_validation_and_collection_naming() -> None:
    assert SETTINGS.collection_name == "syllabusbot_huggingface_all_MiniLM_L6_v2"
    import dataclasses

    openai_settings = dataclasses.replace(SETTINGS, embedding_provider="openai")
    assert openai_settings.collection_name != SETTINGS.collection_name

    for bad in (
        {"chunk_overlap": 1000},
        {"top_k": 0},
        {"embedding_provider": "cohere"},
        {"llm_provider": "llama"},
    ):
        try:
            dataclasses.replace(SETTINGS, **bad).validate()
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(dict(globals()).items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)

"""Document loading + metadata enrichment.

The retriever is only as good as its metadata: requirement #4 says the answer
must cite *source filename and page/section*, so every page we load is tagged
with a ready-to-print citation before it ever reaches the splitter.

Metadata written per chunk:
    source        "CS101_Syllabus.pdf"            (filename — what we cite)
    source_path   "syllabi/CS101_Syllabus.pdf"    (repo-relative, for re-ingest)
    page          4                               (1-indexed, human page number)
    total_pages   12
    doc_type      "syllabus" | "calendar" | "handbook" | "policy" | "other"
    course_code   "CS101" or ""                   (enables the course filter)
    section       "LATE SUBMISSION POLICY" or ""  (nearest heading in the chunk)
    citation      "[CS101_Syllabus.pdf, p.4]"     (copied verbatim by the LLM)
    file_hash     sha256 of the PDF               (change detection on re-ingest)
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document

from syllabusbot.errors import MissingDependencyError

logger = logging.getLogger(__name__)

# Filename / folder keyword -> document type. First match wins, so order the
# more specific terms first.
DOC_TYPE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("syllab", "syllabus"),
    ("course_outline", "syllabus"),
    ("outline", "syllabus"),
    ("calendar", "calendar"),
    ("academic_year", "calendar"),
    ("timetable", "calendar"),
    ("handbook", "handbook"),
    ("student_guide", "handbook"),
    ("code_of_conduct", "policy"),
    ("policy", "policy"),
    ("regulation", "policy"),
)

# Matches CS101, CS 101, CS-101, MATH201A, BIOL 1010. Lookarounds instead of
# \b because "_" is a word character: `\b` would fail on "CS101_Syllabus.pdf",
# which is exactly the filename shape we care most about.
COURSE_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z]{2,4})[\s\-_]?(\d{3,4}[A-Z]?)(?![A-Za-z0-9])"
)

# Words that produce a course-code-shaped false positive ("Turing Hall 105",
# "Room 214", "Fall 2026"). Only 2-4 letter entries can ever match.
NON_COURSE_PREFIXES = frozenset(
    {
        "ROOM", "HALL", "BLDG", "STE", "APT", "BOX", "UNIT", "WING", "DESK",
        "SEAT", "LOT", "PAGE", "SEC", "FIG", "TAB", "ART", "REV", "VER", "NO",
        "FALL", "TEL", "FAX", "EXT", "PIN", "CRN", "ISO", "RFC", "USD", "EUR",
        "AM", "PM",
    }
)


# A "heading" line: ALL CAPS, or numbered ("3.2 Late Work"), or Title Case and
# short. Used to attach a human-meaningful section label to each chunk.
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Z0-9 &/,'\-\(\)]{3,60}"  # ALL CAPS
    r"|(?:\d+(?:\.\d+)*\.?\s+)[A-Z][\w &/,'\-\(\)]{2,60}"  # 3.2 Late Work
    r"|(?:[A-Z][a-z]+\s){1,5}(?:Policy|Policies|Deadlines?|Schedule|Grading|Assessment|Information|Hours|Dates|Rules|Conduct|Overview|Objectives)"
    r")\s*:?\s*$"
)


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Content fingerprint, streamed so a 200-page PDF stays cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_pdfs(data_dir: Path) -> list[Path]:
    """All PDFs under `data_dir`, recursively, skipping editor/OS junk."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Create it and drop your syllabi / calendar / handbook PDFs inside "
            "(subfolders are fine), or run: python scripts/make_sample_pdfs.py"
        )
    return sorted(
        p
        for p in data_dir.rglob("*.pdf")
        if p.is_file() and not p.name.startswith((".", "~$"))
    )


def infer_doc_type(path: Path, data_dir: Path) -> str:
    """Classify from the folder name first (data/syllabi/...), then filename."""
    try:
        parts = path.relative_to(data_dir).parts
    except ValueError:  # path outside data_dir
        parts = (path.name,)
    haystack = "_".join(parts).lower().replace(" ", "_").replace("-", "_")
    for keyword, doc_type in DOC_TYPE_KEYWORDS:
        if keyword in haystack:
            return doc_type
    return "other"


def iter_course_codes(text: str) -> Iterator[str]:
    """Yield normalised course codes ("CS 101" -> "CS101"), skipping false hits."""
    for dept, number in COURSE_CODE_RE.findall(text):
        if dept in NON_COURSE_PREFIXES:
            continue
        yield f"{dept}{number}"


def extract_course_code(filename: str, first_page_text: str = "") -> str:
    """Course code for the document.

    Filename wins because it is authoritative ("CS101_Syllabus.pdf"). We only
    fall back to page 1 text, and only to a code that appears at least twice —
    a single stray mention is usually a cross-reference to another course.
    """
    for code in iter_course_codes(filename.upper()):
        return code

    counts: dict[str, int] = {}
    for code in iter_course_codes(first_page_text[:4000]):
        counts[code] = counts.get(code, 0) + 1
    if counts:
        code, hits = max(counts.items(), key=lambda kv: kv[1])
        if hits >= 2:
            return code
    return ""


def guess_section(text: str) -> str:
    """Nearest heading inside a chunk, for `p.4 · LATE SUBMISSION POLICY`."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if 4 <= len(line) <= 70 and _HEADING_RE.match(line):
            return line.rstrip(":").strip()
    return ""


def make_citation(source: str, page: int | None) -> str:
    """The exact string the LLM is told to copy. Single source of truth."""
    return f"[{source}, p.{page}]" if page else f"[{source}]"


def _clean_page_text(text: str) -> str:
    """Normalise PDF extraction artefacts.

    Extracted text arrives with ragged intra-line spacing and long runs of blank
    lines, which confuse both the heading detector and the splitter's paragraph
    boundary. Collapse them without touching the words themselves.
    """
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_pdf(path: Path, data_dir: Path) -> list[Document]:
    """Load one PDF into page-level Documents with enriched metadata.

    Uses pypdf directly — the same extraction LangChain's PyPDFLoader performs —
    which keeps the dependency list smaller and gives us control over metadata
    and whitespace cleanup. (`langchain-community`, which hosts PyPDFLoader, is
    being sunset upstream.)

    Empty/scanned pages are dropped: they add no retrievable text but would
    otherwise pollute the index with zero-information vectors. For scanned PDFs,
    OCR first — e.g. `ocrmypdf in.pdf out.pdf`.
    """
    try:
        from pypdf import PdfReader  # lazy import
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "PDF loading needs pypdf:\n    pip install pypdf"
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 — corrupt/encrypted file
        logger.error("Cannot read %s: %s", path.name, exc)
        return []

    total_pages = len(reader.pages)
    if total_pages == 0:
        logger.warning("%s has no pages", path.name)
        return []

    rel_path = path.relative_to(data_dir).as_posix()
    doc_type = infer_doc_type(path, data_dir)
    file_hash = sha256_file(path)

    texts: list[str] = []
    for page in reader.pages:
        try:
            texts.append(_clean_page_text(page.extract_text() or ""))
        except Exception as exc:  # noqa: BLE001 — one bad page shouldn't fail the file
            logger.warning("%s: page extraction failed (%s)", path.name, exc)
            texts.append("")

    if not any(texts):
        logger.warning("No extractable text in %s (scanned image PDF?)", path.name)
        return []

    course_code = extract_course_code(path.stem, texts[0])

    documents = [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "source_path": rel_path,
                "page": page_number,  # 1-indexed: what a human reads on the page
                "total_pages": total_pages,
                "doc_type": doc_type,
                "course_code": course_code,
                "citation": make_citation(path.name, page_number),
                "file_hash": file_hash,
            },
        )
        for page_number, text in enumerate(texts, start=1)
        if text
    ]

    if len(documents) < total_pages:
        logger.info(
            "%s: kept %d/%d pages (blank or image-only pages skipped)",
            path.name,
            len(documents),
            total_pages,
        )
    return documents

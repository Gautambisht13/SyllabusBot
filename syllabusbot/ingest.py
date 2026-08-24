"""Ingestion pipeline: PDF -> pages -> chunks -> embeddings -> ChromaDB.

Run it whenever the documents change:

    python -m syllabusbot.ingest                 # incremental (skips unchanged)
    python -m syllabusbot.ingest --reset         # wipe the collection, rebuild
    python -m syllabusbot.ingest --stats         # what's indexed right now
    python -m syllabusbot.ingest --dry-run       # chunk only, no embedding

Idempotency: each PDF's sha256 is stored on every chunk it produces. A re-run
skips files whose hash is unchanged, and for changed files deletes the old
chunks before inserting new ones — so editing one syllabus never leaves stale
text in the index and never duplicates it.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from syllabusbot.config import Settings, enable_utf8_stdout, get_settings
from syllabusbot.errors import SyllabusBotError
from syllabusbot.loaders import discover_pdfs, guess_section, load_pdf
from syllabusbot.vectorstore import collection_summary, get_vectorstore

logger = logging.getLogger(__name__)

# Chroma accepts far bigger batches, but embedding in slices keeps memory flat
# and gives useful progress output on a 500-page handbook.
BATCH_SIZE = 128

# Below this length a chunk is almost certainly a running header/footer or a
# stray heading fragment rather than retrievable content.
MIN_CHUNK_CHARS = 40


def build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """chunk_size=1000, chunk_overlap=150, split on the largest unit that fits.

    Separator order matters: paragraph -> line -> sentence -> word -> character.
    A policy paragraph ("Late work loses 10% per day...") therefore stays whole,
    and the 150-char overlap keeps a rule that straddles a boundary retrievable
    from either side.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
        add_start_index=True,  # -> metadata["start_index"], used in the chunk id
        keep_separator=True,
    )


def chunk_id(metadata: dict[str, Any], text: str) -> str:
    """Deterministic id: same PDF + same position + same text -> same id.

    Content is hashed too, so a reflowed page yields new ids and the delete-then-
    insert path below cannot leave an orphan behind.
    """
    payload = (
        f"{metadata.get('source_path')}|{metadata.get('page')}|"
        f"{metadata.get('start_index', 0)}|{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def chunk_documents(
    pages: list[Document], splitter: RecursiveCharacterTextSplitter
) -> list[Document]:
    """Split pages and finish the per-chunk metadata (section label, citation).

    The splitter happily emits a heading as its own tiny fragment ("LATE
    SUBMISSION POLICY"). Dropping such fragments as page furniture would throw
    away the one phrase a student is most likely to search for, so we remember
    the last heading seen and hand it down to the chunks that follow it.
    """
    chunks = splitter.split_documents(pages)
    kept: list[Document] = []
    current_section = ""
    current_page = None

    for chunk in chunks:
        text = chunk.page_content.strip()
        # A heading only governs the page it appears on.
        if chunk.metadata.get("page") != current_page:
            current_page, current_section = chunk.metadata.get("page"), ""

        heading = guess_section(text)
        if heading:
            current_section = heading

        if len(text) < MIN_CHUNK_CHARS:
            # Page furniture (running headers, footers, page numbers) — dropped,
            # but if it was a standalone heading we just recorded it above.
            continue

        if not heading and current_section:
            # Re-attach the inherited heading: it improves the embedding *and*
            # tells the reader which policy the excerpt belongs to.
            text = f"{current_section}\n{text}"

        chunk.page_content = text
        section = heading or current_section
        chunk.metadata["section"] = section
        chunk.metadata["chunk_index"] = len(kept)
        # Enrich the citation with the section when we found one:
        #   [CS101_Syllabus.pdf, p.4 · LATE SUBMISSION POLICY]
        if section:
            base = chunk.metadata.get("citation", "")
            chunk.metadata["citation"] = (
                f"{base[:-1]} · {section}]" if base.endswith("]") else base
            )

        # Chroma only stores str/int/float/bool — drop None and stringify the rest.
        chunk.metadata = {
            key: (value if isinstance(value, (str, int, float, bool)) else str(value))
            for key, value in chunk.metadata.items()
            if value is not None
        }
        kept.append(chunk)

    return kept


def _existing_ids_and_hash(store: Any, source_path: str) -> tuple[list[str], str]:
    """Ids already indexed for this PDF plus the file hash they were built from."""
    try:
        found = store.get(where={"source_path": source_path}, include=["metadatas"])
    except Exception as exc:  # noqa: BLE001 — empty/new collection
        logger.debug("Lookup failed for %s: %s", source_path, exc)
        return [], ""
    ids = list(found.get("ids") or [])
    metadatas = found.get("metadatas") or []
    stored_hash = (metadatas[0] or {}).get("file_hash", "") if metadatas else ""
    return ids, str(stored_hash)


def ingest(
    settings: Settings | None = None,
    *,
    reset: bool = False,
    force: bool = False,
    dry_run: bool = False,
    store: Any = None,
) -> dict[str, Any]:
    """Index every PDF under `settings.data_dir`. Returns a run report.

    `store` lets a caller inject an already-built vector store (used by
    scripts/smoke_test.py to run the whole pipeline on stub embeddings).
    """
    settings = settings or get_settings()
    splitter = build_splitter(settings)
    pdfs = discover_pdfs(settings.data_dir)

    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found under {settings.data_dir}. Add your syllabi / calendar / "
            "handbook PDFs there, or run: python scripts/make_sample_pdfs.py"
        )

    print(f"Found {len(pdfs)} PDF(s) under {settings.data_dir}")

    if dry_run:
        # No embedding model, no database — just prove the chunking is sane.
        total = 0
        for pdf in pdfs:
            chunks = chunk_documents(load_pdf(pdf, settings.data_dir), splitter)
            total += len(chunks)
            sizes = [len(c.page_content) for c in chunks] or [0]
            print(
                f"  {pdf.name}: {len(chunks)} chunks "
                f"(min {min(sizes)} / avg {sum(sizes)//len(sizes)} / max {max(sizes)} chars)"
            )
            if chunks:
                meta = chunks[0].metadata
                print(
                    f"      type={meta.get('doc_type')} course={meta.get('course_code') or '-'} "
                    f"first_citation={meta.get('citation')}"
                )
        return {"dry_run": True, "files": len(pdfs), "chunks": total}

    store = store if store is not None else get_vectorstore(settings)

    if reset:
        print(f"Resetting collection '{settings.collection_name}' ...")
        try:
            store.delete_collection()
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_collection failed (%s); continuing", exc)
        store = get_vectorstore(settings)  # re-open, now empty

    report = {"files": 0, "skipped": 0, "chunks": 0, "replaced": 0, "empty": 0}

    for pdf in pdfs:
        rel_path = pdf.relative_to(settings.data_dir).as_posix()
        pages = load_pdf(pdf, settings.data_dir)
        if not pages:
            report["empty"] += 1
            print(f"  ! {rel_path}: no extractable text — skipped (needs OCR?)")
            continue

        file_hash = pages[0].metadata["file_hash"]
        existing_ids, stored_hash = _existing_ids_and_hash(store, rel_path)

        if existing_ids and stored_hash == file_hash and not force:
            report["skipped"] += 1
            print(f"  = {rel_path}: unchanged ({len(existing_ids)} chunks) - skipped")
            continue

        if existing_ids:  # changed file: remove the old chunks first
            store.delete(ids=existing_ids)
            report["replaced"] += 1

        chunks = chunk_documents(pages, splitter)
        if not chunks:
            report["empty"] += 1
            print(f"  ! {rel_path}: produced no usable chunks — skipped")
            continue

        ids = [chunk_id(chunk.metadata, chunk.page_content) for chunk in chunks]
        # Guard against the pathological case of two byte-identical chunks at
        # the same offset — duplicate ids would make Chroma's upsert lossy.
        seen: set[str] = set()
        unique: list[tuple[str, Document]] = []
        for identifier, chunk in zip(ids, chunks):
            if identifier in seen:
                continue
            seen.add(identifier)
            unique.append((identifier, chunk))

        for start in range(0, len(unique), BATCH_SIZE):
            batch = unique[start : start + BATCH_SIZE]
            store.add_documents(
                documents=[chunk for _, chunk in batch],
                ids=[identifier for identifier, _ in batch],
            )

        report["files"] += 1
        report["chunks"] += len(unique)
        marker = "~" if existing_ids else "+"
        print(f"  {marker} {rel_path}: {len(unique)} chunks indexed")

    print(
        f"\nDone. {report['chunks']} chunks from {report['files']} file(s) "
        f"({report['replaced']} updated, {report['skipped']} unchanged, "
        f"{report['empty']} unusable)."
    )
    print(f"Collection '{settings.collection_name}' at {settings.chroma_dir}")
    return report


def print_stats(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    summary = collection_summary(get_vectorstore(settings))
    print(f"Collection : {settings.collection_name}")
    print(f"Location   : {settings.chroma_dir}")
    print(f"Embeddings : {settings.embedding_provider} / {settings.embedding_model}")
    print(f"Chunks     : {summary['chunks']}")
    if not summary["chunks"]:
        print("\nIndex is empty — run: python -m syllabusbot.ingest")
        return
    for label, key in (("Documents", "files"), ("Types", "doc_types"), ("Courses", "courses")):
        items = sorted(summary[key].items(), key=lambda kv: -kv[1])
        print(f"\n{label}:")
        for name, count in items:
            print(f"  {count:5d}  {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m syllabusbot.ingest",
        description="Chunk and index university PDFs into ChromaDB.",
    )
    parser.add_argument("--data-dir", type=Path, help="override DATA_DIR")
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    parser.add_argument("--force", action="store_true", help="re-index even if unchanged")
    parser.add_argument("--dry-run", action="store_true", help="chunk only; no embeddings")
    parser.add_argument("--stats", action="store_true", help="show index contents and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    enable_utf8_stdout()  # citations contain '·'

    settings = get_settings()
    if args.data_dir:
        from dataclasses import replace

        settings = replace(settings, data_dir=args.data_dir.resolve())

    try:
        if args.stats:
            print_stats(settings)
        else:
            ingest(settings, reset=args.reset, force=args.force, dry_run=args.dry_run)
    except (FileNotFoundError, SyllabusBotError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

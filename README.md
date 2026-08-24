# SyllabusBot

A RAG assistant that answers student questions about **course syllabi, the
academic calendar and campus handbooks** — and cites the filename, page and
section for every fact. If the answer isn't in the documents, it says so instead
of inventing a policy.

```
You: What is the policy for late assignment submissions in CS101?

SyllabusBot: Assignments are due at 23:59 on the posted due date, and late work
is accepted for up to 72 hours with a penalty of 10% of the earned score per
24-hour period or part thereof [CS101_Syllabus.pdf, p.2 · LATE SUBMISSION POLICY].
After 72 hours a late submission receives a zero and gets no feedback
[CS101_Syllabus.pdf, p.2 · LATE SUBMISSION POLICY]. You also have three
penalty-free late days per semester for programming assignments only, claimed in
the portal before the deadline [CS101_Syllabus.pdf, p.2 · LATE SUBMISSION POLICY].

Sources: [CS101_Syllabus.pdf, p.2 · LATE SUBMISSION POLICY]; [CS101_Syllabus.pdf, p.1 · GRADING]
```

---

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env                                # then add ANTHROPIC_API_KEY (or OPENAI_API_KEY)
python scripts/make_sample_pdfs.py                  # optional: 3 realistic demo PDFs
python -m syllabusbot.ingest                        # chunk + embed + store
python -m syllabusbot.cli                           # chat
```

Streamlit instead of the terminal:

```bash
streamlit run app.py
```

Out of the box, embeddings run **locally** (`all-MiniLM-L6-v2`, no API key, no
cost). Only the chat model needs a key.

---

## Architecture

```
data/*.pdf
   |
   |  loaders.py      pypdf -> one Document per page
   |                  + metadata: source, page, doc_type, course_code, citation, file_hash
   v
   |  ingest.py       RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
   |                  + section label carried down from the nearest heading
   |                  + deterministic chunk ids -> idempotent re-ingest
   v
   |  vectorstore.py  HuggingFace all-MiniLM-L6-v2 (or OpenAI) -> ChromaDB (persistent, cosine)
   v
   |  retriever.py    top-k = 3 similarity search
   |                  + course-code metadata pre-filter, with unfiltered fallback
   v
   |  chain.py        LCEL: normalize -> retrieve -> format context -> prompt -> LLM -> str
   |  prompts.py      closed-book system prompt; every fact must carry a CITE AS tag
   v
   cli.py  /  app.py
```

| Module | Responsibility |
| --- | --- |
| [config.py](syllabusbot/config.py) | One frozen `Settings` dataclass from `.env`; nothing else reads the environment |
| [loaders.py](syllabusbot/loaders.py) | PDF discovery, text extraction, metadata enrichment |
| [ingest.py](syllabusbot/ingest.py) | Chunking + idempotent upsert; `--reset`, `--dry-run`, `--stats` |
| [vectorstore.py](syllabusbot/vectorstore.py) | Embedding factory + persistent Chroma handle + index stats |
| [llm.py](syllabusbot/llm.py) | Chat model factory (Anthropic / OpenAI) with actionable credential errors |
| [prompts.py](syllabusbot/prompts.py) | The grounding rules, the refusal sentence, the follow-up condenser |
| [retriever.py](syllabusbot/retriever.py) | Top-k retrieval, course filter, `<context>` formatting, citation list |
| [chain.py](syllabusbot/chain.py) | The LCEL chain + `SyllabusBot` facade (`ask`, `stream`, `stats`) |
| [cli.py](syllabusbot/cli.py) | Terminal chat loop with slash commands |
| [app.py](app.py) | Streamlit UI (same pipeline, presentation only) |

### The chain

```python
RunnableLambda(_normalize_input)                                  # str or dict in
| RunnablePassthrough.assign(documents=itemgetter("question") | retriever)
| RunnablePassthrough.assign(context=format_docs)                 # CITE AS: blocks
| RunnablePassthrough.assign(answer=RunnableBranch(
      (no documents, canned refusal),                             # no LLM call
      RAG_PROMPT | llm | StrOutputParser(),
  ))
```

Use it directly if you'd rather not use the facade:

```python
from syllabusbot import build_rag_chain
chain = build_rag_chain()
result = chain.invoke("When does the add/drop period end?")
print(result["answer"], result["documents"])
```

---

## How grounding is enforced

Requirement: answer **only** from the retrieved context, and cite the source
filename and page/section. Four mechanisms, because a prompt alone is not a
guarantee:

1. **Closed-book system prompt** ([prompts.py](syllabusbot/prompts.py)) — explicit
   rules against outside knowledge, extrapolation and unit conversion; a fixed
   refusal sentence; instructions to surface conflicts between documents rather
   than silently picking one; and an instruction to ignore any text inside the
   context that looks like a command (retrieved text is data, not instructions).
2. **Citation-by-copying** — each chunk is rendered with a literal
   `CITE AS: [CS101_Syllabus.pdf, p.2 · LATE SUBMISSION POLICY]` line, built from
   metadata by `make_citation()`. The model copies a string rather than composing
   one, which is what makes citations reliable.
3. **Local refusal** — if retrieval returns nothing, `RunnableBranch` returns the
   canned refusal without calling the model at all. "No context" is a property of
   the pipeline, not a hope about the model.
4. **Verifiable output** — `Answer.citations` carries the exact chunk text behind
   each citation, so `/sources` in the CLI (or the Sources panel in Streamlit)
   lets a student check the claim against the document.

Set `MIN_RELEVANCE=0.25` in `.env` for a stricter regime: weak matches are dropped
before generation, so the bot refuses more often and hallucinates less.

---

## Retrieval design notes

**Course-code pre-filter.** Syllabi are near-identical across courses, so plain
similarity search happily answers a CS101 question with CS102's grading table.
When a course code appears in the question (`"...in CS 101?"`), the code is pushed
into Chroma as a metadata filter; if that course isn't indexed, the search is
retried unfiltered rather than refusing on a technicality. Disable with
`USE_COURSE_FILTER=false`.

**Heading inheritance.** The splitter often emits a heading (`LATE SUBMISSION
POLICY`) as its own tiny fragment. Dropping it as page furniture would delete the
exact phrase students search for, so the heading is remembered and re-attached to
the following chunks on the same page — improving both the embedding and the
readability of the excerpt.

**Cosine space.** The collection is created with `hnsw:space=cosine`, so
`relevance` scores shown next to each source are `1 - cosine_distance` and
comparable across queries. (Chroma's default is L2, which makes those scores
meaningless — and negative.)

**Idempotent ingest.** Every chunk stores its PDF's sha256. A re-run skips
unchanged files and, for changed ones, deletes the old chunks before inserting the
new ones — so editing a syllabus never leaves stale text in the index.

**Embedding model in the collection name.** Vector width is a property of the
embedding model (384 for MiniLM, 1536 for `text-embedding-3-small`), so switching
`EMBEDDING_PROVIDER` creates a fresh, correctly-sized collection instead of
failing on a dimension mismatch at query time.

---

## Commands

```bash
python -m syllabusbot.ingest              # incremental index
python -m syllabusbot.ingest --reset      # rebuild from scratch
python -m syllabusbot.ingest --dry-run    # chunk stats only; no embedding, no DB
python -m syllabusbot.ingest --stats      # what's indexed

python -m syllabusbot.cli                 # interactive chat
python -m syllabusbot.cli -q "When is the tuition deadline?"   # one-shot; exit 3 if ungrounded
python -m syllabusbot.cli --show-context   # print the retrieved chunks each turn
```

In-chat: `/help` `/sources` `/context` `/stats` `/k 5` `/stream` `/clear` `/exit`.

---

## Configuration

All settings live in `.env` (see [.env.example](.env.example)). The ones that
matter most:

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `anthropic` | `anthropic` (claude-sonnet-5) or `openai` |
| `EMBEDDING_PROVIDER` | `huggingface` | local MiniLM; `openai` for better retrieval at a cost |
| `TOP_K` | `3` | chunks per question |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | re-run ingest with `--reset` after changing |
| `MIN_RELEVANCE` | `0.0` | `0.25`-`0.4` trades recall for precision |
| `USE_COURSE_FILTER` | `true` | course-code metadata pre-filter |
| `REWRITE_FOLLOWUPS` | `true` | rewrites "and the final exam?" into a standalone query |

---

## Tests

```bash
python tests/test_pipeline.py     # 14 unit tests: metadata, chunking, formatting (no deps beyond langchain-core)
python scripts/smoke_test.py      # 26 end-to-end checks: real pypdf + real ChromaDB, stub embeddings + fake LLM
```

The smoke test needs no API key and downloads nothing. It asserts that ingestion
is idempotent, that the CS101 question retrieves the late-policy chunk, that the
context reaching the model carries `CITE AS` tags, and that an empty index
refuses without spending a token. Both suites pass on Python 3.12.

Not covered by automated tests: the two live provider calls (`ChatAnthropic` /
`ChatOpenAI`) and the real embedding models — those need credentials or a ~2 GB
torch install. Their parameters were verified against the installed package
signatures.

---

## Adding your own documents

Drop PDFs anywhere under `data/` and re-run `python -m syllabusbot.ingest`.
Folder names drive the `doc_type` metadata (`syllabi/`, `calendar/`, `handbook/`),
and a course code in the filename (`CS101_Syllabus.pdf`) enables the course
filter — so that naming convention is worth keeping.

Scanned PDFs have no extractable text and are skipped with a warning; OCR them
first (`ocrmypdf in.pdf out.pdf`).

---

## Limitations

- **Retrieval is per-chunk.** A question whose answer spans five pages of a
  handbook may only get three chunks. Raise `TOP_K` for broad questions.
- **No table understanding.** A grading table in a PDF becomes flat text; numbers
  survive but column alignment does not.
- **Semantic search only.** No BM25/hybrid retrieval, so an exact code that the
  embedding model has never seen (`ENGR-4890X`) may not match well. The course
  filter covers the common case.
- **Multi-turn is stateless at the retrieval layer.** Follow-ups are handled by
  rewriting the question, not by re-ranking against the conversation.
- **The bot quotes the documents, not the truth.** If a syllabus PDF is out of
  date, so is the answer — which is why every fact is cited.

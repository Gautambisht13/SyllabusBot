"""Command-line chat loop.

    python -m syllabusbot.cli
    python -m syllabusbot.cli -q "What is the late submission policy in CS101?"
    python -m syllabusbot.cli --show-context      # debug what was retrieved

Commands inside the loop: /help /sources /context /stats /k N /stream /clear /exit
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from dataclasses import replace

from syllabusbot.chain import Answer, SyllabusBot
from syllabusbot.config import Settings, enable_utf8_stdout, get_settings
from syllabusbot.errors import SyllabusBotError
from syllabusbot.retriever import build_citations

BANNER = r"""
   _____       _ _       _               ____        _
  / ____|     | | |     | |             |  _ \      | |
 | (___  _   _| | | __ _| |__  _   _ ___| |_) | ___ | |_
  \___ \| | | | | |/ _` | '_ \| | | / __|  _ < / _ \| __|
  ____) | |_| | | | (_| | |_) | |_| \__ \ |_) | (_) | |_
 |_____/ \__, |_|_|\__,_|_.__/ \__,_|___/____/ \___/ \__|
          __/ |   grounded answers from your university PDFs
         |___/
"""

HELP = """\
Commands
  /help              show this help
  /sources           full text of the chunks behind the last answer
  /context           the exact <context> block sent to the model
  /stats             what is currently indexed
  /k N               change how many chunks are retrieved (current: {k})
  /stream            toggle token streaming (current: {stream})
  /clear             forget the conversation history
  /exit, /quit       leave (Ctrl+C / Ctrl+D also work)

Try
  What is the policy for late assignment submissions in CS101?
  When does the add/drop period end?
  How many unexcused absences are allowed before it affects my grade?
"""


def _print_sources(answer: Answer | None, *, full: bool = False) -> None:
    if answer is None or not answer.citations:
        print("No sources yet — ask a question first.")
        return
    print("\nSources")
    for index, cite in enumerate(answer.citations, start=1):
        score = f"  relevance {cite['relevance']:.3f}" if cite.get("relevance") is not None else ""
        print(f"  {index}. {cite['label']}{score}")
        if full:
            excerpt = cite["excerpt"].strip()[:1200]
            print("     " + excerpt.replace("\n", "\n     "))
            print()


def _print_header(settings: Settings, bot: SyllabusBot) -> bool:
    """Print the runtime config + index summary. False if the index is empty."""
    stats = bot.stats()
    print(f"Chat model : {settings.llm_provider} / {settings.chat_model}")
    print(f"Embeddings : {settings.embedding_provider} / {settings.embedding_model}")
    print(f"Index      : {stats['chunks']} chunks from {len(stats['files'])} document(s)")
    print(f"Retrieval  : top-{settings.top_k}, chunk {settings.chunk_size}/{settings.chunk_overlap}")
    if stats["courses"]:
        print(f"Courses    : {', '.join(sorted(stats['courses']))}")

    if stats["chunks"] == 0:
        print(
            "\nThe index is empty. Add PDFs to "
            f"{settings.data_dir} and run:\n"
            "    python -m syllabusbot.ingest\n"
        )
        return False
    print("\nType a question, or /help for commands.\n")
    return True


def _answer_streaming(bot: SyllabusBot, question: str, history) -> Answer:
    """Retrieve once, stream the answer tokens, then assemble the Answer."""
    retrieval = bot.prepare(question, history)
    if retrieval.standalone_question != question:
        print(f"(searching for: {retrieval.standalone_question})")

    print("\nSyllabusBot: ", end="", flush=True)
    parts: list[str] = []
    for token in bot.stream_answer(retrieval):
        parts.append(token)
        print(token, end="", flush=True)
    print()

    return Answer(
        question=question,
        answer="".join(parts).strip(),
        documents=retrieval.documents,
        citations=build_citations(retrieval.documents),
        standalone_question=retrieval.standalone_question,
    )


def chat_loop(bot: SyllabusBot, *, stream: bool = True, show_context: bool = False) -> None:
    history: deque[tuple[str, str]] = deque(maxlen=6)
    last: Answer | None = None
    last_context = ""

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not question:
            continue

        # ---- slash commands ------------------------------------------------
        if question.startswith("/"):
            command, _, argument = question[1:].partition(" ")
            command = command.lower()

            if command in {"exit", "quit", "q"}:
                print("Bye.")
                return
            if command in {"help", "h", "?"}:
                print(HELP.format(k=bot.settings.top_k, stream="on" if stream else "off"))
            elif command == "sources":
                _print_sources(last, full=True)
            elif command == "context":
                print(last_context or "No context yet — ask a question first.")
            elif command == "stats":
                stats = bot.stats()
                print(f"\n{stats['chunks']} chunks indexed")
                for label, key in (("Documents", "files"), ("Types", "doc_types"), ("Courses", "courses")):
                    if stats[key]:
                        print(f"  {label}:")
                        for name, count in sorted(stats[key].items(), key=lambda kv: -kv[1]):
                            print(f"    {count:5d}  {name}")
            elif command == "k":
                try:
                    value = int(argument)
                    if value < 1:
                        raise ValueError
                except ValueError:
                    print("Usage: /k 5")
                else:
                    # Settings is frozen, so rebuild the dependent objects.
                    bot.settings = replace(bot.settings, top_k=value)
                    bot.retriever.settings = bot.settings
                    print(f"Now retrieving top-{value} chunks.")
            elif command == "stream":
                stream = not stream
                print(f"Streaming {'on' if stream else 'off'}.")
            elif command == "clear":
                history.clear()
                print("Conversation history cleared.")
            else:
                print(f"Unknown command: /{command}. Try /help")
            continue

        # ---- a real question -----------------------------------------------
        try:
            if stream:
                last = _answer_streaming(bot, question, history)
                last_context = format_context_of(bot, last)
            else:
                last = bot.ask(question, history)
                last_context = format_context_of(bot, last)
                print(f"\nSyllabusBot: {last.answer}")
        except SyllabusBotError as exc:  # missing key / missing package
            print(f"\nERROR: {exc}")
            return
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        except Exception as exc:  # noqa: BLE001 — keep the session alive
            logging.getLogger(__name__).exception("Turn failed")
            print(f"\nERROR: {type(exc).__name__}: {exc}")
            continue

        if last.citations:
            print("\nSources: " + "; ".join(cite["label"] for cite in last.citations))
        if show_context:
            print("\n--- retrieved context ---\n" + last_context)

        history.append((question, last.answer))


def format_context_of(bot: SyllabusBot, answer: Answer) -> str:
    """Re-render the context block for /context (no extra retrieval)."""
    from syllabusbot.retriever import format_docs

    return format_docs(answer.documents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m syllabusbot.cli",
        description="Ask grounded questions about your university PDFs.",
    )
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("-k", "--top-k", type=int, help="chunks to retrieve (default 3)")
    parser.add_argument("--no-stream", action="store_true", help="disable token streaming")
    parser.add_argument("--show-context", action="store_true", help="print retrieved context")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    enable_utf8_stdout()  # citations contain '·'

    settings = get_settings()
    if args.top_k:
        settings = replace(settings, top_k=args.top_k)

    try:
        bot = SyllabusBot(settings, streaming=not args.no_stream)
    except SyllabusBotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.question:  # non-interactive mode, for scripts and smoke tests
        answer = bot.ask(args.question)
        print(answer.answer)
        print("\nSources:\n" + answer.pretty_sources())
        if args.show_context:
            print("\n--- retrieved context ---\n" + format_context_of(bot, answer))
        return 0 if answer.grounded else 3

    print(BANNER)
    if not _print_header(settings, bot):
        return 1
    chat_loop(bot, stream=not args.no_stream, show_context=args.show_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

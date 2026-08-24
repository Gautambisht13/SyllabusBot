"""Error types.

Both failure modes below are first-run problems (a forgotten `pip install`, a
missing API key), so each carries the exact command that fixes it. Entry points
catch `SyllabusBotError` and print `str(exc)` instead of a traceback.
"""

from __future__ import annotations


class SyllabusBotError(RuntimeError):
    """Base class for errors we expect and can explain."""


class MissingCredentialsError(SyllabusBotError):
    """An API key the selected provider needs is not set."""


class MissingDependencyError(SyllabusBotError):
    """An optional integration package for the selected provider isn't installed."""

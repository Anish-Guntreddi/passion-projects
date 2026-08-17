from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CORPUS_PATH = PROJECT_ROOT / "examples" / "alice_in_wonderland.txt"

TINY_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog again! "
    "THE QUICK BROWN FOX. 1234 5678 hello, world -- hello, world.\n"
    "the quick brown fox jumps over the lazy dog once more, for good measure.\n"
)


@pytest.fixture()
def tiny_corpus() -> str:
    """A small, repetitive synthetic corpus for fast unit tests."""
    return TINY_CORPUS


@pytest.fixture(scope="session")
def example_corpus_text() -> str:
    """The committed public-domain example corpus (see examples/README.md)."""
    return EXAMPLE_CORPUS_PATH.read_text(encoding="utf-8")

import pytest
from app.services.semantic_chunker import chunk_document
from app.services.document_parser import ParsedDocument


def test_chunk_single_short_section():
    parsed = ParsedDocument(
        title="doc",
        sections=[{"level": 1, "title": "Intro", "content": "Short text."}],
    )
    chunks = chunk_document(parsed, max_chunk_chars=100)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Short text."
    assert chunks[0]["title_path"] == "Intro"
    assert chunks[0]["chunk_index"] == 0


def test_chunk_long_section_splits():
    parsed = ParsedDocument(
        title="doc",
        sections=[{"level": 1, "title": "Long", "content": "A.\nB.\nC.\nD."}],
    )
    chunks = chunk_document(parsed, max_chunk_chars=5)
    assert len(chunks) >= 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1


def test_chunk_title_path_nested():
    parsed = ParsedDocument(
        title="doc",
        sections=[
            {"level": 1, "title": "Ch1", "content": "c1"},
            {"level": 2, "title": "Sec1", "content": "c2"},
        ],
    )
    chunks = chunk_document(parsed, max_chunk_chars=100)
    assert len(chunks) == 2
    assert chunks[0]["title_path"] == "Ch1"
    assert chunks[1]["title_path"] == "Ch1 > Sec1"

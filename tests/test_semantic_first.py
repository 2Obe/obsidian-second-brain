"""Bounded semantic-first search: one embedding, candidate files only."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "integrations" / "obsidian-mcp-server"))

import vault_ops  # noqa: E402


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    ops = importlib.reload(vault_ops)
    path = tmp_path / "vault"
    path.mkdir()
    monkeypatch.setenv(ops._VAULT_ENV, str(path))
    return path, ops


def _index(vault: Path, notes: dict) -> None:
    payload = {"format": 2, "model": "fake", "notes": notes}
    (vault / vault_ops._SEMANTIC_INDEX_FILE).write_text(json.dumps(payload), encoding="utf-8")


def test_semantic_first_embeds_once_and_reads_only_candidates(vault, monkeypatch):
    path, ops = vault
    (path / "answer.md").write_text("The preferred home base is Valencia.\n", encoding="utf-8")
    (path / "other.md").write_text("An unrelated note.\n", encoding="utf-8")
    _index(path, {
        "answer.md": {"title": "Answer", "vecs": [[1.0, 0.0]]},
        "other.md": {"title": "Other", "vecs": [[0.0, 1.0]]},
    })

    embeddings = []
    monkeypatch.setattr(ops, "_embed_query",
                        lambda query, **_: embeddings.append(query) or [1.0, 0.0])
    monkeypatch.setattr(
        ops, "_iter_notes",
        lambda *_: (_ for _ in ()).throw(AssertionError("semantic-first scanned the vault")),
    )
    reads = []
    real_read = ops._read_safe

    def read_candidate(note, **kwargs):
        reads.append(note.relative_to(path).as_posix())
        return real_read(note, **kwargs)

    monkeypatch.setattr(ops, "_read_safe", read_candidate)

    results = ops.search("where should I settle", limit=1, semantic="semantic-first")

    assert embeddings == ["where should I settle"]
    assert reads == ["answer.md"]
    assert results == [{
        "path": "answer.md",
        "title": "Answer",
        "snippet": "The preferred home base is Valencia.",
    }]


def test_semantic_first_never_falls_back_without_an_index(vault, monkeypatch):
    _, ops = vault
    monkeypatch.setattr(
        ops, "_iter_notes",
        lambda *_: (_ for _ in ()).throw(AssertionError("semantic-first fell back to lexical")),
    )

    with pytest.raises(RuntimeError, match="requires a semantic index"):
        ops.search("conceptual question", semantic="semantic-first")


def test_semantic_first_never_falls_back_when_embedding_fails(vault, monkeypatch):
    path, ops = vault
    (path / "note.md").write_text("body\n", encoding="utf-8")
    _index(path, {"note.md": {"title": "Note", "vecs": [[1.0, 0.0]]}})
    monkeypatch.setattr(ops, "_embed_query",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(
        ops, "_iter_notes",
        lambda *_: (_ for _ in ()).throw(AssertionError("semantic-first fell back to lexical")),
    )

    with pytest.raises(RuntimeError, match="down"):
        ops.search("conceptual question", semantic="semantic-first")


def test_string_modes_preserve_the_legacy_boolean_contract(vault, monkeypatch):
    path, ops = vault
    (path / "note.md").write_text("zebra facts live here\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        ops, "_semantic_fuse",
        lambda *_args, enabled=None, **_kwargs: seen.append(enabled) or None,
    )

    assert ops.search("zebra facts", semantic="lexical")
    assert ops.search("zebra facts", semantic="default")
    assert ops.search("zebra facts", semantic="hybrid")
    assert seen == [False, None, True]


def test_unknown_semantic_mode_is_rejected_before_search(vault, monkeypatch):
    _, ops = vault
    monkeypatch.setattr(
        ops, "_iter_notes",
        lambda *_: (_ for _ in ()).throw(AssertionError("invalid mode started a search")),
    )

    with pytest.raises(ValueError, match="semantic-first"):
        ops.search("anything", semantic="fast-ish")

"""Behavioral tests for source discovery and log-backed ingestion recovery.

These exercise the real helper, not an LLM's extraction or consolidation quality.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import ingest_batch as batch


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "Sources").mkdir()
    (tmp_path / "raw").mkdir()
    return tmp_path


def source(vault, name="note.md", content=b"A source.\r\n"):
    path = vault / "Sources" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def capture(vault, path, name="capture.md"):
    raw = vault / "raw" / name
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("---\ntype: source\nsource_path: " + json.dumps(batch.identity(vault, path))
                   + "\nsource_file_hash: " + batch.digest(path)
                   + "\ncontent_hash: abcdef0123456789\ncapture_scope: full-local\n"
                   "ai-first: true\n---\n\nA retained source.\n", encoding="utf-8")
    return raw.relative_to(vault).as_posix()


def start(vault, path, **kwargs):
    return batch.record(vault, {"source_path": batch.identity(vault, path),
                               "source_file_hash": batch.digest(path),
                               "status": "started", **kwargs})


def finish(vault, event, raw, status="completed", **kwargs):
    return batch.record(vault, {**event, "status": status, "raw_path": raw, **kwargs})


def next_batch(vault, **kwargs):
    return batch.plan(vault, vault / "Sources", allow=["Sources"], **kwargs)


@pytest.mark.parametrize("suffix,kind", batch.FORMATS.items())
def test_supported_formats_are_discovered_and_hashed_as_bytes(vault, suffix, kind):
    data = b"\xef\xbb\xbf\x00\xffbinary\r\n"
    path = source(vault, "sample" + suffix.upper(), data)
    result = next_batch(vault)
    assert result["batch"] == [{"source_path": "Sources/" + path.name, "format": kind,
                                "source_file_hash": hashlib.sha256(data).hexdigest(),
                                "raw_path": None}]
    assert path.read_bytes() == data
    assert not (vault / "log.md").exists()


def test_discovery_orders_and_prunes_without_reading_bodies(vault, monkeypatch):
    for name in ("z.txt", "A.pdf", "a.md", "nested/b.mp3", "x.docx", ".hidden.md",
                 "Templates/t.md", "raw/generated.md", "Private/secret.md", "Skip/a.md"):
        source(vault, name)
    (vault / ".vault-config.json").write_text(json.dumps({
        "exclude-dirs": ["Skip"], "exclude-paths": ["Sources/Private"],
        "exclude-link-scan": ["Sources/z.txt"],
    }), encoding="utf-8-sig")
    monkeypatch.setattr(batch, "digest", lambda _: pytest.fail("Discovery hashed a file"))
    result = batch.discover(vault, vault / "Sources")
    assert [e["source_path"] for e in result["candidates"]] == [
        "Sources/a.md", "Sources/A.pdf", "Sources/nested/b.mp3", "Sources/z.txt"]
    assert len(result["excluded"]) == 6


def test_permissions_are_applied_before_hashing(vault, monkeypatch):
    source(vault, "Private/a.md")
    public = source(vault, "public.txt")
    original = batch.digest
    read = []

    def track(path):
        read.append(path)
        return original(path)

    monkeypatch.setattr(batch, "digest", track)
    denied = batch.plan(vault, vault / "Sources")
    assert denied["batch"] == [] and len(denied["blocked"]) == 2
    result = next_batch(vault, block=["Sources/Private"])
    assert read == [public]
    assert result["blocked"][0]["source_path"] == "Sources/Private/a.md"


def test_batches_skip_finished_sources_and_preserve_originals(vault):
    before = {source(vault, f"{i}.md"): b"A source.\r\n" for i in range(3)}
    assert len(next_batch(vault, limit=2)["batch"]) == 2
    first = next(iter(before))
    finish(vault, start(vault, first), capture(vault, first))
    result = next_batch(vault, limit=2)
    assert result["skipped"][0]["source_path"] == "Sources/0.md"
    assert len(result["batch"]) == result["remaining"] == 2
    assert batch.receipts(vault)["Sources/0.md"]["status"] == "completed"
    assert all(path.read_bytes() == value for path, value in before.items())
    assert set(p.name for p in vault.iterdir()) == {"Sources", "raw", "log.md"}


def test_interruption_after_capture_requires_resume_and_reuses_capture(vault):
    path = source(vault, "recording.wav", b"\x00\xffaudio")
    event = start(vault, path)
    raw = capture(vault, path)
    with pytest.raises(ValueError, match="already started"):
        next_batch(vault)
    result = next_batch(vault, resume=True)
    assert result["batch"][0]["raw_path"] == raw
    resumed = start(vault, path, resume=True, raw_path=raw)
    with pytest.raises(ValueError, match="active source attempt"):
        finish(vault, event, raw)
    finish(vault, resumed, raw)
    assert next_batch(vault)["batch"] == []
    assert len(list((vault / "raw").glob("*.md"))) == 1


def test_capture_without_completion_is_not_skipped(vault):
    path = source(vault)
    raw = capture(vault, path)
    (vault / "log.md").write_text("**10:00** - ingest | note - 1 created\n", encoding="utf-8")
    result = next_batch(vault)
    assert result["batch"][0]["raw_path"] == raw
    assert result["skipped"] == []


def test_unpublished_capture_is_not_reused_until_atomic_publish(vault):
    path = source(vault)
    event = start(vault, path)
    temporary = capture(vault, path, "capture.tmp")
    assert next_batch(vault, resume=True)["batch"][0]["raw_path"] is None
    with pytest.raises(ValueError, match="existing source capture"):
        finish(vault, event, temporary)
    (vault / temporary).rename(vault / "raw/capture.md")
    assert next_batch(vault, resume=True)["batch"][0]["raw_path"] == "raw/capture.md"
    finish(vault, event, "raw/capture.md")
    assert next_batch(vault)["skipped"]


def test_changed_version_is_selected_and_older_captures_are_immutable(vault):
    path = source(vault)
    raw = capture(vault, path, "old.md")
    old_bytes = (vault / raw).read_bytes()
    finish(vault, start(vault, path), raw)
    path.write_bytes(b"Changed source.")
    assert next_batch(vault)["batch"][0]["raw_path"] is None
    new_raw = capture(vault, path, "new.md")
    finish(vault, start(vault, path), new_raw)
    assert (vault / raw).read_bytes() == old_bytes
    assert next_batch(vault)["batch"] == []


@pytest.mark.parametrize("status", ["failed", "blocked"])
def test_failures_and_missing_capabilities_are_retryable(vault, status):
    path = source(vault, "audio.mp3")
    finish(vault, start(vault, path), None, status, error="Transcription unavailable")
    result = next_batch(vault)
    if status == "failed":
        assert result["batch"] == []
        assert result["failed"][0]["error"] == "Transcription unavailable"
    else:
        assert result["batch"] == []
        assert result["blocked"][0]["reason"] == "Transcription unavailable"
    assert next_batch(vault, retry=True)["batch"][0]["source_path"] == "Sources/audio.mp3"


def test_pending_and_declined_approvals_never_establish_completion(vault):
    path = source(vault)
    raw = capture(vault, path)
    event = start(vault, path)
    with pytest.raises(ValueError, match="no unresolved work"):
        finish(vault, event, raw, pending=["wiki/concepts/X.md: rewrite declined"])
    finish(vault, event, raw, "pending_approval", pending=["wiki/concepts/X.md: rewrite declined"])
    result = next_batch(vault)
    assert result["pending"][0]["pending"] == ["wiki/concepts/X.md: rewrite declined"]
    assert result["batch"][0]["raw_path"] == raw
    assert result["skipped"] == []


def test_missing_capture_and_mid_ingest_edit_prevent_false_completion(vault):
    path = source(vault)
    event = start(vault, path)
    with pytest.raises(ValueError, match="existing source capture"):
        finish(vault, event, "raw/missing.md")
    raw = capture(vault, path)
    path.write_bytes(b"Human edit while ingest runs.")
    with pytest.raises(ValueError, match="Source changed"):
        finish(vault, event, raw)
    finish(vault, event, raw, "failed", error="Source changed during processing")
    assert next_batch(vault)["batch"]


def test_deleted_capture_makes_completed_source_eligible_again(vault):
    path = source(vault)
    raw = capture(vault, path)
    finish(vault, start(vault, path), raw)
    (vault / raw).unlink()
    assert next_batch(vault)["batch"][0]["raw_path"] is None


def test_logs_are_append_only_and_daily_receipts_survive_new_sessions(vault):
    (vault / "Logs").mkdir()
    (vault / "log.md").write_bytes(b"Daily log pointer.\r\n")
    path = source(vault)
    raw = capture(vault, path)
    start(vault, path)
    log = next((vault / "Logs").glob("*.md"))
    before = log.read_bytes()
    event = batch.receipts(vault)["Sources/note.md"]
    finish(vault, event, raw)
    assert log.read_bytes().startswith(before)
    assert (vault / "log.md").read_bytes() == b"Daily log pointer.\r\n"
    assert "## For future agent" in log.read_text(encoding="utf-8")
    assert next_batch(vault)["skipped"]


def test_torn_append_is_not_a_completion_record(vault):
    path = source(vault)
    start(vault, path)
    with (vault / "log.md").open("a", encoding="utf-8") as stream:
        stream.write(batch.MARKER + '{"v":1,"status":"completed"')
    assert batch.receipts(vault)["Sources/note.md"]["status"] == "started"
    assert next_batch(vault, resume=True)["batch"]


def test_invalid_receipts_and_exclusions_fail_visibly(vault):
    source(vault)
    (vault / "log.md").write_text(batch.MARKER + '{"v":1} -->\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid ingest-progress"):
        next_batch(vault)
    (vault / "log.md").unlink()
    (vault / ".vault-config.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        next_batch(vault)


def test_no_second_source_can_be_claimed_during_processing(vault):
    start(vault, source(vault, "a.md"))
    with pytest.raises(ValueError, match="in progress"):
        start(vault, source(vault, "b.md"), resume=True)


def test_overlap_and_invalid_batch_sizes_are_rejected(vault):
    source(vault)
    with pytest.raises(ValueError, match="overlap"):
        next_batch(vault, outputs=["Sources/derived"])
    with pytest.raises(ValueError, match="positive"):
        next_batch(vault, limit=0)


def test_external_collections_use_absolute_identity_without_first_brain(vault, tmp_path):
    external = tmp_path.parent / (tmp_path.name + "-external")
    external.mkdir()
    path = external / "photo.png"
    path.write_bytes(b"image bytes")
    result = batch.plan(vault, external, allow=[external])
    assert result["batch"][0]["source_path"] == path.resolve().as_posix()


def test_linked_sources_are_not_followed(vault, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "secret.md").write_bytes(b"secret")
    try:
        (vault / "Sources" / "link").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit creating directory symlinks")
    assert batch.discover(vault, vault / "Sources")["candidates"] == []


def test_cli_reports_json_and_uses_stdin_for_progress(vault):
    path = source(vault)
    script = Path(batch.__file__)
    command = [sys.executable, str(script), "--vault", str(vault)]
    result = subprocess.run([*command, "plan", "Sources", "--allow", "Sources"],
                            capture_output=True, text=True, check=True)
    item = json.loads(result.stdout)["batch"][0]
    result = subprocess.run([*command, "record"], input=json.dumps({**item, "status": "started"}),
                            capture_output=True, text=True, check=True)
    event = json.loads(result.stdout)
    finish(vault, event, capture(vault, path))
    result = subprocess.run([*command, "plan", "Sources", "--allow", "Sources"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["skipped"]


def test_progress_guard_rejects_a_second_process(vault):
    path = source(vault)
    script = Path(batch.__file__)
    outcome = {"source_path": "Sources/note.md", "source_file_hash": batch.digest(path),
               "status": "started"}
    with batch.log_guard(vault):
        result = subprocess.run([sys.executable, str(script), "--vault", str(vault), "record"],
                                input=json.dumps(outcome), text=True, capture_output=True)
    assert result.returncode == 1
    assert "error" in json.loads(result.stderr)
    assert not batch.receipts(vault)


def test_custom_raw_route_is_used_for_recovery(vault):
    path = source(vault)
    raw = capture(vault, path)
    custom = vault / "Evidence"
    (vault / "raw").rename(custom)
    raw = "Evidence/" + Path(raw).name
    event = batch.record(vault, {"source_path": "Sources/note.md",
                                "source_file_hash": batch.digest(path), "status": "started"},
                         raw_root="Evidence")
    batch.record(vault, {**event, "raw_path": raw, "status": "completed"}, raw_root="Evidence")
    assert next_batch(vault, raw_root="Evidence")["skipped"]


def test_bad_raw_metadata_never_leaves_a_partially_loaded_capture_index(vault):
    for name in ("a.md", "b.md"):
        path = source(vault, name)
        capture(vault, path, name)
    (vault / "raw" / "0-bad.md").write_text("---\ntype: [\n---\n", encoding="utf-8")
    result = next_batch(vault)
    assert result["batch"] == []
    assert len(result["failed"]) == 2


def test_unreadable_subtree_is_reported_instead_of_silently_omitted(vault, monkeypatch):
    def unreadable_walk(*args, onerror, **kwargs):
        onerror(PermissionError("Unreadable source subtree"))

    monkeypatch.setattr(batch.os, "walk", unreadable_walk)
    with pytest.raises(PermissionError, match="Unreadable source subtree"):
        batch.discover(vault, vault / "Sources")

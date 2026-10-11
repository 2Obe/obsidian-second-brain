"""Directory ingestion bookkeeping; extraction and vault knowledge stay with the agent.

Usage: ingest_batch.py --vault VAULT discover SOURCE
       ingest_batch.py --vault VAULT plan SOURCE --allow SOURCE --limit 10
       ingest_batch.py --vault VAULT record < outcome.json

Discover reads paths only. Plan hashes only explicitly allowed paths, after the
agent resolves sensitive-content permissions. Record takes JSON on stdin and
appends versioned receipts to the existing operation logs. No separate state file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vault_health import VaultExcludes  # noqa: E402

FORMATS = {
    **dict.fromkeys((".md", ".markdown", ".txt"), "text"),
    ".pdf": "pdf",
    **dict.fromkeys((".m4a", ".mp3", ".wav", ".ogg", ".webm"), "audio"),
    **dict.fromkeys((".png", ".jpg", ".jpeg", ".webp"), "image"),
}
OUTPUT_DIRS = {"raw", "wiki", "logs", "research", "bases", "boards"}
STATUSES = {"started", "completed", "failed", "pending_approval", "blocked"}
MARKER = "<!-- ingest-progress "
HASH = re.compile(r"[0-9a-f]{64}")


def local_path(vault: Path, name: str | Path) -> Path:
    path = Path(name)
    return path if path.is_absolute() else vault / path


def linked(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def checked(path: Path) -> Path:
    if any(linked(p) for p in (path, *path.parents)):
        raise ValueError(f"Linked paths are not supported: {path}")
    return path.resolve()


def identity(vault: Path, path: Path) -> str:
    path = checked(path)
    return (path.relative_to(vault) if path.is_relative_to(vault) else path).as_posix()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def discover(vault: Path, source: Path, outputs=()) -> dict:
    source = checked(source)
    if not source.is_dir():
        raise ValueError("The batch source must be a directory")
    for destination in outputs:
        destination = checked(local_path(vault, destination))
        if not destination.is_relative_to(vault):
            raise ValueError("Output destinations must remain inside the vault")
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError(f"Source and output destinations overlap: {destination}")
    config = checked(vault / ".vault-config.json")
    data = {}
    if config.exists():
        data = json.loads(config.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or any(
            not isinstance(data.get(key, []), list)
            or any(not isinstance(v, str) for v in data.get(key, []))
            for key in ("exclude-dirs", "exclude-paths")
        ):
            raise ValueError("Invalid ingestion exclusion configuration")
    rules = VaultExcludes(set(data.get("exclude-dirs", [])),
                         [p.strip("/") for p in data.get("exclude-paths", []) if p.strip("/")])
    candidates, excluded = [], []

    def reason(path: Path) -> str | None:
        if linked(path):
            return "linked path"
        rel = path.relative_to(vault) if path.is_relative_to(vault) else path.relative_to(source)
        if rules.skip(rel.parts, rel.as_posix()):
            return "scan exclusion"
        if any(part.lower() in OUTPUT_DIRS for part in rel.parts):
            return "generated output"
        return None

    root_reason = reason(source)
    if root_reason:
        return {"candidates": [], "excluded": [{"path": str(source), "reason": root_reason}]}

    def scan_error(error):
        raise error  # Never report an unreadable subtree as an empty collection.

    for directory, dirs, files in os.walk(source, followlinks=False, onerror=scan_error):
        for name in list(dirs):
            path = Path(directory) / name
            why = reason(path)
            if why:
                dirs.remove(name)
                excluded.append({"path": str(path), "reason": why, "directory": True})
        for name in files:
            path = Path(directory) / name
            why = reason(path)
            if not why and path.suffix.lower() not in FORMATS:
                why = "unsupported format"
            if why:
                excluded.append({"path": str(path), "reason": why})
            else:
                candidates.append({"source_path": identity(vault, path),
                                   "format": FORMATS[path.suffix.lower()]})
    candidates.sort(key=lambda item: (item["source_path"].casefold(), item["source_path"]))
    excluded.sort(key=lambda item: item["path"])
    return {"candidates": candidates, "excluded": excluded}


def receipts(vault: Path) -> dict:
    """Only our exact, versioned log lines establish progress; old prose does not."""
    latest, timestamps = {}, {}
    logs = checked(vault / "Logs")
    paths = [vault / "log.md", *sorted(logs.glob("????-??-??.md"))]
    for path in paths:
        checked(path)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.startswith(MARKER):
                continue
            if not line.endswith(" -->"):
                continue  # a torn final append never establishes completion
            event = json.loads(line[len(MARKER):-4])
            if (not isinstance(event, dict) or event.get("v") != 1
                    or event.get("status") not in STATUSES
                    or not isinstance(event.get("source_path"), str)
                    or not HASH.fullmatch(event.get("source_file_hash", ""))):
                raise ValueError(f"Invalid ingest-progress entry in {path}")
            stamp = datetime.fromisoformat(event["timestamp"])
            if stamp.tzinfo is None:
                raise ValueError(f"Missing timezone in ingest-progress entry: {path}")
            name = event["source_path"]
            if name not in timestamps or stamp >= timestamps[name]:
                latest[name], timestamps[name] = event, stamp
    return latest


def raw_metadata(path: Path) -> dict:
    """Read only frontmatter, never a raw source body during discovery."""
    with path.open(encoding="utf-8-sig") as stream:
        if stream.readline().strip() != "---":
            return {}
        lines = []
        for line in stream:
            if line.strip() == "---":
                return yaml.safe_load("".join(lines)) or {}
            lines.append(line)
            if sum(map(len, lines)) > 65536:
                raise ValueError(f"Oversized raw frontmatter: {path}")
    return {}


def raw_reference(vault: Path, value: str | None, raw_root: Path) -> str | None:
    if not value:
        return None
    path = checked(local_path(vault, value))
    if not path.is_relative_to(raw_root) or not path.is_file() or path.suffix.lower() != ".md":
        return None
    meta = raw_metadata(path)
    return identity(vault, path) if isinstance(meta, dict) and meta.get("type") == "source" else None


def plan(vault: Path, source: Path, allow=(), block=(), limit=10, retry=False,
         resume=False, outputs=(), raw_root=None) -> dict:
    if limit < 1:
        raise ValueError("Batch size must be positive")
    raw_root = checked(local_path(vault, raw_root or "raw"))
    if not raw_root.is_relative_to(vault):
        raise ValueError("Raw captures must remain inside the vault")
    scope = discover(vault, source, (*outputs, identity(vault, raw_root)))
    allowed = [checked(local_path(vault, p)) for p in allow]
    denied = [checked(local_path(vault, p)) for p in block]
    latest = receipts(vault)
    active = [e for e in latest.values() if e["status"] == "started"]
    batch, skipped, pending, failed, blocked = [], [], [], [], []
    raw_index = None
    for item in scope["candidates"]:
        path = checked(local_path(vault, item["source_path"]))
        if (not any(path.is_relative_to(p) for p in allowed)
                or any(path.is_relative_to(p) for p in denied)):
            blocked.append({**item, "reason": "read permission not granted"})
            continue
        try:
            source_hash = digest(path)
            previous = latest.get(item["source_path"], {})
            same = previous.get("source_file_hash") == source_hash
            raw = raw_reference(vault, previous.get("raw_path") if same else None, raw_root)
            item = {**item, "source_file_hash": source_hash, "raw_path": raw}
            if same and previous.get("status") == "completed" and raw:
                skipped.append(item)
                continue
            if same and previous.get("status") == "failed" and not retry:
                failed.append({**item, "error": previous.get("error")})
                continue
            if same and previous.get("status") == "blocked" and not retry:
                blocked.append({**item, "reason": previous.get("error")})
                continue
            if same and previous.get("status") == "pending_approval":
                pending.append({**item, "pending": previous.get("pending", [])})
            if not raw:
                if raw_index is None:
                    index = {}
                    for capture in sorted(raw_root.rglob("*.md")):
                        checked(capture)
                        meta = raw_metadata(capture)
                        if isinstance(meta, dict) and meta.get("type") == "source":
                            index[(meta.get("source_path"), meta.get("source_file_hash"))] = (
                                identity(vault, capture))
                    raw_index = index
                item["raw_path"] = raw_index.get((item["source_path"], source_hash))
            batch.append(item)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            failed.append({**item, "error": str(exc)})
    if active:
        active_paths = {e["source_path"] for e in active}
        if not resume or len(active) != 1 or not active_paths <= {e["source_path"] for e in batch}:
            raise ValueError("An ingestion is already started. Confirm it stopped, then use "
                             "--resume with its permitted source collection.")
        batch.sort(key=lambda item: item["source_path"] not in active_paths)
    return {**scope, "batch": batch[:limit], "remaining": len(batch), "skipped": skipped,
            "pending": pending, "failed": failed, "blocked": blocked}


def log_header(day: str) -> str:
    return (f"---\ntype: log\ndate: {day}\ntags: [log, operations]\nai-first: true\n---\n\n"
            "## For future agent\n\nAppend-only vault operation history and ingestion progress.\n\n")


@contextmanager
def log_guard(vault: Path):
    """Serialize cooperating helpers using the existing root log, not a lock file."""
    root = checked(vault / "log.md")
    with root.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            # Lock an unwritten byte so other handles can still read the log.
            stream.seek(2**31 - 1)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if root.stat().st_size == 0:
                body = log_header(datetime.now().astimezone().date().isoformat())
                if (vault / "Logs").is_dir():
                    body += "Daily entries live in `Logs/`.\n"
                stream.write(body.encode("utf-8"))
                stream.flush()
            yield
        finally:
            if os.name == "nt":
                stream.seek(2**31 - 1)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def record(vault: Path, outcome: dict, raw_root=None) -> dict:
    source = checked(local_path(vault, outcome["source_path"]))
    name = identity(vault, source)
    source_hash, status = outcome["source_file_hash"], outcome["status"]
    if not HASH.fullmatch(source_hash) or status not in STATUSES:
        raise ValueError("Invalid source hash or progress status")
    pending, error = outcome.get("pending", []), outcome.get("error")
    if not isinstance(pending, list) or any(not isinstance(p, str) for p in pending):
        raise ValueError("Pending approvals must be a list of descriptions")
    raw_root = checked(local_path(vault, raw_root or "raw"))
    if not raw_root.is_relative_to(vault):
        raise ValueError("Raw captures must remain inside the vault")
    checked(vault / "Logs")
    raw = raw_reference(vault, outcome.get("raw_path"), raw_root)
    if outcome.get("raw_path") and not raw:
        raise ValueError("Raw reference must identify an existing source capture (.md)")
    if status in {"started", "completed"} and digest(source) != source_hash:
        raise ValueError("Source changed; do not mark this version complete. Replan.")
    if status == "completed" and (not raw or pending or error):
        raise ValueError("Completion requires a raw capture and no unresolved work")
    if status == "pending_approval" and not pending:
        raise ValueError("Pending approval requires a description")
    if status in {"failed", "blocked"} and not error:
        raise ValueError("Failed/blocked progress requires a reason")
    with log_guard(vault):
        latest = receipts(vault)
        previous = latest.get(name, {})
        if status == "started":
            active = [e for e in latest.values() if e["status"] == "started"]
            if active and not (outcome.get("resume") and len(active) == 1
                               and active[0]["source_path"] == name):
                raise ValueError("Another source is in progress; finish or explicitly resume it")
            attempt = uuid4().hex
        else:
            attempt = outcome.get("attempt")
            if (previous.get("status") != "started" or not attempt
                    or previous.get("attempt") != attempt
                    or previous.get("source_file_hash") != source_hash):
                raise ValueError("Outcome must match the active source attempt")
        now = datetime.now().astimezone()
        event = {"v": 1, "timestamp": now.isoformat(), "source_path": name,
                 "source_file_hash": source_hash, "status": status, "attempt": attempt,
                 "raw_path": raw, "error": error, "pending": pending}
        target = vault / "Logs" / f"{now.date().isoformat()}.md" if (
            vault / "Logs").is_dir() else vault / "log.md"
        target = checked(target)
        old = target.read_bytes() if target.exists() else b""
        old.decode("utf-8-sig")  # refuse to append to a log we cannot decode
        payload = json.dumps(event, ensure_ascii=True, separators=(",", ":"))
        payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
        link = f" | [[{raw.removesuffix('.md')}]]" if raw else ""
        addition = (f"\n**{now:%H:%M}** - ingest-progress | {status}{link}\n"
                    f"{MARKER}{payload} -->\n")
        with target.open("ab") as stream:
            if not old:
                stream.write(log_header(now.date().isoformat()).encode("utf-8"))
            stream.write(addition.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        return event


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--raw-root", default="raw")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("discover", "plan"):
        command = sub.add_parser(action)
        command.add_argument("source", type=Path)
        command.add_argument("--output", action="append", default=[])
        if action == "plan":
            command.add_argument("--allow", action="append", default=[])
            command.add_argument("--block", action="append", default=[])
            command.add_argument("--limit", type=int, default=10)
            command.add_argument("--retry-failed", action="store_true")
            command.add_argument("--resume", action="store_true")
    sub.add_parser("record")
    args = parser.parse_args(argv)
    try:
        vault = checked(args.vault)
        if not vault.is_dir():
            raise ValueError("Vault must be an existing directory")
        if args.action == "record":
            result = record(vault, json.load(sys.stdin), args.raw_root)
        elif args.action == "discover":
            result = discover(vault, local_path(vault, args.source), args.output)
        else:
            result = plan(vault, local_path(vault, args.source), args.allow, args.block,
                          args.limit, args.retry_failed, args.resume, args.output, args.raw_root)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

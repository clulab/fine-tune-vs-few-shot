"""file, hash, and version helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """return the directory with prepared splits."""
    return Path(os.environ.get("FTFS_DATA", Path.home() / "ftfs-data")).expanduser()


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # use a different temp file for each process
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: str | Path, payload: Any) -> None:
    _atomic_write_text(Path(path), json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(Path(path), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def library_versions(names: Iterable[str] = ("torch", "transformers", "peft", "bitsandbytes",
                                              "accelerate", "vllm", "tokenizers", "unsloth")) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def commit_sha() -> str:
    if not (REPO_ROOT / ".git").exists():
        return "unknown"
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

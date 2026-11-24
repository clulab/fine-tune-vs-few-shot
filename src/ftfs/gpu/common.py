"""load specs, data, and tokenizers for training and inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ftfs.utils import REPO_ROOT, data_dir, read_jsonl


def out_dir() -> Path:
    return Path(os.environ.get("FTFS_OUT", Path.home() / "ftfs-out")).expanduser()


def load_specs(path: str | Path, worker: int | None = None, n_workers: int | None = None) -> list[dict]:
    specs = json.loads(Path(path).read_text())["specs"]
    if worker is not None and n_workers:
        specs = [s for i, s in enumerate(specs) if i % n_workers == worker]
    return specs


def claim(kind: str, run_id: str) -> bool:
    """use an atomic directory creation to reserve a run within the current slurm job."""
    root = out_dir() / "claims" / os.environ.get("SLURM_JOB_ID", "local") / kind
    root.mkdir(parents=True, exist_ok=True)
    try:
        (root / run_id).mkdir()
        return True
    except FileExistsError:
        return False


def load_records(task: str, split: str) -> list[dict]:
    """read a prepared split from `$FTFS_DATA`."""
    return read_jsonl(data_dir() / task / f"{split}.jsonl")


def records_by_id(task: str, split: str) -> dict[str, dict]:
    return {r["id"]: r for r in load_records(task, split)}


def select(pool_by_id: dict[str, dict], ids: list[str]) -> list[dict]:
    return [pool_by_id[i] for i in ids]


def load_tokenizer(model: dict, path: str | None = None) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path or model["hf_id"], revision=None if path else model["revision"])
    return tok


def check_fixtures(tokenizer: Any, model_key: str, strict: bool = True) -> None:
    """check that the tokenizer reproduces the saved prompt token ids."""
    from ftfs.prompts import prompt_token_ids

    path = REPO_ROOT / "tests" / "fixtures" / f"tokens_{model_key}.json"
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"missing token fixture {path}")
        return
    fixture = json.loads(path.read_text())
    for case in fixture["cases"]:
        got = prompt_token_ids(tokenizer, case["messages"], fixture["template_kwargs"])
        if got != case["prompt_token_ids"]:
            raise RuntimeError(f"{model_key}: chat template in this image differs from the local fixture "
                               f"(case {case['name']}); check transformers/tokenizer versions")


def done(meta_path: Path, expected: dict[str, int]) -> bool:
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get("status") == "complete" and meta.get("counts") == expected

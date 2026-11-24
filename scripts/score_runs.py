"""score completed runs and save summaries and per-item scores under `$FTFS_OUT`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ftfs.gpu.common import load_records, out_dir  # noqa: E402
from ftfs.score import score_items  # noqa: E402
from ftfs.utils import read_jsonl, write_jsonl  # noqa: E402

KEEP = ("run_id", "stage", "mode", "model_key", "task", "sample", "n", "fewshot_format", "serve_base", "adapter_run_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows, missing = [], 0
    for spec in manifest["specs"]:
        run_dir = out_dir() / "runs" / spec["run_id"]
        if not (run_dir / "meta.json").exists():
            missing += 1
            continue
        for split in spec["eval_splits"]:
            outputs = read_jsonl(run_dir / f"{split.replace('/', '__')}.jsonl")
            res = score_items(spec["task_cfg"], load_records(spec["task"], split), outputs)
            write_jsonl(out_dir() / "items" / spec["run_id"] / f"{split.replace('/', '__')}.jsonl", res["items"])
            row = {k: spec.get(k) for k in KEEP}
            rec = spec.get("recipe") or {}
            row.update(lr=rec.get("lr"), epochs=rec.get("epochs"), split=split, **res["summary"])
            rows.append(row)
    dst = out_dir() / "scores" / f"{manifest['stage']}.jsonl"
    merged = {}  # merge outputs from jobs in the same stage
    if dst.exists():
        merged = {(r["run_id"], r["split"]): r for r in read_jsonl(dst)}
    merged.update({(r["run_id"], r["split"]): r for r in rows})
    write_jsonl(dst, list(merged.values()))
    print(f"scored {len(rows)} run/splits -> {dst} ({missing} runs not finished)")


if __name__ == "__main__":
    main()

"""convert study exports to jsonl under `$FTFS_DATA`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ftfs.data import BUILDERS, summarize
from ftfs.utils import data_dir, sha256_file, write_json, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="directory of study-format exports")
    ap.add_argument("--tasks", nargs="+", choices=tuple(BUILDERS), default=list(BUILDERS))
    args = ap.parse_args()

    out_root = data_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    os.chmod(out_root, 0o700)
    manifest = {}
    for task in args.tasks:
        for key, records in BUILDERS[task](args.raw).items():
            path = out_root / task / f"{key}.jsonl"
            ids = [r["id"] for r in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{task}/{key}: duplicate ids")
            write_jsonl(path, records)
            manifest[f"{task}/{key}"] = {**summarize(records), "sha256": sha256_file(path)}
            print(f"{task}/{key:14s} {manifest[f'{task}/{key}']['n']:>6}  {manifest[f'{task}/{key}']['labels']}")
    write_json(out_root / "data_manifest.json", manifest)
    print(f"wrote {len(manifest)} files to {out_root}")


if __name__ == "__main__":
    main()

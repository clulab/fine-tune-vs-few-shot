"""write run specs for the main grid or a subset."""

from __future__ import annotations

import argparse

import yaml

from ftfs.data import DATA_SEED
from ftfs.gpu.common import load_records, out_dir
from ftfs.samples import make_sample, sizes, training_ids
from ftfs.utils import REPO_ROOT, canonical_json, data_dir, read_json, sha256_file, short_hash, write_json


def split_hashes(task, splits, reference, allow_different_data):
    hashes = {}
    for split in splits:
        key = f"{task}/{split}"
        path = data_dir() / f"{key}.jsonl"
        if not path.is_file():
            raise SystemExit(f"Missing prepared split: {path}")
        digest = sha256_file(path)
        if digest != reference[key]["sha256"] and not allow_different_data:
            raise SystemExit(f"{key} differs from the study data. Verify the input or use --allow-different-data.")
        hashes[split] = digest
    return hashes


def build_specs(args, task_setup, model_setup, recipes, reference):
    tasks = task_setup["tasks"]
    grid = task_setup["grid"]
    llms, encoders = model_setup["llms"], model_setup["encoders"]
    specs = []
    for task in args.task:
        cfg = tasks[task]
        model_modes = {}
        for key in args.model:
            if key in llms:
                supported = ("zeroshot", "fewshot", "finetune")
            else:
                supported = ("encoder",) if task in encoders[key]["tasks"] else ()
            if selected := [mode for mode in supported if mode in args.mode]:
                model_modes[key] = selected
        if not model_modes:
            continue
        train_split = f"{cfg['source']}/pool"
        domains = [cfg["source"], cfg["target"]] if args.split == "test" else [cfg["source"]]
        eval_splits = [f"{domain}/{args.split}" for domain in domains]
        hashes = split_hashes(task, [train_split, *eval_splits], reference, args.allow_different_data)
        draws = []
        if any(mode != "zeroshot" for modes in model_modes.values() for mode in modes):
            pool = load_records(task, train_split)
            draws = [make_sample(task, cfg, pool, i, grid["pool_draw"], DATA_SEED) for i in args.sample]

        for model_key, modes in model_modes.items():
            is_encoder = model_key in encoders
            model = (encoders if is_encoder else llms)[model_key]
            common = {
                "model_key": model_key, "model": model, "task": task,
                "task_cfg": {**cfg, "name": task}, "prompt_version": task_setup["prompt_version"],
                "eval_splits": eval_splits, "train_split": train_split, "data_hashes": hashes,
            }
            if "zeroshot" in modes:
                specs.append({**common, "mode": "zeroshot"})
            train_modes = [mode for mode in modes if mode != "zeroshot"]
            for draw in draws:
                for n in sizes(draw, grid["sizes"]):
                    if args.size and n not in args.size:
                        continue
                    for mode in train_modes:
                        recipe = recipes["encoder" if is_encoder else "llm"]
                        if model_key == "qwen2.5-72b" and mode == "finetune":
                            large = recipes["qwen2.5-72b"]
                            if n not in large["sizes"]:
                                continue
                            recipe = {**recipe, "lr": large["lr"]}
                        spec = {**common, "mode": mode, "sample": draw["sample"],
                                "n": n, "train_ids": training_ids(draw, n)}
                        if mode == "fewshot":
                            spec["fewshot_format"] = task_setup["fewshot_format"]
                        else:
                            spec["recipe"] = recipe
                        specs.append(spec)

    return [{**spec, "run_id": short_hash(canonical_json(spec)), "stage": "study"} for spec in specs]


def main():
    config = REPO_ROOT / "configs"
    task_setup = yaml.safe_load((config / "tasks.yaml").read_text())
    model_setup = yaml.safe_load((config / "models.yaml").read_text())
    recipes = yaml.safe_load((config / "recipe.yaml").read_text())
    reference = read_json(config / "data_manifest.json")
    models = tuple(model_setup["llms"]) + tuple(model_setup["encoders"])
    tasks = tuple(task_setup["tasks"])
    modes = ("zeroshot", "fewshot", "finetune", "encoder")
    samples = range(task_setup["grid"]["n_samples"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=models, default=models)
    parser.add_argument("--task", nargs="+", choices=tasks, default=tasks)
    parser.add_argument("--mode", nargs="+", choices=modes, default=modes)
    parser.add_argument("--size", nargs="+", type=int, help="filter nonzero example budgets")
    parser.add_argument("--sample", nargs="+", type=int, choices=samples, default=samples)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--allow-different-data", action="store_true", help="accept compatible data with different checksums")
    parser.add_argument("--output", default=out_dir() / "manifests" / "study.json")
    args = parser.parse_args()
    for name in ("model", "task", "sample"):
        setattr(args, name, tuple(dict.fromkeys(getattr(args, name))))
    specs = build_specs(args, task_setup, model_setup, recipes, reference)
    if not specs:
        parser.error("no study runs match these filters")
    write_json(args.output, {"stage": "study", "specs": specs})
    print(f"Wrote {len(specs)} run specifications to {args.output}")


if __name__ == "__main__":
    main()

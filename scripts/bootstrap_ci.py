"""bootstrap scores, paired differences, and crossovers.

resample test items and training draws together for paired comparisons.
for zero-shot, resample test items only. write summaries to `$FTFS_OUT`.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ftfs.data import norm_ws  # noqa: E402
from ftfs.score import _overlap, parse_entities, parse_label, parse_triples, score_items  # noqa: E402


def item_stats(task_cfg: dict, records: list[dict], outputs: list[dict]) -> dict[str, np.ndarray]:
    """collect per-item scores in record order."""
    by_id = {o["id"]: o for o in outputs}
    kind = task_cfg["kind"]
    if kind == "classification":
        labels = task_cfg["labels"]
        gold = np.array([labels.index(r["label"]) for r in records])
        pred = np.array([(labels.index(p) if (p := parse_label(by_id[r["id"]]["output"], labels)) is not None else -1) for r in records])
        return {"gold": gold, "pred": pred}
    cols = {k: [] for k in ("tp", "fp", "fn", "tp_p", "n_p", "tp_g", "n_g")}
    types = task_cfg.get("relation_types", [])
    for r in records:
        o = by_id[r["id"]]["output"]
        if kind == "ner":
            pred, _ = parse_entities(o)
            gold = {norm_ws(e) for e in r["label"]}
            match = _overlap
        else:
            pred, _ = parse_triples(o, types)
            gold = {(norm_ws(d), t, norm_ws(e)) for d, t, e in r["label"]}
            match = lambda a, b: a[1] == b[1] and _overlap(a[0], b[0]) and _overlap(a[2], b[2])
        tp = len(pred & gold)
        vals = (tp, len(pred) - tp, len(gold) - tp, sum(any(match(p, g) for g in gold) for p in pred), len(pred),
                sum(any(match(p, g) for p in pred) for g in gold), len(gold))
        for k, v in zip(cols, vals):
            cols[k].append(v)
    return {k: np.array(v, dtype=np.float64) for k, v in cols.items()}


def _f1(tp, fp, fn):
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)


def metrics(task_cfg: dict, S: dict[str, np.ndarray], W: np.ndarray) -> dict[str, np.ndarray]:
    """compute one value per bootstrap replicate from the item weights."""
    if task_cfg["kind"] == "classification":
        labels = task_cfg["labels"]
        f1 = {}
        for i, lab in enumerate(labels):
            tp = ((S["pred"] == i) & (S["gold"] == i)).astype(np.float64)
            fp = ((S["pred"] == i) & (S["gold"] != i)).astype(np.float64)
            fn = ((S["pred"] != i) & (S["gold"] == i)).astype(np.float64)
            f1[lab] = _f1(W @ tp, W @ fp, W @ fn)
        main = np.mean([f1[l] for l in labels], axis=0) if task_cfg["metric"] == "macro_f1" else f1[task_cfg["positive_label"]]
        return {"score": main}
    strict = _f1(W @ S["tp"], W @ S["fp"], W @ S["fn"])
    tp_p, n_p, tp_g, n_g = (W @ S[k] for k in ("tp_p", "n_p", "tp_g", "n_g"))
    p = np.divide(tp_p, n_p, out=np.zeros_like(tp_p), where=n_p > 0)
    r = np.divide(tp_g, n_g, out=np.zeros_like(tp_g), where=n_g > 0)
    lenient = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return {"score": strict, "lenient": lenient}


_W: dict[tuple[str, int], np.ndarray] = {}


def weights(split: str, n_items: int, B: int) -> np.ndarray:
    key = (split, n_items)
    if key not in _W:
        rng = np.random.default_rng(zlib.crc32(split.encode()))
        _W[key] = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=B).astype(np.float64)
    return _W[key]


def ci(v: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def sample_draws(task: str, split: str, n, k: int, B: int) -> np.ndarray:
    """draw matching training samples for both methods in each replicate."""
    return np.random.default_rng(zlib.crc32(f"{task}|{split}|{n}".encode())).integers(0, k, size=(B, k))


def summarize(groups: dict) -> list[dict]:
    """compute item-only intervals (ci_*) and item-plus-draw intervals (ci2_*)."""
    rows = []
    cell: dict[tuple, dict] = {}
    for (model, task, mode, n, split), runs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        runs = sorted(runs, key=lambda r: r["sample"] if r["sample"] is not None else -1)
        k = len(runs)
        for name in runs[0]["boot"]:
            stack = np.stack([r["boot"][name] for r in runs])  # draws by replicates
            boot = stack.mean(axis=0)
            B = stack.shape[1]
            idx = sample_draws(task, split, n, k, B)
            boot2 = stack[idx, np.arange(B)[:, None]].mean(axis=1) if k > 1 else boot
            points = [r["point"][name] for r in runs]
            lo, hi = ci(boot)
            lo2, hi2 = ci(boot2)
            rows.append({"kind": "mean", "model": model, "task": task, "mode": mode, "n": n, "split": split, "metric": name, "n_samples": k,
                         "mean": float(np.mean(points)), "ci_lo": lo, "ci_hi": hi, "ci2_lo": lo2, "ci2_hi": hi2,
                         "sd_samples": float(np.std(points, ddof=1)) if k > 1 else None})
            cell[(model, task, mode, n, split, name)] = {"stack": stack, "boot": boot, "boot2": boot2, "point": float(np.mean(points)),
                                                        "samples": [r["sample"] for r in runs]}
    for (model, task, mode, n, split, name), a in list(cell.items()):
        if mode != "finetune":
            continue
        b = cell.get((model, task, "fewshot", n, split, name))
        if b is None:
            continue
        d = a["boot"] - b["boot"]
        lo, hi = ci(d)
        assert a["samples"] == b["samples"], (model, task, n, split, a["samples"], b["samples"])
        k, B = a["stack"].shape
        idx = sample_draws(task, split, n, k, B)
        d2 = (a["stack"] - b["stack"])[idx, np.arange(B)[:, None]].mean(axis=1)
        lo2, hi2 = ci(d2)
        rows.append({"kind": "diff_ft_minus_fs", "model": model, "task": task, "n": n, "split": split, "metric": name, "diff": a["point"] - b["point"],
                     "ci_lo": lo, "ci_hi": hi, "p_gt0": float((d > 0).mean()), "ci2_lo": lo2, "ci2_hi": hi2, "p2_gt0": float((d2 > 0).mean())})
    return rows + shift_rows(cell) + crossover_rows(cell)


def crossover_rows(cell: dict) -> list[dict]:
    """find the first budget where fine-tuning leads at that budget and every larger one.

    share item and training-draw resamples across budgets. report how often no crossover occurs.
    """
    rows = []
    keys = sorted({(m, t, sp, nm) for (m, t, mode, n, sp, nm) in cell if mode == "finetune" and (m, t, "fewshot", n, sp, nm) in cell})
    for model, task, split, name in keys:
        ns = sorted(n for (m, t, mode, n, sp, nm) in cell if (m, t, mode, sp, nm) == (model, task, "finetune", split, name) and (m, t, "fewshot", n, sp, nm) in cell)
        if len(ns) < 3:
            continue
        k, B = cell[(model, task, "finetune", ns[0], split, name)]["stack"].shape
        idx = np.random.default_rng(zlib.crc32(f"{task}|{split}|crossover".encode())).integers(0, k, size=(B, k))
        D = np.stack([(cell[(model, task, "finetune", n, split, name)]["stack"] - cell[(model, task, "fewshot", n, split, name)]["stack"])
                      [idx, np.arange(B)[:, None]].mean(axis=1) for n in ns])  # budgets by replicates
        ahead_from = np.full(B, np.inf)
        stays = np.ones(B, dtype=bool)
        for i in range(len(ns) - 1, -1, -1):  # scan budgets from largest to smallest
            stays &= D[i] > 0
            ahead_from = np.where(stays, ns[i], ahead_from)
        pts = [cell[(model, task, "finetune", n, split, name)]["point"] - cell[(model, task, "fewshot", n, split, name)]["point"] for n in ns]
        point = next((ns[i] for i in range(len(ns)) if all(x > 0 for x in pts[i:])), None)
        capped = np.where(np.isinf(ahead_from), 1e12, ahead_from)  # place runs with no crossover above the largest budget
        lo, med, hi = (float(np.percentile(capped, q, method="nearest")) for q in (2.5, 50, 97.5))
        fin = lambda v: None if v >= 1e12 else v
        rows.append({"kind": "crossover", "model": model, "task": task, "split": split, "metric": name, "sizes": ns, "point_crossover": point,
                     "median": fin(med), "lo": fin(lo), "hi": fin(hi), "p_never": float(np.isinf(ahead_from).mean()),
                     "p_by_20": float((ahead_from <= 20).mean()), "p_by_40": float((ahead_from <= 40).mean()), "p_by_100": float((ahead_from <= 100).mean())})
    return rows


SOURCE = {"causal": "press/test", "negation": "i2b2/test", "disease": "cdr/test", "ade": "n2c2/test"}
# domain-shift groups
GROUP = {"causal": "large", "disease": "large", "negation": "small", "ade": "small"}


def shift_rows(cell: dict) -> list[dict]:
    """subtract the source score from the target score.

    resample items separately by domain and share training draws across domains and methods.
    give each model and task cell equal weight in group averages.
    """
    rows, vec = [], {}
    for (model, task, mode, n, split, name), s_ in cell.items():
        if name != "score" or task not in SOURCE or split != SOURCE[task]:
            continue
        tkeys = [k for k in cell if k[:4] == (model, task, mode, n) and k[4] != split and k[5] == "score"]
        if len(tkeys) != 1:
            continue
        t_ = cell[tkeys[0]]
        k, B = s_["stack"].shape
        idx = sample_draws(task, "shift", n, k, B)
        pick = lambda st: st[idx, np.arange(B)[:, None]].mean(axis=1)
        ms, mt = pick(s_["stack"]), pick(t_["stack"])
        drop, rel = mt - ms, (mt - ms) / np.maximum(ms, 1e-9)
        dp, rp = t_["point"] - s_["point"], (t_["point"] - s_["point"]) / max(s_["point"], 1e-9)
        vec[(model, task, mode, n)] = {"drop": drop, "rel": rel, "dp": dp, "rp": rp}
        (dl, dh), (rl, rh) = ci(drop), ci(rel)
        rows.append({"kind": "shift", "model": model, "task": task, "group": GROUP[task], "mode": mode, "n": n, "drop": dp, "ci2_lo": dl, "ci2_hi": dh,
                     "rel_drop": rp, "rel_ci2_lo": rl, "rel_ci2_hi": rh})
    for (model, task, mode, n), a in list(vec.items()):
        if mode != "finetune" or (model, task, "fewshot", n) not in vec:
            continue
        b = vec[(model, task, "fewshot", n)]
        d, r = a["drop"] - b["drop"], a["rel"] - b["rel"]
        rows.append({"kind": "shift_diff_ft_minus_fs", "model": model, "task": task, "group": GROUP[task], "n": n, "diff_drop": a["dp"] - b["dp"],
                     "ci2_lo": ci(d)[0], "ci2_hi": ci(d)[1], "p2_gt0": float((d > 0).mean()), "diff_rel": a["rp"] - b["rp"],
                     "rel_ci2_lo": ci(r)[0], "rel_ci2_hi": ci(r)[1]})
    pooled: dict[tuple, list] = defaultdict(list)
    for (model, task, mode, n), v in vec.items():
        if mode in ("finetune", "fewshot"):
            pooled[(GROUP[task], mode, n)].append(v)
    for (g, mode, n), vs in sorted(pooled.items()):
        d, r = np.mean([v["drop"] for v in vs], axis=0), np.mean([v["rel"] for v in vs], axis=0)
        rows.append({"kind": "shift_group", "group": g, "mode": mode, "n": n, "n_cells": len(vs), "drop": float(np.mean([v["dp"] for v in vs])),
                     "ci2_lo": ci(d)[0], "ci2_hi": ci(d)[1], "rel_drop": float(np.mean([v["rp"] for v in vs])),
                     "rel_ci2_lo": ci(r)[0], "rel_ci2_hi": ci(r)[1]})
    for (g, mode, n), vs in sorted(pooled.items()):
        if mode != "finetune" or (g, "fewshot", n) not in pooled:
            continue
        keys_ft = [k for k in vec if k[2] == "finetune" and k[3] == n and GROUP[k[1]] == g and (k[0], k[1], "fewshot", n) in vec]
        d = np.mean([vec[k]["drop"] - vec[(k[0], k[1], "fewshot", n)]["drop"] for k in keys_ft], axis=0)
        r = np.mean([vec[k]["rel"] - vec[(k[0], k[1], "fewshot", n)]["rel"] for k in keys_ft], axis=0)
        rows.append({"kind": "shift_group_diff_ft_minus_fs", "group": g, "n": n, "n_cells": len(keys_ft),
                     "diff_drop": float(np.mean([vec[k]["dp"] - vec[(k[0], k[1], "fewshot", n)]["dp"] for k in keys_ft])), "ci2_lo": ci(d)[0],
                     "ci2_hi": ci(d)[1], "p2_gt0": float((d > 0).mean()),
                     "diff_rel": float(np.mean([vec[k]["rp"] - vec[(k[0], k[1], "fewshot", n)]["rp"] for k in keys_ft])),
                     "rel_ci2_lo": ci(r)[0], "rel_ci2_hi": ci(r)[1]})
    return rows


def selftest() -> None:
    rng = np.random.default_rng(0)
    labs = ["A", "B", "C"]
    cfg = {"kind": "classification", "labels": labs, "metric": "macro_f1"}
    recs = [{"id": str(i), "label": labs[rng.integers(3)], "input": "x"} for i in range(300)]
    outs = [{"id": r["id"], "output": labs[rng.integers(3)] if rng.random() < .9 else "??", "finish_reason": "stop"} for r in recs]
    ref = score_items(cfg, recs, outs)["summary"]["score"]
    got = metrics(cfg, item_stats(cfg, recs, outs), np.ones((1, 300)))["score"][0]
    assert abs(ref - got) < 1e-12, (ref, got)
    cfg2 = {"kind": "ner", "metric": "f1"}
    words = ["alpha", "beta", "gamma", "delta", "kappa"]
    recs2 = [{"id": str(i), "input": " ".join(words), "label": [str(w) for w in rng.choice(words, rng.integers(0, 3), replace=False)]} for i in range(200)]
    outs2 = [{"id": r["id"], "output": json.dumps([str(w) for w in rng.choice(words, rng.integers(0, 3), replace=False)]), "finish_reason": "stop"} for r in recs2]
    s = score_items(cfg2, recs2, outs2)["summary"]
    m = metrics(cfg2, item_stats(cfg2, recs2, outs2), np.ones((1, 200)))
    assert abs(s["score"] - m["score"][0]) < 1e-12 and abs(s["lenient"]["f1"] - m["lenient"][0]) < 1e-12, (s, m)
    W = weights("synthetic/test", 300, 500)
    v = metrics(cfg, item_stats(cfg, recs, outs), W)["score"]
    assert ci(v)[0] < ref < ci(v)[1]
    runs = [{"boot": {"score": metrics(cfg, item_stats(cfg, recs, outs), W)["score"] + 0.05 * i}, "point": {"score": ref + 0.05 * i}, "sample": i}
            for i in range(5)]
    r = [x for x in summarize({("m", "t", "finetune", 20, "synthetic/test"): runs, ("m", "t", "fewshot", 20, "synthetic/test"): runs}) if x["kind"] == "mean"][0]
    assert r["ci2_hi"] - r["ci2_lo"] > r["ci_hi"] - r["ci_lo"], r  # resampling draws widens the interval in this example
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after this many runs")
    ap.add_argument("--stages", nargs="+", default=["study"], help="stages to include as STAGE[:mode,mode]")
    ap.add_argument("--stored", help="scores JSONL for the point-estimate check (default: $FTFS_OUT/scores/study.jsonl)")
    ap.add_argument("--roots", nargs="+", help="run directories from the same environment (default: $FTFS_OUT)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    from ftfs.gpu.common import load_records, out_dir
    from ftfs.utils import read_jsonl

    allow = {}
    for item in args.stages:
        st, _, modes = item.partition(":")
        allow[st] = set(modes.split(",")) if modes else None
    cell_stage: dict = {}
    stored = {}
    sp = Path(args.stored) if args.stored else out_dir() / "scores" / "study.jsonl"
    if sp.exists():
        stored = {(r["run_id"], r["split"]): r["score"] for r in read_jsonl(sp)}
    groups: dict = defaultdict(list)
    recs_cache: dict = {}
    worst, checked, done = 0.0, 0, 0
    roots = [Path(r) for r in args.roots] if args.roots else [out_dir()]
    for meta_p in sorted(p for root in roots for p in (root / "runs").glob("*/meta.json")):
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        spec = meta.get("spec", {})
        st = spec.get("stage")
        if st not in allow or (allow[st] is not None and spec.get("mode") not in allow[st]) or meta.get("status") not in (None, "complete"):
            continue
        for split in spec["eval_splits"]:
            f = meta_p.parent / f"{split.replace('/', '__')}.jsonl"
            if not f.exists():
                continue
            if (spec["task"], split) not in recs_cache:
                recs_cache[(spec["task"], split)] = load_records(spec["task"], split)
            recs = recs_cache[(spec["task"], split)]
            S = item_stats(spec["task_cfg"], recs, read_jsonl(f))
            W = weights(split, len(recs), args.B)
            boot = metrics(spec["task_cfg"], S, W)
            point = {k: float(v[0]) for k, v in metrics(spec["task_cfg"], S, np.ones((1, len(recs)))).items()}
            key = (meta_p.parent.name, split)
            if key in stored:
                worst, checked = max(worst, abs(stored[key] - point["score"])), checked + 1
            n = spec.get("n") if spec["mode"] != "zeroshot" else 0
            ck = (spec["model_key"], spec["task"], spec["mode"], n, split)
            if (ck, spec.get("sample")) in cell_stage:  # each cell can use several stages, with one row per draw
                raise SystemExit(f"cell {ck} sample {spec.get('sample')} comes from two stages ({cell_stage[(ck, spec.get('sample'))]}, {st}): restrict the modes per stage")
            cell_stage[(ck, spec.get("sample"))] = st
            groups[(spec["model_key"], spec["task"], spec["mode"], n, split)].append({"boot": boot, "point": point, "sample": spec.get("sample")})
        done += 1
        if args.limit and done >= args.limit:
            break
    short = {k: len(v) for k, v in groups.items() if k[2] != "zeroshot" and len(v) != 5}
    if short:
        print(f"WARNING: {len(short)} cells without exactly 5 samples, e.g. {list(short.items())[:3]}")
    rows = summarize(groups)
    dst = out_dir() / "analysis" / f"bootstrap_{args.tag}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{done} runs, {len(rows)} rows -> {dst}; point estimates vs stored scores: {checked} checked, max |diff| {worst:.2e}")


if __name__ == "__main__":
    main()

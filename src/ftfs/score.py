"""parse answers and score all methods with the same rules.

invalid class labels count as errors. invalid extraction outputs count as empty lists.
spans absent from the input can count as false positives.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from ftfs.data import norm_ws

FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
PREFIX = re.compile(r"^(?:output|answer)\s*:\s*", re.I)
THINK = re.compile(r"<think>.*?</think>", re.S)
WORD = re.compile(r"\w+")


def clean(text: str) -> str:
    text = THINK.sub("", text or "").strip()
    m = FENCE.match(text)
    if m:
        text = m.group(1).strip()
    return PREFIX.sub("", text).strip()


def parse_label(text: str, labels: list[str]) -> str | None:
    t = clean(text).strip().strip(".\"'`").strip().casefold()
    for lab in labels:
        if t == lab.casefold():
            return lab
    return None


def parse_json_list(text: str) -> tuple[list, bool]:
    try:
        val = json.loads(clean(text))
    except (json.JSONDecodeError, TypeError):
        return [], False
    return (val, True) if isinstance(val, list) else ([], False)


def parse_entities(text: str) -> tuple[set[str], bool]:
    val, ok = parse_json_list(text)
    if not ok or not all(isinstance(v, str) for v in val):
        return set(), False
    return {norm_ws(v) for v in val if v.strip()}, True


def parse_triples(text: str, types: list[str]) -> tuple[set[tuple[str, str, str]], bool]:
    val, ok = parse_json_list(text)
    if not ok:
        return set(), False
    out = set()
    for rel in val:
        if not (isinstance(rel, list) and len(rel) == 3 and all(isinstance(x, str) for x in rel)):
            return set(), False
        drug, typ, ent = rel
        if typ not in types:
            return set(), False
        out.add((norm_ws(drug), typ, norm_ws(ent)))
    return out, True


def prf(tp: float, fp: float, fn: float) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _overlap(a: str, b: str) -> bool:
    return bool(set(WORD.findall(a.lower())) & set(WORD.findall(b.lower())))


def _in_text(span: str, text: str) -> bool:
    return norm_ws(span).lower() in norm_ws(text).lower()


def score_items(task_cfg: dict, records: list[dict], outputs: list[dict]) -> dict[str, Any]:
    """score generation rows against prepared records by item id."""
    by_id = {o["id"]: o for o in outputs}
    missing = [r["id"] for r in records if r["id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} records have no output (e.g. {missing[:3]})")
    kind = task_cfg["kind"]
    n = len(records)
    invalid = sum_len = 0
    items: list[dict] = []
    if kind == "classification":
        labels = task_cfg["labels"]
        for r in records:
            o = by_id[r["id"]]
            pred = parse_label(o["output"], labels)
            invalid += pred is None
            sum_len += o.get("finish_reason") == "length"
            items.append({"id": r["id"], "gold": r["label"], "pred": pred})
        per_label = {}
        for lab in labels:
            tp = sum(i["pred"] == lab and i["gold"] == lab for i in items)
            fp = sum(i["pred"] == lab and i["gold"] != lab for i in items)
            fn = sum(i["pred"] != lab and i["gold"] == lab for i in items)
            per_label[lab] = prf(tp, fp, fn)
        if task_cfg["metric"] == "macro_f1":
            main = sum(v["f1"] for v in per_label.values()) / len(labels)
        else:
            main = per_label[task_cfg["positive_label"]]["f1"]
        summary = {"score": main, "accuracy": sum(i["pred"] == i["gold"] for i in items) / n,
                   "per_label": per_label, "pred_dist": dict(Counter(str(i["pred"]) for i in items))}
    else:
        strict = Counter()
        lenient = Counter()
        halluc = total_pred = 0
        types = task_cfg.get("relation_types", [])
        for r in records:
            o = by_id[r["id"]]
            sum_len += o.get("finish_reason") == "length"
            if kind == "ner":
                pred, ok = parse_entities(o["output"])
                gold = {norm_ws(e) for e in r["label"]}
                spans = lambda x: [x]
                match = lambda a, b: _overlap(a, b)
            else:
                pred, ok = parse_triples(o["output"], types)
                gold = {(norm_ws(d), t, norm_ws(e)) for d, t, e in r["label"]}
                spans = lambda x: [x[0], x[2]]
                match = lambda a, b: a[1] == b[1] and _overlap(a[0], b[0]) and _overlap(a[2], b[2])
            invalid += not ok
            total_pred += len(pred)
            halluc += sum(not all(_in_text(s, r["input"]) for s in spans(p)) for p in pred)
            tp = len(pred & gold)
            strict.update(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)
            lenient.update(tp_p=sum(any(match(p, g) for g in gold) for p in pred), n_p=len(pred),
                           tp_g=sum(any(match(p, g) for p in pred) for g in gold), n_g=len(gold))
            items.append({"id": r["id"], "tp": tp, "fp": len(pred) - tp, "fn": len(gold) - tp})
        lp = lenient["tp_p"] / lenient["n_p"] if lenient["n_p"] else 0.0
        lr = lenient["tp_g"] / lenient["n_g"] if lenient["n_g"] else 0.0
        s = prf(strict["tp"], strict["fp"], strict["fn"])
        summary = {"score": s["f1"], **s,
                   "lenient": {"precision": lp, "recall": lr, "f1": 2 * lp * lr / (lp + lr) if lp + lr else 0.0},
                   "hallucination_rate": halluc / total_pred if total_pred else 0.0, "n_pred": total_pred}
    summary.update(n_items=n, invalid_rate=invalid / n, length_rate=sum_len / n)
    return {"summary": summary, "items": items}

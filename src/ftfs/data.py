"""build records with an id, input, label, and metadata from the study exports."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATA_SEED = 20260928
WS = re.compile(r"\s+")


def norm_ws(text: str) -> str:
    """collapse whitespace to single spaces."""
    return WS.sub(" ", text).strip()


def _load(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _carve(pool: list[dict], n: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    idx = set(rng.sample(range(len(pool)), n))
    dev = [r for i, r in enumerate(pool) if i in idx]
    rest = [r for i, r in enumerate(pool) if i not in idx]
    return rest, dev


def build_causal(raw: Path) -> dict[str, list[dict]]:
    d = raw / "causal_language_use"
    press_train = _load(d / "press_releases_causal_language_use_train_1000_sent_to_class.json")
    press_test = _load(d / "press_releases_causal_language_use_test_sent_to_class.json")
    pubmed_test = _load(d / "pubmed_causal_language_use_test_sent_to_class.json")

    def recs(rows, prefix):
        return [{"id": f"{prefix}_{i}", "input": norm_ws(r["question"]), "label": r["gold_answer"], "meta": {}}
                for i, r in enumerate(rows)]

    tests = {"press": recs(press_test, "causal_press_test"), "pubmed": recs(pubmed_test, "causal_pubmed_test")}
    test_texts = {r["input"] for rs in tests.values() for r in rs}
    seen, pool = set(), []
    for r in recs(press_train, "causal_press_train"):
        if r["input"] in seen or r["input"] in test_texts:  # skip duplicates and test overlap
            continue
        seen.add(r["input"])
        pool.append(r)
    pool, dev = _carve(pool, 150, DATA_SEED)
    return {"press/pool": pool, "press/dev": dev, "press/test": tests["press"], "pubmed/test": tests["pubmed"]}


CONTEXT = re.compile(r"context:\s*(.*)$", re.S)


def build_negation(raw: Path) -> dict[str, list[dict]]:
    d = raw / "negation"
    i2b2_dev = _load(d / "i2b2" / "dev.json")  # use the i2b2 dev file as the source pool
    i2b2_test = _load(d / "i2b2" / "test_sent_to_class_name.json")
    mimic_test = _load(d / "mimic" / "test_sent_to_class_name.json")

    def recs(rows, prefix, from_question=False):
        out = []
        for i, r in enumerate(rows):
            text = r["question"]
            if from_question:  # keep the marked context
                m = CONTEXT.search(text)
                if not m:
                    raise ValueError(f"no context in negation row {i}")
                text = m.group(1)
            out.append({"id": f"{prefix}_{i}", "input": norm_ws(text), "label": r["gold_answer"].strip(), "meta": {}})
        return out

    tests = {"i2b2": recs(i2b2_test, "neg_i2b2_test"), "mimic": recs(mimic_test, "neg_mimic_test")}
    i2b2_test_texts = {r["input"] for r in tests["i2b2"]}
    pool = [r for r in recs(i2b2_dev, "neg_i2b2_train", from_question=True) if r["input"] not in i2b2_test_texts]
    pool, dev = _carve(pool, 150, DATA_SEED)
    return {"i2b2/pool": pool, "i2b2/dev": dev, "i2b2/test": tests["i2b2"], "mimic/test": tests["mimic"]}


INPUT = re.compile(r"<input>(.*?)</input>", re.S)
OUTPUT = re.compile(r"<output>(.*?)</output>", re.S)


def _ner_rec(r: dict) -> dict:
    m_in, m_out = INPUT.search(r["question"]), OUTPUT.search(r["gold_answer"])
    if not m_in or not m_out:
        raise ValueError(f"malformed NER row {r['id']}")
    ents = [norm_ws(e) for e in m_out.group(1).split("\n") if e.strip()]
    return {"id": r["id"], "input": norm_ws(m_in.group(1)), "label": ents,
            "meta": {"doc_id": r.get("doc_id"), "sent_idx": r.get("sent_idx")}}


def build_disease(raw: Path) -> dict[str, list[dict]]:
    d = raw / "ner"
    load = lambda p: [_ner_rec(r) for r in _load(p)]
    return {"cdr/pool": load(d / "cdr" / "train.jsonl"), "cdr/dev": load(d / "cdr" / "dev.jsonl"),
            "cdr/test": load(d / "cdr" / "test.jsonl"), "ehealth/test": load(d / "ehealth" / "test.jsonl")}


ADE_TEST_COUNTS = {"n2c2": 27963, "made": 10175}


def _ade_rec(r: dict) -> dict:
    triples = []
    for rel in r["output"]:  # export order is type, drug, entity
        typ, drug, ent = rel
        triples.append([norm_ws(drug), typ, norm_ws(ent)])
    doc = r["id"].rsplit("_", 1)[0]
    return {"id": r["id"], "input": norm_ws(r["input"]), "label": triples, "meta": {"doc_id": doc}}


def build_ade(raw: Path) -> dict[str, list[dict]]:
    root = raw / "relation_extraction"
    pool = [_ade_rec(r) for r in _load(root / "n2c2" / "n2c2_train.jsonl")]
    # hold out 15 documents, then sample 500 sentences from them
    docs = sorted({r["meta"]["doc_id"] for r in pool})
    held = set(random.Random(DATA_SEED).sample(docs, 15))
    dev_all = [r for r in pool if r["meta"]["doc_id"] in held]
    pool = [r for r in pool if r["meta"]["doc_id"] not in held]
    dev = sorted(random.Random(DATA_SEED).sample(dev_all, min(500, len(dev_all))), key=lambda r: r["id"])

    out = {"n2c2/pool": pool, "n2c2/dev": dev}
    for dom, expected in ADE_TEST_COUNTS.items():
        test = [_ade_rec(r) for r in _load(root / dom / "test.jsonl")]
        if len(test) != expected:
            raise ValueError(f"{dom}: expected {expected} test sentences, got {len(test)}")
        out[f"{dom}/test"] = test
    return out


BUILDERS = {"causal": build_causal, "negation": build_negation, "disease": build_disease, "ade": build_ade}


def summarize(records: list[dict]) -> dict[str, Any]:
    labels = records and records[0]["label"]
    if isinstance(labels, str):
        dist = Counter(r["label"] for r in records)
    else:
        dist = Counter("empty" if not r["label"] else "nonempty" for r in records)
    return {"n": len(records), "labels": dict(sorted(dist.items()))}

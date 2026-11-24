"""train and evaluate the bioclinical modernbert baseline.

use sequence classification for causal and negation and bio tagging for disease mentions.
write predictions in the format used by the shared scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import traceback
from pathlib import Path

from ftfs.gpu.common import claim, done, load_records, load_specs, out_dir, records_by_id, select
from ftfs.utils import commit_sha, library_versions, write_json, write_jsonl

BIO = ["O", "B", "I"]


def char_spans(text: str, entities: list[str]) -> list[tuple[int, int]]:
    spans = []
    for e in sorted(set(entities), key=len, reverse=True):
        for m in re.finditer(re.escape(e), text):
            if not any(m.start() < b and a < m.end() for a, b in spans):
                spans.append((m.start(), m.end()))
    return sorted(spans)


def trim_offsets(text: str, offsets) -> list[tuple[int, int]]:
    """trim the spaces that byte-level token offsets can include at word boundaries."""
    out = []
    for a, b in offsets:
        if a == b:
            out.append((a, b))
            continue
        piece = text[a:b]
        lead, trail = len(piece) - len(piece.lstrip()), len(piece) - len(piece.rstrip())
        out.append((a + lead, b - trail) if lead + trail < len(piece) else (a, b))
    return out


def bio_labels(offsets, spans) -> list[int]:
    labels = []
    for a, b in offsets:
        if a == b:
            labels.append(-100)  # ignore special tokens in the loss
            continue
        tag = 0
        for s, e in spans:
            if a >= s and b <= e:
                tag = 1 if a == s else 2
                break
        labels.append(tag)
    return labels


def decode_spans(text: str, offsets, tags) -> list[str]:
    ents, cur = [], None
    for (a, b), t in zip(offsets, tags):
        if a == b:
            continue
        if t == 1 or (t == 2 and cur is None):
            if cur:
                ents.append(cur)
            cur = [a, b]
        elif t == 2 and cur is not None:
            cur[1] = b
        else:
            if cur:
                ents.append(cur)
            cur = None
    if cur:
        ents.append(cur)
    return [text[a:b].strip() for a, b in ents if text[a:b].strip()]


def run_one(spec: dict) -> None:
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    run_dir = out_dir() / "runs" / spec["run_id"]
    counts = {s: len(load_records(spec["task"], s)) for s in spec["eval_splits"]}
    if done(run_dir / "meta.json", counts):
        print(f"[skip] {spec['run_id']}")
        return
    enc, task_cfg, recipe = spec["model"], spec["task_cfg"], spec["recipe"]
    seed = spec["sample"]
    random.seed(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(enc["hf_id"], revision=enc["revision"])
    ner = task_cfg["kind"] == "ner"
    labels = BIO if ner else task_cfg["labels"]
    cls = AutoModelForTokenClassification if ner else AutoModelForSequenceClassification
    model = cls.from_pretrained(enc["hf_id"], revision=enc["revision"], num_labels=len(labels)).cuda()

    def encode(r: dict, with_labels: bool):
        e = tok(r["input"], truncation=False, return_offsets_mapping=ner)
        # evaluation can use the encoder's full context for long ehealth records
        limit = recipe["max_len"] if with_labels else model.config.max_position_embeddings
        if len(e["input_ids"]) > limit:
            raise ValueError(f"{r['id']}: {len(e['input_ids'])} tokens > {limit}; never truncating")
        item = {"input_ids": e["input_ids"]}
        if ner:
            item["offsets"] = trim_offsets(r["input"], e["offset_mapping"])
            if with_labels:
                item["labels"] = bio_labels(item["offsets"], char_spans(r["input"], r["label"]))
        elif with_labels:
            item["labels"] = labels.index(r["label"])
        return item

    def batchify(items, with_labels):
        width = max(len(i["input_ids"]) for i in items)
        ids = torch.full((len(items), width), tok.pad_token_id)
        mask = torch.zeros((len(items), width), dtype=torch.long)
        for k, i in enumerate(items):
            ids[k, :len(i["input_ids"])] = torch.tensor(i["input_ids"])
            mask[k, :len(i["input_ids"])] = 1
        out = {"input_ids": ids.cuda(), "attention_mask": mask.cuda()}
        if with_labels:
            if ner:
                lab = torch.full((len(items), width), -100)
                for k, i in enumerate(items):
                    lab[k, :len(i["labels"])] = torch.tensor(i["labels"])
            else:
                lab = torch.tensor([i["labels"] for i in items])
            out["labels"] = lab.cuda()
        return out

    train = [encode(r, True) for r in select(records_by_id(spec["task"], spec["train_split"]), spec["train_ids"])]
    bs = recipe["batch_size"]
    steps = math.ceil(len(train) / bs) * recipe["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe.get("weight_decay", 0.01))
    sched = get_linear_schedule_with_warmup(opt, int(recipe["warmup_ratio"] * steps), steps)
    rng, losses, t0 = random.Random(seed), [], time.time()
    model.train()
    for _ in range(recipe["epochs"]):
        order = list(range(len(train)))
        rng.shuffle(order)
        for s in range(0, len(order), bs):
            loss = model(**batchify([train[i] for i in order[s:s + bs]], True)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(loss.item(), 5))
    model.eval()
    for split in spec["eval_splits"]:
        records = load_records(spec["task"], split)
        items = [encode(r, False) for r in records]
        outs = [None] * len(records)
        order = sorted(range(len(records)), key=lambda k: len(items[k]["input_ids"]))
        s = 0
        while s < len(order):  # batch by token budget for long records
            e = s + 1
            while e < len(order) and e - s < 64 and (e - s + 1) * len(items[order[e]]["input_ids"]) <= 16384:
                e += 1
            idx = order[s:e]
            with torch.no_grad():
                logits = model(**batchify([items[k] for k in idx], False)).logits
            for k, lg in zip(idx, logits):
                it = items[k]
                if ner:
                    tags = lg[: len(it["input_ids"])].argmax(-1).tolist()
                    outs[k] = json.dumps(decode_spans(records[k]["input"], it["offsets"], tags), ensure_ascii=False)
                else:
                    outs[k] = labels[int(lg.argmax(-1))]
            s = e
        rows = [{"id": r["id"], "output": o, "prompt_tokens": len(it["input_ids"]), "output_tokens": 0, "finish_reason": "stop"}
                for r, o, it in zip(records, outs, items)]
        write_jsonl(run_dir / f"{split.replace('/', '__')}.jsonl", rows)
    write_json(run_dir / "meta.json", {"status": "complete", "counts": counts, "run_id": spec["run_id"],
                                       "losses": losses, "seconds": round(time.time() - t0, 1),
                                       "commit": commit_sha(), "libraries": library_versions(), "spec": spec})
    print(f"[done] {spec['run_id']} encoder {spec['task']} n={spec['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--worker", type=int)
    ap.add_argument("--n-workers", type=int)
    ap.add_argument("--queue", action="store_true", help="claim specs from the shared queue (any number of workers)")
    args = ap.parse_args()
    failed = 0
    for spec in load_specs(args.specs, None if args.queue else args.worker, args.n_workers):
        if spec["mode"] == "encoder" and (not args.queue or claim("encoder", spec["run_id"])):
            try:
                run_one(spec)
            except Exception:
                failed += 1
                traceback.print_exc()
                print(f"[FAILED] {spec['run_id']} ({spec['task']} n={spec['n']})", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

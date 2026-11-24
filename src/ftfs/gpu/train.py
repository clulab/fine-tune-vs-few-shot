"""train qlora adapters from run specs using the answer-only loss mask."""

from __future__ import annotations

import argparse
import gc
import math
import random
import sys
import time
import traceback
from pathlib import Path

from ftfs.gpu.common import claim, check_fixtures, done, load_specs, load_tokenizer, out_dir, records_by_id, select
from ftfs.prompts import messages_finetune, train_example
from ftfs.utils import commit_sha, library_versions, write_json


def build_model(model_cfg: dict, recipe: dict):
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = transformers.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=model_cfg.get("skip_quant_modules"))
    cls = (transformers.AutoModelForImageTextToText if model_cfg["loader"] == "image_text_to_text"
           else transformers.AutoModelForCausalLM)
    model = cls.from_pretrained(model_cfg["hf_id"], revision=model_cfg["revision"], quantization_config=bnb,
                                dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = model_cfg.get("lora_target_regex") or recipe["lora"]["target_modules"]
    lora = LoraConfig(r=recipe["lora"]["r"], lora_alpha=recipe["lora"]["alpha"],
                      lora_dropout=recipe["lora"]["dropout"], bias="none", task_type="CAUSAL_LM",
                      target_modules=targets)
    return get_peft_model(model, lora)


def collate(batch: list[dict], pad_id: int, gemma: bool):
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    out = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    if gemma:
        out["token_type_ids"] = torch.zeros_like(ids)  # text-only inputs
    return out


def log_mem(tag: str) -> None:
    """log free, allocated, and reserved gpu memory per run."""
    import torch
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[mem] {tag}: process allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, reserved "
          f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB, GPU free {free_b / 2**30:.1f} of {total_b / 2**30:.1f} GiB", flush=True)


def train_one(spec: dict) -> None:
    if spec.get("adapter_run_id"):  # this spec reuses an existing adapter
        return
    import torch
    from transformers import get_cosine_schedule_with_warmup

    run_dir = out_dir() / "adapters" / spec["run_id"]
    if done(run_dir / "train_meta.json", {"n_train": spec["n"]}):
        print(f"[skip] {spec['run_id']} already trained")
        return
    log_mem(f"start {spec['run_id']} n={spec['n']}")
    model_cfg, task_cfg, recipe = spec["model"], spec["task_cfg"], spec["recipe"]
    seed = spec["sample"]
    random.seed(seed)
    torch.manual_seed(seed)

    tok = load_tokenizer(model_cfg)
    check_fixtures(tok, spec["model_key"])
    pool = records_by_id(spec["task"], spec["train_split"])
    examples = []
    for r in select(pool, spec["train_ids"]):
        ex = train_example(tok, messages_finetune(task_cfg, r), model_cfg["chat_template_kwargs"],
                           model_cfg["stop_tokens"][0])
        if len(ex["input_ids"]) > recipe["max_train_tokens"]:
            raise ValueError(f"{r['id']}: {len(ex['input_ids'])} tokens > max_train_tokens; never truncating")
        examples.append(ex)

    t_load = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(model_cfg, recipe)
    load_seconds = round(time.time() - t_load, 1)
    pad_id = tok.convert_tokens_to_ids(model_cfg["pad_token"])
    gemma = model_cfg["family"] == "gemma3"
    bs = recipe["batch_size"]
    steps_per_epoch = math.ceil(len(examples) / bs)
    total = steps_per_epoch * recipe["epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=recipe["lr"], weight_decay=recipe.get("weight_decay", 0.0))
    sched = get_cosine_schedule_with_warmup(opt, int(round(recipe["warmup_ratio"] * total)), total)
    order_rng = random.Random(seed)
    losses, t0 = [], time.time()
    model.train()
    for epoch in range(recipe["epochs"]):
        order = list(range(len(examples)))
        order_rng.shuffle(order)
        for s in range(steps_per_epoch):
            batch = collate([examples[i] for i in order[s * bs:(s + 1) * bs]], pad_id, gemma)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, recipe.get("max_grad_norm", 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(loss.item(), 5))
    model.save_pretrained(run_dir)
    write_json(run_dir / "train_meta.json", {
        "status": "complete", "counts": {"n_train": spec["n"]}, "run_id": spec["run_id"],
        "steps": total, "losses": losses, "seconds": round(time.time() - t0, 1), "load_seconds": load_seconds,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "max_len": max(len(e["input_ids"]) for e in examples),
        "commit": commit_sha(), "libraries": library_versions(), "spec": spec})
    print(f"[done] {spec['run_id']} steps={total} loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    del model, opt, params, sched
    gc.collect()  # release peft reference cycles before clearing the cuda cache
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--worker", type=int)
    ap.add_argument("--n-workers", type=int)
    ap.add_argument("--queue", action="store_true", help="claim specs from the shared queue (any number of workers)")
    args = ap.parse_args()
    failed = 0
    for spec in load_specs(args.specs, None if args.queue else args.worker, args.n_workers):
        if spec["mode"] == "finetune" and (not args.queue or claim("finetune", spec["run_id"])):
            try:
                train_one(spec)
            except Exception:  # keep the worker running after a failed spec
                failed += 1
                traceback.print_exc()
                print(f"[FAILED] {spec['run_id']} ({spec['model_key']} {spec['task']} n={spec['n']})", flush=True)
        import torch
        gc.collect()  # collect after the exception releases its traceback
        torch.cuda.empty_cache()
        if spec['mode'] == 'finetune':
            log_mem('after spec (post gc + empty_cache)')
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""run greedy inference with vllm and save predictions by split.

group runs by serving base. pass token ids directly so vllm does not apply the chat template again.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import traceback
from pathlib import Path

from ftfs.gpu.common import check_fixtures, claim, done, load_records, load_specs, load_tokenizer, out_dir, records_by_id, select
from ftfs.prompts import fewshot_prompt_ids
from ftfs.utils import commit_sha, library_versions, write_json, write_jsonl


def serving_base(spec: dict) -> str:
    """use the nf4 round-trip for adapters and the original base for prompting by default."""
    return spec.get("serve_base") or ("nf4" if spec["mode"] == "finetune" else "original")


def adapter_id(spec: dict) -> str:
    """resolve the run that trained the adapter."""
    return spec.get("adapter_run_id") or spec["run_id"]


def make_llm(model_cfg: dict, model_key: str, base: str, lora: bool, gpu_mem: float = 0.90,
             kv_gb: float | None = None, max_model_len: int | None = None):
    from vllm import LLM

    from ftfs.gpu.nf4_base import base_path

    kwargs = dict(dtype="bfloat16", tensor_parallel_size=model_cfg["tensor_parallel_size"],
                  max_model_len=max_model_len or model_cfg["max_model_len"], enable_prefix_caching=True,
                  generation_config="vllm", seed=0, gpu_memory_utilization=gpu_mem)
    if kv_gb:  # explicit kv cache size skips memory profiling
        kwargs["kv_cache_memory_bytes"] = int(kv_gb * 2**30)
    if model_cfg["family"] == "gemma3":
        kwargs["limit_mm_per_prompt"] = {"image": 0}
    if lora:
        kwargs.update(enable_lora=True, max_lora_rank=16, max_loras=1)
    if base == "nf4":
        path = str(base_path(model_key))
        return LLM(model=path, tokenizer=path, **kwargs), path
    return LLM(model=model_cfg["hf_id"], revision=model_cfg["revision"], tokenizer=model_cfg["hf_id"],
               tokenizer_revision=model_cfg["revision"], **kwargs), None


def build_prompts(spec: dict, tok, records: list[dict]) -> list[list[int]]:
    task_cfg, model_cfg = spec["task_cfg"], spec["model"]
    examples = []
    if spec["mode"] == "fewshot":
        pool = records_by_id(spec["task"], spec["train_split"])
        examples = select(pool, spec["train_ids"])
    prompts = fewshot_prompt_ids(tok, task_cfg, examples, records, spec.get("fewshot_format", "multiturn"),
                                 model_cfg["chat_template_kwargs"])
    limit = model_cfg["max_model_len"] - task_cfg["max_tokens"]
    too_long = [(r["id"], len(p)) for r, p in zip(records, prompts) if len(p) > limit]
    if too_long:
        raise ValueError(f"{spec['run_id']}: {len(too_long)} prompts exceed {limit} tokens, e.g. {too_long[:3]}")
    return prompts


def run_group(specs: list[dict], queue: bool = False, gpu_mem: float = 0.90, kv_gb: float | None = None,
              max_model_len: int | None = None) -> int:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    first = specs[0]
    model_cfg, model_key, base = first["model"], first["model_key"], serving_base(first)
    pending, failed, sizes = [], 0, {}
    for spec in specs:
        for s in spec["eval_splits"]:
            if (spec["task"], s) not in sizes:  # count items once per task and split
                sizes[(spec["task"], s)] = len(load_records(spec["task"], s))
        counts = {s: sizes[(spec["task"], s)] for s in spec["eval_splits"]}
        if done(out_dir() / "runs" / spec["run_id"] / "meta.json", counts):
            print(f"[skip] {spec['run_id']}")
        else:
            pending.append((spec, counts))
    llm = tok = stop_ids = None
    lora_int = 0
    for spec, counts in pending:
        if spec["mode"] == "finetune" and not (out_dir() / "adapters" / adapter_id(spec) / "train_meta.json").exists():
            print(f"[not-ready] {spec['run_id']}: adapter missing (training failed?), not claimed", flush=True)
            failed += 1
            continue
        if queue and not claim("generate", spec["run_id"]):
            continue
        if llm is None:  # load the engine on the first claimed run
            llm, local_path = make_llm(model_cfg, model_key, base, lora=any(s["mode"] == "finetune" for s in specs), gpu_mem=gpu_mem,
                                       kv_gb=kv_gb, max_model_len=max_model_len)
            tok = load_tokenizer(model_cfg, local_path)
            check_fixtures(tok, model_key)
            stop_ids = [tok.convert_tokens_to_ids(t) for t in model_cfg["stop_tokens"]]
        lora_int += 1
        try:
            run_spec(llm, tok, stop_ids, spec, counts, base, lora_int, SamplingParams, LoRARequest)
        except Exception:  # keep the worker running after a failed spec
            failed += 1
            traceback.print_exc()
            print(f"[FAILED] {spec['run_id']} ({spec['model_key']} {spec['mode']} {spec['task']} n={spec.get('n')})",
                  flush=True)
    return failed


def run_spec(llm, tok, stop_ids, spec: dict, counts: dict, base: str, lora_int: int, SamplingParams,
             LoRARequest) -> None:
    run_dir = out_dir() / "runs" / spec["run_id"]
    ban = {tok.convert_tokens_to_ids(t): -100.0 for t in spec["model"].get("suppress_tokens", [])}
    params = SamplingParams(temperature=0.0, max_tokens=spec["task_cfg"]["max_tokens"], stop_token_ids=stop_ids,
                            repetition_penalty=1.0, skip_special_tokens=True, logit_bias=ban or None)
    lora_req = (LoRARequest(spec["run_id"], lora_int, str(out_dir() / "adapters" / adapter_id(spec)))
                if spec["mode"] == "finetune" else None)
    t0 = time.time()
    prompt_s = gen_s = 0.0
    n_prompt = n_cached = n_out = 0
    for split in spec["eval_splits"]:
        records = load_records(spec["task"], split)
        t1 = time.time()
        prompts = build_prompts(spec, tok, records)
        prompt_s += time.time() - t1
        t1 = time.time()
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=lora_req,
                            use_tqdm=False)
        gen_s += time.time() - t1
        n_prompt += sum(len(p) for p in prompts)
        n_cached += sum(o.num_cached_tokens or 0 for o in outs)
        n_out += sum(len(o.outputs[0].token_ids) for o in outs)
        rows = [{"id": r["id"], "output": o.outputs[0].text, "prompt_tokens": len(p),
                 "output_tokens": len(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason}
                for r, p, o in zip(records, prompts, outs)]
        write_jsonl(run_dir / f"{split.replace('/', '__')}.jsonl", rows)
    write_json(run_dir / "meta.json", {
        "status": "complete", "counts": counts, "run_id": spec["run_id"], "seconds": round(time.time() - t0, 1),
        "prompt_build_seconds": round(prompt_s, 1), "generate_seconds": round(gen_s, 1),
        "prompt_tokens": n_prompt, "cached_prompt_tokens": n_cached, "output_tokens": n_out,
        "serving_base": base, "commit": commit_sha(), "libraries": library_versions(), "spec": spec})
    print(f"[done] {spec['run_id']} ({spec['mode']}, n={spec.get('n')}) in {time.time() - t0:.0f}s "
          f"(prompts {prompt_s:.0f}s, generate {gen_s:.0f}s, cached {n_cached / max(n_prompt, 1):.0%})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--worker", type=int)
    ap.add_argument("--n-workers", type=int)
    ap.add_argument("--tp", type=int, required=True,
                    help="only specs whose model uses this tensor-parallel size (1: per-GPU workers, 4: whole node)")
    ap.add_argument("--base", choices=["original", "nf4"],
                    help="only specs served on this base (one engine per process: run the bases as separate processes)")
    ap.add_argument("--queue", action="store_true",
                    help="claim specs from the shared queue: any number of engines per GPU, balanced load")
    ap.add_argument("--kv-gb", type=float, help="explicit KV-cache GiB per engine (vLLM kv_cache_memory_bytes)")
    ap.add_argument("--max-model-len", type=int, help="override max_model_len (default: models.yaml)")
    ap.add_argument("--gpu-mem", type=float, default=0.90,
                    help="vLLM gpu_memory_utilization for this engine (use ~0.9/K when K engines share a GPU)")
    args = ap.parse_args()
    specs = [s for s in load_specs(args.specs, None if args.queue else args.worker, args.n_workers)
             if s["mode"] != "encoder" and s["model"]["tensor_parallel_size"] == args.tp
             and (args.base is None or serving_base(s) == args.base)]
    key = lambda s: (s["model_key"], serving_base(s))
    failed = 0
    for _, group in itertools.groupby(sorted(specs, key=key), key=key):
        failed += run_group(list(group), queue=args.queue, gpu_mem=args.gpu_mem, kv_gb=args.kv_gb,
                            max_model_len=args.max_model_len)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""build the bf16 serving base for qlora adapters.

round-trip one tensor at a time through nf4 with block size 64 and double quantization.
this keeps the full model out of gpu memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

from ftfs.gpu.common import out_dir
from ftfs.utils import library_versions, write_json

PROJ = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")


def quantized_by_training(name: str, model_cfg: dict) -> bool:
    """select the projection weights quantized during training."""
    if not PROJ.search(name):
        return False
    skip = model_cfg.get("skip_quant_modules") or ["lm_head"]
    return not any(s in name.split(".") or f".{s}." in f".{name}." for s in skip)


def base_path(model_key: str) -> Path:
    return out_dir() / "nf4_bases" / model_key


def build(model_key: str, model_cfg: dict) -> Path:
    import torch
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    src = Path(snapshot_download(model_cfg["hf_id"], revision=model_cfg["revision"], local_files_only=True))
    dst = base_path(model_key)
    if (dst / "nf4_meta.json").exists():
        print(f"[skip] {dst} exists")
        return dst
    # use an atomic directory lock for the base build
    lock = dst.with_name(dst.name + ".lock")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        print(f"[wait] another job is building {dst}")
        for _ in range(720):  # wait up to two hours for the other builder
            if (dst / "nf4_meta.json").exists():
                return dst
            time.sleep(10)
        raise TimeoutError(f"{lock} held but {dst} never appeared (stale lock? remove it and resubmit)")
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n_q = n_kept = 0
    for f in sorted(src.iterdir()):
        if f.suffix == ".safetensors":
            tensors = load_file(str(f))
            for name, w in tensors.items():
                if quantized_by_training(name, model_cfg):
                    wb = w.to(device="cuda", dtype=torch.bfloat16)
                    q, state = quantize_4bit(wb, blocksize=64, compress_statistics=True, quant_type="nf4")
                    tensors[name] = dequantize_4bit(q, state).to(torch.bfloat16).cpu()
                    n_q += 1
                else:
                    n_kept += 1
            save_file(tensors, str(tmp / f.name), metadata={"format": "pt"})
        elif f.name == "config.json":
            cfg = json.loads(f.read_text())
            cfg.pop("quantization_config", None)
            (tmp / f.name).write_text(json.dumps(cfg, indent=2))
        elif f.is_file():
            shutil.copy(f.resolve(), tmp / f.name)  # copy tokenizer and model metadata
    write_json(tmp / "nf4_meta.json", {"model_key": model_key, "source": str(src), "revision": model_cfg["revision"],
                                        "n_quantized": n_q, "n_kept": n_kept, "blocksize": 64,
                                        "double_quant": True, "libraries": library_versions()})
    missing = {f.name for f in src.glob("*.safetensors")} - {f.name for f in tmp.glob("*.safetensors")}
    if missing:
        raise RuntimeError(f"NF4 base incomplete, missing shards: {sorted(missing)}")
    tmp.rename(dst)
    shutil.rmtree(lock, ignore_errors=True)
    print(f"[done] {dst}: {n_q} tensors round-tripped, {n_kept} kept")
    return dst


def verify(model_key: str, model_cfg: dict) -> None:
    """compare the serving weights with the dequantized training model."""
    import torch
    import transformers
    from safetensors import safe_open

    bnb = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                          llm_int8_skip_modules=model_cfg.get("skip_quant_modules"))
    cls = (transformers.AutoModelForImageTextToText if model_cfg["loader"] == "image_text_to_text"
           else transformers.AutoModelForCausalLM)
    model = cls.from_pretrained(model_cfg["hf_id"], revision=model_cfg["revision"], quantization_config=bnb,
                                dtype=torch.bfloat16, device_map={"": 0})
    model.dequantize()
    ref = model.state_dict()
    ours = {}
    for f in base_path(model_key).glob("*.safetensors"):
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():
                if quantized_by_training(k, model_cfg):
                    ours[k] = fh.get_tensor(k)
    mismatched, compared = [], 0
    for k, v in ours.items():
        rk = next((c for c in (k, f"model.{k}", k.removeprefix("model.")) if c in ref), None)
        if rk is None:
            continue
        compared += 1
        if not torch.equal(ref[rk].cpu().to(torch.bfloat16), v):
            mismatched.append(k)
    print(f"verify {model_key}: compared {compared}, mismatched {len(mismatched)} {mismatched[:3]}")
    if not compared or mismatched:
        raise SystemExit(1)


def main() -> None:
    from ftfs.gpu.common import load_specs

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", help="default: every model with fine-tune specs in the job")
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    specs = [s for s in load_specs(args.specs) if s["mode"] == "finetune"]
    models = {s["model_key"]: s["model"] for s in specs}
    for key, model_cfg in models.items():
        if args.model_key and key != args.model_key:
            continue
        build(key, model_cfg)
        if args.verify:
            verify(key, model_cfg)


if __name__ == "__main__":
    main()

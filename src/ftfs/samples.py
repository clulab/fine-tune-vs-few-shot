"""draw ordered training samples shared by few-shot and fine-tuning.

start with the label-covering set and extend each draw without replacement.
each budget takes a prefix of the same draw.
"""

from __future__ import annotations

import random
from typing import Any


def _min_set(task_cfg: dict, pool: list[dict], rng_order: list[int]) -> list[int]:
    kind = task_cfg["min_set"]
    chosen: list[int] = []
    if kind == "one_per_label":
        for label in task_cfg["labels"]:
            chosen.append(next(i for i in rng_order if pool[i]["label"] == label))
    elif kind == "one_with_one_without":
        chosen.append(next(i for i in rng_order if pool[i]["label"]))
        chosen.append(next(i for i in rng_order if not pool[i]["label"]))
    elif kind == "one_per_type_plus_empty":
        for typ in task_cfg["relation_types"]:
            chosen.append(next(i for i in rng_order if i not in chosen
                               and any(t[1] == typ for t in pool[i]["label"])))
        chosen.append(next(i for i in rng_order if not pool[i]["label"]))
    else:
        raise ValueError(f"unknown min_set {kind}")
    return chosen


def make_sample(task: str, task_cfg: dict, pool: list[dict], index: int, pool_draw: int, seed: int) -> dict[str, Any]:
    rng = random.Random(f"{seed}-{task}-{index}")
    order = list(range(len(pool)))
    rng.shuffle(order)
    head = _min_set(task_cfg, pool, order)
    rest = [i for i in order if i not in set(head)][: pool_draw - len(head)]
    ids = [pool[i]["id"] for i in head + rest]
    return {"task": task, "sample": index, "seed": f"{seed}-{task}-{index}", "min_set_size": len(head), "ids": ids}


def sizes(sample: dict, grid_sizes: list[int]) -> list[int]:
    """return the minimum budget followed by the rest of the grid."""
    return [sample["min_set_size"], *grid_sizes]


def training_ids(sample: dict, n: int) -> list[str]:
    if n > len(sample["ids"]):
        raise ValueError(f"sample has {len(sample['ids'])} ids, asked for {n}")
    return sample["ids"][:n]

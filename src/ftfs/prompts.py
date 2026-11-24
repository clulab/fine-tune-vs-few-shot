"""build prompts and answer-only training targets.

zero-shot puts the instructions and input in one turn. the main few-shot format adds
demonstrations as separate turns before the test input.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

# single_cue adds the same answer cue used in the examples
FEWSHOT_FORMATS = ("multiturn", "single", "single_cue")


def render_answer(task_cfg: dict, label: Any) -> str:
    if task_cfg["kind"] == "classification":
        return str(label)
    return json.dumps(label, ensure_ascii=False)


def _input_block(x: str) -> str:
    return f"Input: {x}"


def user_turn(task_cfg: dict, x: str) -> str:
    return f"{task_cfg['instruction']}\n\n{_input_block(x)}"


def messages_finetune(task_cfg: dict, record: dict, with_answer: bool = True) -> list[dict]:
    msgs = [{"role": "user", "content": user_turn(task_cfg, record["input"])}]
    if with_answer:
        msgs.append({"role": "assistant", "content": render_answer(task_cfg, record["label"])})
    return msgs


def messages_fewshot(task_cfg: dict, examples: Sequence[dict], record: dict, fmt: str) -> list[dict]:
    if not examples:
        return messages_finetune(task_cfg, record, with_answer=False)
    if fmt == "multiturn":
        msgs: list[dict] = []
        for j, ex in enumerate(examples):
            content = user_turn(task_cfg, ex["input"]) if j == 0 else _input_block(ex["input"])
            msgs.append({"role": "user", "content": content})
            msgs.append({"role": "assistant", "content": render_answer(task_cfg, ex["label"])})
        msgs.append({"role": "user", "content": _input_block(record["input"])})
        return msgs
    if fmt in ("single", "single_cue"):
        blocks = "\n\n".join(f"{_input_block(ex['input'])}\nOutput: {render_answer(task_cfg, ex['label'])}"
                             for ex in examples)
        content = f"{task_cfg['instruction']}\n\nExamples:\n\n{blocks}\n\n{_input_block(record['input'])}"
        if fmt == "single_cue":
            content += "\nOutput:"
        return [{"role": "user", "content": content}]
    raise ValueError(f"unknown few-shot format {fmt!r}")


def _ids(out: Any) -> list[int]:
    """unwrap input ids from tokenizers that return a mapping."""
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def prompt_token_ids(tokenizer: Any, messages: list[dict], template_kwargs: dict) -> list[int]:
    return _ids(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              **template_kwargs))


_MARK = "\u241fFTFS_INPUT\u241f"


def fewshot_prompt_ids(tokenizer: Any, task_cfg: dict, examples: Sequence[dict], records: Sequence[dict],
                       fmt: str, template_kwargs: dict, verify_every: int = 25) -> list[list[int]]:
    """tokenize the shared multi-turn prefix once.

    check the first three prompts and every `verify_every`th prompt against full tokenization.
    fall back to full tokenization if the template or token ids differ.
    """
    def slow() -> list[list[int]]:
        return [prompt_token_ids(tokenizer, messages_fewshot(task_cfg, examples, r, fmt), template_kwargs)
                for r in records]

    if fmt != "multiturn" or not examples or not records:
        return slow()
    msgs = messages_fewshot(task_cfg, examples, {"input": _MARK}, fmt)
    render = lambda m, gen: tokenizer.apply_chat_template(m, add_generation_prompt=gen, tokenize=False,
                                                          **template_kwargs)
    prefix_text, full_text = render(msgs[:-1], False), render(msgs, True)
    if not full_text.startswith(prefix_text) or full_text.count(_MARK) != 1:
        return slow()
    head, tail = full_text[len(prefix_text):].split(_MARK)
    enc = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]
    prefix_ids = enc(prefix_text)
    out = [prefix_ids + enc(head + r["input"] + tail) for r in records]
    for i in sorted({0, 1, 2, *range(0, len(records), verify_every)} & set(range(len(records)))):
        exact = prompt_token_ids(tokenizer, messages_fewshot(task_cfg, examples, records[i], fmt), template_kwargs)
        if out[i] != exact:
            print(f"[warn] fast few-shot tokenization differs at record {records[i]['id']}; using the exact path",
                  flush=True)
            return slow()
    return out


def train_example(tokenizer: Any, messages: list[dict], template_kwargs: dict,
                  end_of_turn: str) -> dict[str, list[int]]:
    """mask the prompt and supervise the answer through its end-of-turn token."""
    assert messages[-1]["role"] == "assistant"
    prompt = prompt_token_ids(tokenizer, messages[:-1], template_kwargs)
    full = _ids(tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                              **template_kwargs))
    if full[: len(prompt)] != prompt:
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    eot_id = tokenizer.convert_tokens_to_ids(end_of_turn)
    answer_part = full[len(prompt):]
    if eot_id not in answer_part:
        raise ValueError(f"end-of-turn token {end_of_turn!r} not found after the prompt")
    cut = answer_part.index(eot_id) + 1  # stop supervision at the end-of-turn token
    supervised = answer_part[:cut]
    expected = messages[-1]["content"] + end_of_turn
    decoded = tokenizer.decode(supervised, skip_special_tokens=False)
    if decoded != expected:
        raise ValueError(f"supervised span decodes to {decoded!r}, expected {expected!r}")
    input_ids = prompt + supervised
    labels = [-100] * len(prompt) + supervised
    return {"input_ids": input_ids, "labels": labels}

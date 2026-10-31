#!/usr/bin/env python3
"""Collect teacher masked-token outputs (offline distillation only).

Usage:
  python tools/modernbert_teacher.py examples/modernbert_masks.jsonl traces.jsonl
Requires torch + transformers>=4.48 and downloads answerdotai/ModernBERT-base.
"""
import argparse
import hashlib
import json
from time import perf_counter_ns
from pathlib import Path


def split_id(ids: list[int], mask_index: int, validation_fraction: float,
             group_key: str | None = None) -> str:
    # A caller-supplied source group keeps related masks from the same document
    # on one side of the holdout; older callers retain token-based grouping.
    text = ('source:' + ' '.join(group_key.casefold().split())) if group_key is not None else \
           json.dumps([ids, mask_index], separators=(",", ":"))
    ratio = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") / 2**64
    return "validation" if ratio < validation_fraction else "train"



def mask_preserving_window(token_ids: list[int], mask_id: int, capacity: int) -> tuple[list[int], bool]:
    """Keep exactly one mask even when the text exceeds max sequence length."""
    if capacity < 1:
        raise ValueError("token window capacity must be positive")
    positions = [i for i, token in enumerate(token_ids) if token == mask_id]
    if len(positions) != 1:
        raise ValueError("masked context must contain exactly one mask token")
    if len(token_ids) <= capacity:
        return token_ids, False
    start = max(0, min(positions[0] - capacity // 2, len(token_ids) - capacity))
    return token_ids[start:start + capacity], True


def prepare_masked_inputs(tokenizer, rendered: str, max_length: int) -> tuple[list[int], int, bool]:
    """Build ModernBERT inputs with explicit [CLS]/[SEP] boundaries.

    Some fast-tokenizer build_inputs_with_special_tokens() implementations do
    not insert those boundaries when invoked directly.
    """
    cls, sep, mask = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id
    if any(type(v) is not int or v < 0 for v in (cls, sep, mask)):
        raise ValueError("ModernBERT tokenizer requires CLS, SEP and MASK token IDs")
    if max_length < 3:
        raise ValueError("max_length must accommodate CLS, MASK, SEP")
    content = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
    if hasattr(content, "tolist"):
        content = content.tolist()
    window, cropped = mask_preserving_window(content, mask, max_length - 2)
    ids = [cls, *window, sep]
    if not (len(ids) <= max_length and ids.count(cls) == 1 and ids.count(sep) == 1
            and ids.count(mask) == 1 and 0 < ids.index(mask) < len(ids) - 1):
        raise ValueError("invalid ModernBERT special-token structure")
    return ids, ids.index(mask), cropped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path, help='JSONL with {"text": "... [MASK] ..."}')
    p.add_argument("output", type=Path)
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--emit-hidden", action="store_true", help="Store final mask hidden state for offline semantic-signature training (large traces)")
    p.add_argument("--validation-fraction", type=float, default=0.3)
    p.add_argument("--honor-input-split", action="store_true", help="Preserve trusted corpus train/validation labels instead of hash splitting")
    p.add_argument("--max-length", type=int, default=128)
    a = p.parse_args()
    if not 0 < a.validation_fraction < 1 or not 1 <= a.topk <= 256:
        p.error("validation-fraction must be (0,1), topk must be 1..256")
    if not 8 <= a.max_length <= 8192:
        p.error("max-length must be 8..8192")

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(a.model, trust_remote_code=False)
    model.to(a.device).eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    counts = {"train": 0, "validation": 0}
    with a.inputs.open(encoding="utf-8") as inp, a.output.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(inp, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = item["text"]
            if text.count("[MASK]") != 1:
                raise ValueError(f"line {line_number}: exactly one literal [MASK] required")
            rendered = text.replace("[MASK]", tokenizer.mask_token)
            ids, pos, cropped = prepare_masked_inputs(tokenizer, rendered, a.max_length)
            enc = {"input_ids": torch.tensor([ids], dtype=torch.long),
                   "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
            key = (tuple(ids), pos)
            if key in seen:
                continue
            seen.add(key)
            inputs = {k: v.to(a.device) for k, v in enc.items()}
            if a.device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter_ns()
            with torch.no_grad():
                result = model(**inputs, output_hidden_states=a.emit_hidden)
                logits = result.logits[0, pos].float()
                probs = torch.softmax(logits, dim=-1)
                values, indices = torch.topk(probs, a.topk)
            if a.device == "cuda":
                torch.cuda.synchronize()
            teacher_forward_ms = (perf_counter_ns() - start) / 1e6
            source_group = item.get('group_id')
            if source_group is not None and (not isinstance(source_group, str) or not source_group.strip()):
                raise ValueError(f'line {line_number}: group_id must be a nonblank string')
            if a.honor_input_split:
                split = item.get("split")
                if split not in ("train", "validation"):
                    raise ValueError(f"line {line_number}: --honor-input-split requires train/validation")
            else:
                split = split_id(ids, pos, a.validation_fraction, group_key=source_group or text)
            trace = {"input_ids": ids, "mask_index": pos, "topk": [
                {"token": int(tok), "prob": float(prob)}
                for tok, prob in zip(indices.cpu().tolist(), values.cpu().tolist())
            ], "split": split, "model": a.model, "source_text": text,
                     "teacher_forward_ms": round(teacher_forward_ms, 4),
                     "teacher_device": a.device,
                     "teacher_revision": getattr(model.config, '_commit_hash', None),
                     "teacher_input_has_boundaries": True,
                     "teacher_window_cropped": cropped,
                     "teacher_cls_id": tokenizer.cls_token_id,
                     "teacher_sep_id": tokenizer.sep_token_id}
            if source_group is not None:
                trace["source_group"] = source_group
            if a.emit_hidden:
                if not result.hidden_states:
                    raise RuntimeError("teacher did not return hidden states")
                hidden = result.hidden_states[-1][0, pos].float().cpu().tolist()
                trace["mask_hidden"] = [round(float(x), 6) for x in hidden]
            out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            counts[split] += 1
    print(json.dumps({"saved": sum(counts.values()), **counts, "output": str(a.output)}))


if __name__ == "__main__":
    main()

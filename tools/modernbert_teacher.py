#!/usr/bin/env python3
"""Collect teacher masked-token outputs (offline distillation only).

Usage:
  python tools/modernbert_teacher.py examples/modernbert_masks.jsonl traces.jsonl
Requires torch + transformers>=4.48 and downloads answerdotai/ModernBERT-base.
"""
import argparse
import hashlib
import json
from pathlib import Path


def split_id(ids: list[int], mask_index: int, validation_fraction: float) -> str:
    # Same token sequence always lands in the same split, independent of row order.
    text = json.dumps([ids, mask_index], separators=(",", ":"))
    ratio = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") / 2**64
    return "validation" if ratio < validation_fraction else "train"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path, help='JSONL with {"text": "... [MASK] ..."}')
    p.add_argument("output", type=Path)
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--validation-fraction", type=float, default=0.3)
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
            enc = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=a.max_length)
            ids = enc["input_ids"][0].tolist()
            mask_positions = [i for i, token in enumerate(ids) if token == tokenizer.mask_token_id]
            if len(mask_positions) != 1:
                raise ValueError(f"line {line_number}: mask missing or truncated")
            pos = mask_positions[0]
            key = (tuple(ids), pos)
            if key in seen:
                continue
            seen.add(key)
            with torch.no_grad():
                result = model(**{k: v.to(a.device) for k, v in enc.items()})
                logits = result.logits[0, pos].float()
                probs = torch.softmax(logits, dim=-1)
                values, indices = torch.topk(probs, a.topk)
            split = split_id(ids, pos, a.validation_fraction)
            trace = {"input_ids": ids, "mask_index": pos, "topk": [
                {"token": int(tok), "prob": float(prob)}
                for tok, prob in zip(indices.cpu().tolist(), values.cpu().tolist())
            ], "split": split, "model": a.model, "source_text": text}
            out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            counts[split] += 1
    print(json.dumps({"saved": sum(counts.values()), **counts, "output": str(a.output)}))


if __name__ == "__main__":
    main()

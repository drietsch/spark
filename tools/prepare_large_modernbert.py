#!/usr/bin/env python3
"""Reproducible WikiText-2 masked-token inputs with official held-out splits."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

SOURCE = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
WORD_RE = re.compile(r"\b[A-Za-z]{3,18}\b")
SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"(])')
STOP = frozenset("the and that this with from have were they their there which would could about into over under while where what when also than then them your said been some only other".split())

def pick_mask(text: str, key: str, seed: int) -> str | None:
    """Pick a reproducible content word without looking at teacher labels."""
    if "[MASK]" in text or len(text) < 65:
        return None
    words = [m for m in WORD_RE.finditer(text) if m.group(0).lower() not in STOP]
    if len(words) < 8:
        return None
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    match = words[int.from_bytes(digest[:8], "little") % len(words)]
    return text[:match.start()] + "[MASK]" + text[match.end():]

def prepare_splits(splits: dict[str, list[str]], train_count: int, val_count: int,
                   *, seed: int = 20261031, max_chars: int = 360) -> tuple[list[dict], dict]:
    if not (train_count > 0 and val_count > 0 and 120 <= max_chars <= 2048):
        raise ValueError("train_count and val_count must be >0; max_chars in 120..2048")
    examples: list[dict] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for split, requested in (("train", train_count), ("validation", val_count)):
        if split not in splits:
            raise ValueError(f"missing {split} source split")
        saved = 0
        for line_id, paragraph in enumerate(splits[split]):
            if saved >= requested:
                break
            if not isinstance(paragraph, str):
                continue
            paragraph = " ".join(paragraph.split())
            if len(paragraph) < 65 or paragraph.lstrip().startswith("="):
                continue
            for part_id, sentence in enumerate(SENTENCE_RE.split(paragraph)):
                if saved >= requested:
                    break
                sentence = sentence.strip()
                if len(sentence) > max_chars:
                    sentence = sentence[:max_chars].rsplit(" ", 1)[0]
                masked = pick_mask(sentence, f"{split}:{line_id}:{part_id}", seed)
                if masked is None:
                    continue
                normalized = " ".join(masked.casefold().split())
                if normalized in seen:
                    continue
                seen.add(normalized)
                examples.append({"text": masked, "split": split,
                                 "group_id": f"wikitext2-{split}-line-{line_id}"})
                saved += 1
        counts[split] = saved
        if saved < requested:
            raise RuntimeError(f"WikiText yielded only {saved}/{requested} {split} masks")
    info = {"format": "spark-large-masks-v1", "dataset": SOURCE, "config": CONFIG,
            "seed": seed, "max_chars": max_chars, "examples": counts,
            "mask_policy": "deterministic SHA256 word choice, one mask per sentence",
            "source_split_policy": "original WikiText-2 train versus validation",
            "warning": "Teacher top-1 agreement is not ground-truth accuracy"}
    return examples, info

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--validation", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--max-chars", type=int, default=360)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset(SOURCE, CONFIG)
    rows, info = prepare_splits(
        {name: [row["text"] for row in ds[name]] for name in ("train", "validation")},
        args.train, args.validation, seed=args.seed, max_chars=args.max_chars)
    info["dataset_fingerprints"] = {name: ds[name]._fingerprint for name in ("train", "validation")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    args.output.write_text(data, encoding="utf-8")
    info["sha256"] = hashlib.sha256(data.encode("utf-8")).hexdigest()
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest), **info}, indent=2))

if __name__ == "__main__":
    main()

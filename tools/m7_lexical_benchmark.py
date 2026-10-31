#!/usr/bin/env python3
"""Compare WikiText-trained lexical associations and supervised SPARK on real teacher traces.

The complete WikiText-2 train split is the only source of lexical training
examples. The WikiText-2 validation split is never loaded for training or
tuning. Teacher-supervised mixing weights are selected on an inner TRAIN fold.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.m7_context import LexicalBackoff, sparse_scores, fused_predictions
from spark.associative_classifier import SparseAssoc
from spark.modernbert import read_traces
from spark.benchmark import audit_traces, teacher_top1, wilson_95_interval

WORDS = re.compile(r"\b[A-Za-z]{3,18}\b")
SENTENCES = re.compile(r'(?<=[.!?])\s+(?=[A-Z"(])')
STOP = frozenset("the and that this with from have were they their there which would could about into over under while where what when also than then them your said been some only other".split())


def wikitext_training_samples(dataset, tokenizer, *, limit: int = 550000, max_chars: int = 360):
    """Yield position-sensitive known word targets without accessing validation."""
    cls, sep, mask = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id
    if any(token is None for token in (cls, sep, mask)):
        raise ValueError("tokenizer lacks ModernBERT special tokens")
    used = 0
    for row in dataset:
        paragraph = " ".join(row["text"].split())
        if len(paragraph) < 65 or paragraph.startswith("="):
            continue
        for sentence in SENTENCES.split(paragraph):
            if len(sentence) > max_chars:
                sentence = sentence[:max_chars].rsplit(" ", 1)[0]
            if len(sentence) < 65 or "[MASK]" in sentence:
                continue
            encoded = tokenizer(sentence, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True)
            tokens, offsets = encoded["input_ids"], encoded["offset_mapping"]
            if not 9 <= len(tokens) <= 192:
                continue
            positions = {tuple(span): i for i, span in enumerate(offsets) if span[1] > span[0]}
            context = [cls, *tokens, sep]
            for match in WORDS.finditer(sentence):
                if match.group(0).casefold() in STOP:
                    continue
                pos = positions.get(match.span())
                if pos is None:
                    continue  # multilayer / multi-subword word; not a single MLM target
                target = tokens[pos]
                if target in (cls, sep, mask, tokenizer.unk_token_id):
                    continue
                context[pos + 1] = mask
                yield context, pos + 1, target
                context[pos + 1] = target
                used += 1
                if used >= limit:
                    return


def split_inner_training(rows):
    train, tune = [], []
    for row in rows:
        group = row.get("source_group") or row.get("source_text") or repr(row["input_ids"])
        bucket = int.from_bytes(hashlib.sha256(("cv:" + group).encode()).digest()[:4], "big") % 10
        (tune if bucket == 0 else train).append(row)
    if not train or not tune:
        raise ValueError("inner fold is empty")
    return train, tune


def agreement(rows, predictions):
    correct = sum(teacher_top1(row) == pred for row, pred in zip(rows, predictions))
    return {"examples": len(rows), "correct": correct,
            "teacher_top1_agreement": correct / len(rows),
            "wilson95": wilson_95_interval(correct, len(rows))}


def run(traces: Path, corpus, tokenizer, *, limit: int = 550000) -> dict:
    rows = read_traces(traces)
    audit_traces(rows)
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if not all(r["input_ids"][0] == tokenizer.cls_token_id
               and r["input_ids"][-1] == tokenizer.sep_token_id
               and r["input_ids"][r["mask_index"]] == tokenizer.mask_token_id for r in rows):
        raise ValueError("teacher trace special-token boundaries invalid")
    memory = LexicalBackoff()
    begin = time.monotonic()
    for ids, pos, token in wikitext_training_samples(corpus, tokenizer, limit=limit):
        memory.add(ids, pos, token)
        if memory.seen % 100000 == 0:
            print(f"lexical observations: {memory.seen}, features: {len(memory.postings)}",
                  file=sys.stderr, flush=True)
    if memory.seen < 100000:
        raise ValueError(f"insufficient lexical training examples: {memory.seen}")
    lexical_build_s = round(time.monotonic() - begin, 3)

    inner, tune = split_inner_training(train)
    teacher_inner = SparseAssoc.fit(inner)
    teacher_full = SparseAssoc.fit(train)
    tune_s = [sparse_scores(teacher_inner, r["input_ids"], r["mask_index"]) for r in tune]
    valid_s = [sparse_scores(teacher_full, r["input_ids"], r["mask_index"]) for r in validation]

    # Exact validation split is not touched until model and weights are frozen.
    params = [(smooth, specific, blend) for smooth in (2.0, 10.0, 40.0)
              for specific in (0.7, 1.0, 1.3)
              for blend in (0.0, 0.25, 0.5, 0.75, 1.0)]
    cached = {}
    scored = []
    for smooth, specific, blend in params:
        key = (smooth, specific)
        if key not in cached:
            cached[key] = [memory.score(r["input_ids"], r["mask_index"],
                                        smoothing=smooth, specificity=specific) for r in tune]
        predictions = [fused_predictions(l, s, lexical_weight=blend)
                       for l, s in zip(cached[key], tune_s)]
        hits = sum(pred == teacher_top1(row) for pred, row in zip(predictions, tune))
        scored.append((hits, smooth, specific, blend))
    best_hits, smoothing, specificity, blend = max(
        scored, key=lambda x: (x[0], -abs(x[3] - 0.5), -x[1], -x[2]))
    valid_l = [memory.score(r["input_ids"], r["mask_index"],
                             smoothing=smoothing, specificity=specificity) for r in validation]
    predict_only = lambda s: max(s, key=lambda t: (s[t], -t)) if s else None
    lexical_pred = [predict_only(s) for s in valid_l]
    sparse_pred = [predict_only(s) for s in valid_s]
    combined = [fused_predictions(l, s, lexical_weight=blend)
                for l, s in zip(valid_l, valid_s)]
    majority = Counter(teacher_top1(r) for r in train).most_common(1)[0][0]
    return {
        "format": "spark-m7-lexical-teacher-fusion-v1",
        "dataset": "Salesforce/wikitext / wikitext-2-raw-v1 (TRAIN ONLY)",
        "teacher": "answerdotai/ModernBERT-base",
        "training_teacher_examples": len(train), "held_out_validation": len(validation),
        "lexical_memory": {**memory.stats(), "build_seconds": lexical_build_s},
        "train_only_tuning": {"inner_train": len(inner), "inner_tune": len(tune),
                               "best_inner_correct": best_hits, "grid_size": len(params),
                               "smoothing": smoothing, "specificity": specificity,
                               "lexical_weight": blend},
        "majority_baseline": agreement(validation, [majority] * len(validation)),
        "lexical_ngram_only": agreement(validation, lexical_pred),
        "supervised_sparse_only": agreement(validation, sparse_pred),
        "multi_stage_fusion": agreement(validation, combined),
        "limitations": [
            "Teacher top-1 agreement is not ground-truth original masked word accuracy.",
            "The lexical memory is trained on original WikiText TRAIN text only.",
            "The WikiText VALIDATION split is never loaded for fitting or tuning.",
            "Sparse inference uses posting lookups and scalar votes without neural matrices.",
            "Student inference is Python reference only; no native C++ latency claims."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher_traces", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-lexical-tokens", type=int, default=550000)
    args = parser.parse_args()
    if args.max_lexical_tokens < 100000 or args.max_lexical_tokens > 2000000:
        parser.error("max-lexical-tokens must be 100000..2000000")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", trust_remote_code=False)
    corpus = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    report = run(args.teacher_traces, corpus, tokenizer, limit=args.max_lexical_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

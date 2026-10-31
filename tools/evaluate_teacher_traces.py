#!/usr/bin/env python3
"""Evaluate exact-context SPARKIDX student vs held-out teacher top-1 choices."""
import argparse
import bisect
import json
import struct
from pathlib import Path
from train_from_traces import HEADER, ENTRY, MAGIC, signature


def load_index(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("Truncated SPARK index")
    magic, major, minor, flags, buckets, count, seed, bucket_offset, offset, size = HEADER.unpack_from(data)
    if (magic, major, minor, flags, buckets, seed, bucket_offset, offset, size) != \
       (MAGIC, 1, 0, 0, 0, 0, 0, HEADER.size, len(data)):
        raise ValueError("Unsupported index header")
    if count > (len(data) - offset) // ENTRY.size or offset + count * ENTRY.size != len(data):
        raise ValueError("Invalid SPARK index length")
    entries = list(ENTRY.iter_unpack(data[offset:]))
    if entries != sorted(entries):
        raise ValueError("Index not sorted")
    return entries


def next_token(entries, context):
    # SPARK C++ runtime keeps only the last four input IDs.
    context = context[-4:]
    for n in range(len(context), 0, -1):
        key = signature(context[-n:])
        pos = bisect.bisect_left(entries, (key, 0, 0))
        best_token, best_weight = None, -1
        while pos < len(entries) and entries[pos][0] == key:
            _, tok, weight = entries[pos]
            if weight > best_weight or (weight == best_weight and tok < best_token):
                best_token, best_weight = tok, weight
            pos += 1
        if best_token is not None:
            return best_token, n
    return None, 0


def evaluate(index, holdout):
    entries = load_index(index)
    seen, matched, correct, total_order = 0, 0, 0, 0
    for line in Path(holdout).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        record = json.loads(line)
        teacher = min(record["targets"], key=lambda t: (-t["weight"], t["token"]))["token"]
        predicted, order = next_token(entries, record["context"])
        seen += 1
        if predicted is not None:
            matched += 1
            total_order += order
            correct += int(teacher == predicted)
    if not seen:
        raise ValueError("No eval traces")
    return {"held_out_traces": seen, "covered_traces": matched,
            "coverage": round(matched / seen, 4),
            "top1_agreement_when_covered": round(correct / matched, 4) if matched else None,
            "top1_agreement_all": round(correct / seen, 4),
            "mean_match_order": round(total_order / matched, 2) if matched else None,
            "warning": "Top-1 teacher agreement only; not full language-model quality."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("index", type=Path)
    p.add_argument("holdout_traces", type=Path)
    args = p.parse_args()
    print(json.dumps(evaluate(args.index, args.holdout_traces), indent=2))

if __name__ == "__main__": main()

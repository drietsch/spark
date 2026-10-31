#!/usr/bin/env python3
"""Compile teacher trace JSONL into the SPARK v0.1 *exact-context* baseline index.
Input: {"context":[10,11],"targets":[{"token":12,"weight":9},...]}.
This is NOT a general Transformer-to-SPARK converter; it tests trace ingestion.
"""
import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path

MAGIC = b"SPARKIDX"
HEADER = struct.Struct("<8sHHIQQQQQQ")  # exactly 64 bytes
ENTRY = struct.Struct("<QII")             # exactly 16 bytes
UINT32_MAX = 0xffffffff


def signature(tokens):
    h = 14695981039346656037
    for token in tokens:
        for byte in token.to_bytes(4, "little"):
            h ^= byte
            h = (h * 1099511628211) & 0xffffffffffffffff
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-order", type=int, choices=(1,2,3,4), default=4)
    args = p.parse_args()
    counts = collections.Counter()
    lines = 0
    with args.traces.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            context = item["context"]
            if not isinstance(context, list) or any(type(x) is not int or not 0 <= x <= UINT32_MAX for x in context):
                raise ValueError("context must be an array of uint32 IDs")
            for target in item["targets"]:
                tok, weight = target["token"], target["weight"]
                if type(tok) is not int or not 0 <= tok <= UINT32_MAX or type(weight) is not int or not 1 <= weight <= UINT32_MAX:
                    raise ValueError("token and weight must be uint32, weight must be positive")
                for order in range(1, min(len(context), args.max_order) + 1):
                    key = signature(context[-order:])
                    counts[(key, tok)] = min(UINT32_MAX, counts[(key, tok)] + weight)
            lines += 1
    entries = sorted((key, token, count) for (key, token), count in counts.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = HEADER.size + len(entries) * ENTRY.size
    with args.output.open("wb") as out:
        out.write(HEADER.pack(MAGIC, 1, 0, 0, 0, len(entries), 0, 0, HEADER.size, total_bytes))
        for entry in entries:
            out.write(ENTRY.pack(*entry))
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"records={len(entries)} trace_lines={lines} bytes={total_bytes} sha256={sha}")

if __name__ == "__main__": main()

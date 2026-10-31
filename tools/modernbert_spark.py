#!/usr/bin/env python3
"""Compile and evaluate matrix-free SPARK MLM retrieval student."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.modernbert import compile_index, evaluate, load_index, read_traces


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    build = sub.add_parser("compile")
    build.add_argument("traces", type=Path)
    build.add_argument("output", type=Path)
    run = sub.add_parser("evaluate")
    run.add_argument("traces", type=Path)
    run.add_argument("index", type=Path)
    run.add_argument("--fallback-nearest", action="store_true", help="Full scan if no LSH bucket matches; slow")
    demo = sub.add_parser("predict")
    demo.add_argument("index", type=Path)
    demo.add_argument("token_ids", help="Comma-separated token IDs (including mask token)")
    demo.add_argument("mask_index", type=int)
    demo.add_argument("--fallback-nearest", action="store_true")
    a = p.parse_args()
    if a.command == "predict":
        ids = [int(x) for x in a.token_ids.split(",")]
        result = load_index(a.index).predict(ids, a.mask_index, fallback=a.fallback_nearest)
    else:
        traces = read_traces(a.traces)
        if a.command == "compile":
            selected = [t for t in traces if t.get("split") == "train"]
            if not selected:
                p.error("no training traces: run teacher collector first")
            result = compile_index(selected, a.output)
        else:
            selected = [t for t in traces if t.get("split") == "validation"]
            if not selected:
                p.error("no validation traces: supply more data or adjust split")
            result = evaluate(load_index(a.index), selected, fallback=a.fallback_nearest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

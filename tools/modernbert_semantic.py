#!/usr/bin/env python3
"""Compile and evaluate ModernBERT-supervised matrix-free SPARK signatures."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.modernbert import compile_index, evaluate, load_index, read_traces
from spark.semantic import SemanticEncoder, _target_code, train_encoder, train_signatures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile", help="train signature student and teacher-keyed index")
    c.add_argument("traces", type=Path)
    c.add_argument("prefix", type=Path)
    c.add_argument("--min-support", type=int, default=1)
    e = sub.add_parser("evaluate", help="held-out validation only")
    e.add_argument("traces", type=Path)
    e.add_argument("prefix", type=Path)
    e.add_argument("--fallback-nearest", action="store_true")
    p = sub.add_parser("predict", help="matrix-free masked-token prediction")
    p.add_argument("prefix", type=Path)
    p.add_argument("token_ids", help="comma-separated input token IDs with a mask")
    p.add_argument("mask_index", type=int)
    p.add_argument("--fallback-nearest", action="store_true")
    args = ap.parse_args()

    encoder_path = Path(str(args.prefix) + ".encoder.json")
    index_path = Path(str(args.prefix) + ".smlm")
    if args.command == "compile":
        traces = [t for t in read_traces(args.traces) if t.get("split") == "train"]
        if not traces:
            ap.error("no training traces")
        try:
            encoder = train_encoder(traces, min_support=args.min_support)
            encoder.save(encoder_path)
            summary = compile_index(traces, index_path,
                                    signatures=train_signatures(traces, encoder), encoder=encoder)
        except (ValueError, KeyError) as exc:
            ap.error(str(exc))
        summary.update({"features": len(encoder.weights),
                        "encoder_bytes": encoder_path.stat().st_size,
                        "warning": "research baseline, no proven semantic equivalence"})
    else:
        encoder = SemanticEncoder.load(encoder_path)
        index = load_index(index_path, encoder=encoder)
        if args.command == "predict":
            ids = [int(v) for v in args.token_ids.split(",")]
            summary = index.predict(ids, args.mask_index, fallback=args.fallback_nearest)
            summary["known_feature_fraction"] = encoder.coverage(ids, args.mask_index)
        else:
            traces = [t for t in read_traces(args.traces) if t.get("split") == "validation"]
            if not traces:
                ap.error("no held-out validation traces")
            summary = evaluate(index, traces, fallback=args.fallback_nearest)
            covered = [encoder.coverage(t["input_ids"], t["mask_index"]) for t in traces]
            summary["mean_known_feature_fraction"] = sum(covered) / len(covered)
            if all("mask_hidden" in t for t in traces):
                agreements = []
                for t in traces:
                    target = _target_code(t["mask_hidden"], encoder.seed)
                    predicted = encoder.encode(t["input_ids"], t["mask_index"])
                    agreements.append(1 - ((target ^ predicted).bit_count() / 256))
                summary["mean_teacher_signature_bit_agreement"] = sum(agreements) / len(agreements)
            summary["note"] = "validation traces are never used for encoder fitting"
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

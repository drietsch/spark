#!/usr/bin/env python3
"""Capture top-k next-token teacher traces from a local llama.cpp server.

A REAL teacher must be running at --url. No synthetic data is silently substituted.
This only extracts observations; it does not convert GGUF weights or train
learned associative signatures. Trace contexts are real model token IDs.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path
from urllib import request, error

UINT32_MAX = (1 << 32) - 1


def api(url, route, payload=None, timeout=120):
    base = url.rstrip("/")
    req = request.Request(base + route,
                          data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"},
                          method="GET" if payload is None else "POST")
    try:
        with request.urlopen(req, timeout=timeout) as result:
            return json.load(result)
    except (error.URLError, error.HTTPError) as exc:
        raise RuntimeError(f"llama.cpp server request failed: {route}: {exc}") from exc


def top_from_response(response, k, scale):
    """Convert llama.cpp raw next-token logprobs to SPARK uint32 weights.

    Use pre-sampling logprobs, since sampler behavior should not change the
    teacher distribution. Weights are rounded proxy scores, not calibrated
    probabilities and may omit most of the distribution mass.
    """
    chunks = response.get("completion_probabilities") or response.get("probs") or []
    if not chunks:
        raise ValueError("No next-token probabilities: llama-server must support n_probs")
    first = chunks[0]
    top = first.get("top_logprobs")
    if not top:
        raise ValueError("Missing top_logprobs (or got post_sampling_probs=true)")
    candidates = {}
    for entry in top:
        tok = entry.get("id")
        lp = entry.get("logprob")
        if type(tok) is not int or not 0 <= tok <= UINT32_MAX:
            raise ValueError("llama.cpp response has invalid token ID")
        if not isinstance(lp, (int, float)) or not math.isfinite(lp) or lp > 1e-5:
            raise ValueError("llama.cpp response has invalid log probability")
        w = max(1, min(UINT32_MAX, round(math.exp(float(lp)) * scale)))
        candidates[tok] = max(candidates.get(tok, 0), w)
    return [{"token": tok, "weight": w} for tok, w in
            sorted(candidates.items(), key=lambda pair: (-pair[1], pair[0]))[:k]]


def collect(args):
    health = api(args.url, "/health", timeout=args.timeout)
    if health.get("status") != "ok":
        raise RuntimeError(f"llama.cpp model not ready: {health}")
    prompts = [s.strip() for s in args.prompts.read_text(encoding="utf-8").splitlines()
               if s.strip() and not s.lstrip().startswith("#")]
    if len(prompts) < 2:
        raise ValueError("Need at least two different prompts for train/holdout split")
    indices = list(range(len(prompts)))
    random.Random(args.seed).shuffle(indices)
    holdout_n = max(1, min(len(prompts) - 1, round(len(prompts) * args.holdout)))
    holdout_ids = set(indices[:holdout_n])

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    if args.train_out.resolve() == args.eval_out.resolve():
        raise ValueError("train and eval output paths must be different")
    total = {"train": 0, "eval": 0}
    with args.train_out.open("w", encoding="utf-8") as train, \
         args.eval_out.open("w", encoding="utf-8") as holdout:
        for i, prompt in enumerate(prompts):
            response = api(args.url, "/tokenize", {"content": prompt,
                             "add_special": True, "parse_special": True}, args.timeout)
            tokens = response.get("tokens", [])
            if not tokens or any(type(t) is not int or not 0 <= t <= UINT32_MAX for t in tokens):
                raise ValueError("Invalid tokens from llama.cpp /tokenize")
            limit = min(len(tokens), args.max_prefix_tokens)
            prefixes = list(range(args.min_prefix, limit + 1, args.prefix_step))
            prefixes = prefixes[:args.max_per_prompt]
            subset = "eval" if i in holdout_ids else "train"
            out = holdout if subset == "eval" else train
            for n in prefixes:
                result = api(args.url, "/completion", {
                    "prompt": tokens[:n],
                    "n_predict": 1,
                    "n_probs": args.top_k,
                    "post_sampling_probs": False,
                    "stream": False,
                    "return_tokens": True,
                    "cache_prompt": True,
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "repeat_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "seed": args.seed,
                }, args.timeout)
                targets = top_from_response(result, args.top_k, args.weight_scale)
                if not targets:
                    raise ValueError("Empty teacher prediction candidates")
                record = {"context": tokens[max(0, n - args.max_order):n],
                          "targets": targets,
                          "source": {"teacher": "LiquidAI/LFM3-1.7B-GGUF", "split": subset,
                                     "prompt_index": i, "prefix_tokens": n}}
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                total[subset] += 1
            print(f"prompt {i+1}/{len(prompts)}  subset={subset}  prefixes={len(prefixes)}", file=sys.stderr)
    if min(total.values()) == 0:
        raise ValueError("No train or eval traces captured; reduce --min-prefix")
    meta = {"teacher": "LiquidAI/LFM3-1.7B-GGUF", "server": args.url,
            "prompts": len(prompts), "train_traces": total["train"],
            "eval_traces": total["eval"], "max_order": args.max_order,
            "top_k": args.top_k, "seed": args.seed,
            "note": "Exact n-gram baseline: no weights converted, no MLIR backend."}
    print(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--prompts", type=Path, default=Path("examples/lfm3_prompts.txt"))
    p.add_argument("--train-out", type=Path, default=Path("outputs/lfm3_train.jsonl"))
    p.add_argument("--eval-out", type=Path, default=Path("outputs/lfm3_holdout.jsonl"))
    p.add_argument("--holdout", type=float, default=.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-prefix", type=int, default=5)
    p.add_argument("--prefix-step", type=int, default=3)
    p.add_argument("--max-prefix-tokens", type=int, default=128)
    p.add_argument("--max-per-prompt", type=int, default=6)
    p.add_argument("--max-order", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--weight-scale", type=int, default=100000)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()
    if not 0 < args.holdout < 1 or not 1 <= args.top_k <= 128 or \
       args.min_prefix < 1 or args.prefix_step < 1 or args.max_per_prompt < 1 or \
       args.max_prefix_tokens < args.min_prefix or not 1 <= args.weight_scale <= UINT32_MAX:
        p.error("invalid split, prefix, top-k, or weight-scale settings")
    try:
        collect(args)
    except (RuntimeError, ValueError, OSError) as exc:
        p.exit(1, f"error: {exc}\n")

if __name__ == "__main__":
    main()

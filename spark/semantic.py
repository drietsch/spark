"""Teacher-supervised, matrix-free semantic signatures for ModernBERT SPARK.

Teacher hidden states are only used offline to create 256-bit binary targets.
Inference is token feature lookup, integer voting and bitwise retrieval.
This is not direct ModernBERT weight compilation.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spark.modernbert import _features, validate_trace

BITS = 256
FORMAT = "spark-semantic-encoder-v1"
SEED = 20261031
PROJECTION_SAMPLES = 24


def _target_code(hidden: list[float], seed: int = SEED) -> int:
    """OFFLINE sparse random hyperplanes applied to teacher mask embeddings."""
    if not isinstance(hidden, list) or not 1 <= len(hidden) <= 8192:
        raise ValueError("teacher mask_hidden must be a nonempty list of <=8192 floats")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in hidden):
        raise ValueError("mask_hidden contains invalid values")
    rng = random.Random(seed)
    code = 0
    for bit in range(BITS):
        value = 0.0
        for _ in range(PROJECTION_SAMPLES):
            index = rng.randrange(len(hidden))
            value += (1 if rng.getrandbits(1) else -1) * hidden[index]
        if value > 0:
            code |= 1 << bit
    return code


def _feature_counts(ids: list[int], pos: int) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for name, weight in _features(ids, pos):
        counts[name.decode("ascii")] += weight
    return dict(counts)


@dataclass
class SemanticEncoder:
    weights: dict[str, list[int]]
    seed: int = SEED
    min_support: int = 1
    fallback_bits: int = 0

    def encode(self, ids: list[int], pos: int) -> int:
        votes = [0] * BITS
        matched = 0
        for feature, freq in _feature_counts(ids, pos).items():
            weights = self.weights.get(feature)
            if weights is None:
                continue
            matched += 1
            for bit, val in enumerate(weights):
                votes[bit] += val * freq
        if not matched:
            return self.fallback_bits
        return sum(1 << i for i, vote in enumerate(votes) if vote > 0)

    def coverage(self, ids: list[int], pos: int) -> float:
        counts = _feature_counts(ids, pos)
        return sum(int(f in self.weights) for f in counts) / len(counts) if counts else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"format": FORMAT, "bits": BITS, "seed": self.seed,
                "projection_samples": PROJECTION_SAMPLES,
                "min_support": self.min_support,
                "fallback_bits": f"{self.fallback_bits:064x}",
                "features": {k: self.weights[k] for k in sorted(self.weights)}}

    def fingerprint64(self) -> int:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SemanticEncoder":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (raw.get("format"), raw.get("bits"), raw.get("projection_samples")) != (FORMAT, BITS, PROJECTION_SAMPLES):
            raise ValueError("unsupported semantic encoder format")
        weights = raw.get("features")
        if not isinstance(weights, dict) or len(weights) > 3_000_000:
            raise ValueError("invalid feature dictionary")
        for feature, values in weights.items():
            if not isinstance(feature, str) or not isinstance(values, list) or len(values) != BITS:
                raise ValueError("invalid feature entry")
            if any(type(x) is not int or not -7 <= x <= 7 for x in values):
                raise ValueError("invalid feature weights")
        fallback = int(raw["fallback_bits"], 16)
        if not 0 <= fallback < (1 << BITS):
            raise ValueError("invalid fallback bits")
        return cls(weights, int(raw["seed"]), int(raw["min_support"]), fallback)


def train_encoder(traces: list[dict], *, min_support: int = 1, seed: int = SEED) -> SemanticEncoder:
    """Fit sparse signed feature correlations using train-only teacher vectors."""
    if min_support < 1:
        raise ValueError("min_support must be positive")
    stats = {}
    all_targets = []
    for trace in traces:
        validate_trace(trace)
        if trace.get("split") != "train":
            raise ValueError("semantic encoder may only train on train traces")
        target = _target_code(trace["mask_hidden"], seed)
        all_targets.append(target)
        features = _feature_counts(trace["input_ids"], trace["mask_index"])
        for feature in features:
            if feature not in stats:
                stats[feature] = [0, [0] * BITS]
            count, votes = stats[feature]
            stats[feature][0] = count + 1
            for bit in range(BITS):
                votes[bit] += 1 if target & (1 << bit) else -1
    if not all_targets:
        raise ValueError("training traces are empty")
    out = {}
    for feature, (count, votes) in stats.items():
        if count < min_support:
            continue
        out[feature] = [max(-7, min(7, round(7 * v / (count + 3)))) for v in votes]
    global_votes = [sum(1 if val & (1 << i) else -1 for val in all_targets) for i in range(BITS)]
    fallback = sum(1 << i for i, v in enumerate(global_votes) if v > 0)
    return SemanticEncoder(out, seed, min_support, fallback)


def train_signatures(traces: list[dict], encoder: SemanticEncoder) -> list[int]:
    return [_target_code(t["mask_hidden"], encoder.seed) for t in traces]

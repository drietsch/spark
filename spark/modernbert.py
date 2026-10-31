"""Versioned SPARKMLM binary index and matrix-free masked-token retrieval.

The student operates on token IDs only: no teacher or neural model is loaded for
prediction. This is a research baseline, NOT a Transformer-weight compiler.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"SPARKMLM"
HEADER = struct.Struct("<8sHHIQQQQQQ")  # 64 B
RECORD = struct.Struct("<32sQII")      # signature, offset (element index), count, mask pos
CANDIDATE = struct.Struct("<If")       # token ID, teacher probability
VERSION = (1, 0)
BANDS = 8
BAND_BITS = 32


def _features(token_ids: list[int], mask_index: int):
    if not token_ids or not 0 <= mask_index < len(token_ids):
        raise ValueError("mask_index outside nonempty token_ids")
    if len(token_ids) > 8192:
        raise ValueError("max context length is 8192 tokens")
    if any(type(t) is not int or not 0 <= t < 2**32 for t in token_ids):
        raise ValueError("token IDs must be uint32")
    # Surrounding tokens provide position-sensitive features; the sequence bag
    # provides less brittle signals for related but non-identical contexts.
    left = token_ids[max(0, mask_index - 12):mask_index]
    right = token_ids[mask_index + 1:mask_index + 13]
    for distance, token in enumerate(reversed(left), 1):
        yield f"L{min(distance, 5)}:{token}".encode(), 3 if distance <= 2 else 1
    for distance, token in enumerate(right, 1):
        yield f"R{min(distance, 5)}:{token}".encode(), 3 if distance <= 2 else 1
    for a, b in zip(left, left[1:]):
        yield f"LB:{a}:{b}".encode(), 2
    for a, b in zip(right, right[1:]):
        yield f"RB:{a}:{b}".encode(), 2
    for token in set(token_ids[:128]):
        if token != token_ids[mask_index]:
            yield f"B:{token}".encode(), 1


def signature(token_ids: list[int], mask_index: int) -> int:
    votes = [0] * 256
    for feature, weight in _features(token_ids, mask_index):
        value = int.from_bytes(hashlib.blake2b(feature, digest_size=32, person=b"SPARKMLM").digest(), "little")
        for bit in range(256):
            votes[bit] += weight if value & (1 << bit) else -weight
    return sum(1 << bit for bit, vote in enumerate(votes) if vote > 0)


def validate_trace(trace: dict) -> None:
    ids, pos, targets = trace["input_ids"], trace["mask_index"], trace["topk"]
    if type(pos) is not int:
        raise ValueError("mask_index must be an integer")
    _features_valid = list(_features(ids, pos))  # validation
    if not _features_valid:
        raise ValueError("empty feature vector")
    if not isinstance(targets, list) or not targets:
        raise ValueError("topk must contain at least one candidate")
    for t in targets:
        if type(t["token"]) is not int or not 0 <= t["token"] < 2**32:
            raise ValueError("invalid candidate token")
        p = t["prob"]
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid candidate probability")


def read_traces(path: Path) -> list[dict]:
    traces = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                validate_trace(obj)
                traces.append(obj)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return traces


@dataclass(frozen=True)
class Entry:
    bits: int
    targets: tuple[tuple[int, float], ...]


class Index:
    def __init__(self, entries: list[Entry], encoder=None):
        self.entries = entries
        self.encoder = encoder
        self.buckets = defaultdict(list)
        # Less brittle LSH for noisy learned signatures; baseline remains 8x32.
        self.band_bits = 8 if encoder is not None else BAND_BITS
        self.bands = 256 // self.band_bits
        mask = (1 << self.band_bits) - 1
        for i, entry in enumerate(entries):
            for band in range(self.bands):
                self.buckets[(band, (entry.bits >> (self.band_bits * band)) & mask)].append(i)

    def predict(self, token_ids: list[int], mask_index: int, *, neighbors: int = 8, fallback: bool = False):
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        sig = self.encoder.encode(token_ids, mask_index) if self.encoder is not None else signature(token_ids, mask_index)
        candidate_indices = set()
        mask = (1 << self.band_bits) - 1
        for band in range(self.bands):
            candidate_indices.update(self.buckets.get((band, (sig >> (self.band_bits * band)) & mask), ()))
        used_fallback = False
        if not candidate_indices and fallback:
            candidate_indices = set(range(len(self.entries)))
            used_fallback = True
        if not candidate_indices:
            return {"token": None, "retrieved": 0, "distance": None, "fallback": False}
        ranked = sorted(((sig ^ self.entries[i].bits).bit_count(), i) for i in candidate_indices)[:neighbors]
        scores = defaultdict(float)
        for distance, i in ranked:
            # Weight nearby signatures substantially more heavily than remote ones.
            weight = (1 + (256 - distance) / 256) ** 4
            for tok, prob in self.entries[i].targets:
                scores[tok] += prob * weight
        winner = min(scores, key=lambda tok: (-scores[tok], tok))
        return {"token": winner, "retrieved": len(candidate_indices), "distance": ranked[0][0], "fallback": used_fallback}


def compile_index(traces: list[dict], path: Path, *, signatures: list[int] | None = None, encoder=None) -> dict:
    if (signatures is None) != (encoder is None):
        raise ValueError("semantic signatures require an encoder and vice versa")
    if signatures is not None and len(signatures) != len(traces):
        raise ValueError("signatures length mismatch")
    seen = set()
    entries = []
    for row_index, trace in enumerate(traces):
        validate_trace(trace)
        key = (tuple(trace["input_ids"]), trace["mask_index"])
        if key in seen:
            raise ValueError("duplicate prompt in train set; deduplicate before compiling")
        seen.add(key)
        bits = signatures[row_index] if signatures is not None else signature(*key)
        if type(bits) is not int or not 0 <= bits < 1 << 256:
            raise ValueError("invalid 256-bit signature")
        targets = tuple((t["token"], float(t["prob"])) for t in trace["topk"])
        entries.append(Entry(bits, targets))
    count = sum(len(entry.targets) for entry in entries)
    data_offset = HEADER.size + len(entries) * RECORD.size
    total = data_offset + count * CANDIDATE.size
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(HEADER.pack(MAGIC, *VERSION, 1 if encoder else 0, len(entries), count, HEADER.size, data_offset, total, encoder.fingerprint64() if encoder else 0))
        offset = 0
        for entry in entries:
            f.write(RECORD.pack(entry.bits.to_bytes(32, "little"), offset, len(entry.targets), 0))
            offset += len(entry.targets)
        for entry in entries:
            for token, prob in entry.targets:
                f.write(CANDIDATE.pack(token, prob))
    return {"prompts": len(entries), "candidates": count, "bytes": total}


def load_index(path: Path, *, encoder=None) -> Index:
    with path.open("rb") as f:
        raw = f.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("short index header")
        magic, major, minor, flags, n, candidates, rec_off, data_off, total, reserved = HEADER.unpack(raw)
        if magic != MAGIC or major != VERSION[0] or flags not in (0, 1):
            raise ValueError("unsupported index header")
        if flags == 0 and (reserved != 0 or encoder is not None):
            raise ValueError("baseline index requires no semantic encoder")
        if flags == 1 and (encoder is None or reserved != encoder.fingerprint64()):
            raise ValueError("semantic index requires its exact encoder")
        if (minor > VERSION[1] or rec_off != HEADER.size or
                data_off != HEADER.size + n * RECORD.size or
                total != data_off + candidates * CANDIDATE.size):
            raise ValueError("invalid index offsets or version")
        if path.stat().st_size != total:
            raise ValueError("index size mismatch")
        if n > 10_000_000 or candidates > 100_000_000:
            raise ValueError("index too large for prototype loader")
        records = [RECORD.unpack(f.read(RECORD.size)) for _ in range(n)]
        candidate_data = [CANDIDATE.unpack(f.read(CANDIDATE.size)) for _ in range(candidates)]
    entries = []
    for raw_sig, offset, length, reserved in records:
        if reserved or not length or offset + length > candidates:
            raise ValueError("bad index record")
        targets = tuple(candidate_data[offset:offset + length])
        if any(not math.isfinite(prob) or not 0 <= prob <= 1 for _, prob in targets):
            raise ValueError("invalid candidate probability in index")
        entries.append(Entry(int.from_bytes(raw_sig, "little"), targets))
    return Index(entries, encoder=encoder)


def evaluate(index: Index, traces: list[dict], *, fallback: bool = False) -> dict:
    total = hits = correct = fallbacks = 0
    for trace in traces:
        validate_trace(trace)
        predicted = index.predict(trace["input_ids"], trace["mask_index"], fallback=fallback)
        total += 1
        if predicted["token"] is not None:
            hits += 1
            fallbacks += int(predicted["fallback"])
            teacher_top1 = max(trace["topk"], key=lambda x: x["prob"])["token"]
            correct += predicted["token"] == teacher_top1
    return {
        "examples": total,
        "coverage": hits / total if total else 0,
        "top1_agreement_covered": correct / hits if hits else None,
        "top1_agreement_all": correct / total if total else None,
        "fallback_queries": fallbacks,
    }

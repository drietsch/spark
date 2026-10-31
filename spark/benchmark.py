"""Auditable held-out evaluation helpers for a masked-LM SPARK student.

No neural dependencies; trace collection (and teacher forward pass) happens
separately. A `teacher_top1` trace label is not ground-truth task accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from typing import Any

EXPECTED_TEACHER = 'answerdotai/ModernBERT-base'


def teacher_top1(row: dict) -> int:
    # Deterministic tie breaking is important when comparing native predictions.
    return min(row['topk'], key=lambda t: (-t['prob'], t['token']))['token']


def _content_key(row: dict) -> str:
    # Exact-context keys alone miss leakage when the same source text is masked
    # at different positions. Group by normalized source text if available.
    group = row.get('source_group')
    if isinstance(group, str) and group.strip():
        return hashlib.sha256(('group:' + group).encode('utf-8')).hexdigest()
    text = row.get('source_text')
    if isinstance(text, str) and text.strip():
        content = ' '.join(text.casefold().split())
        return hashlib.sha256(('text:' + content).encode('utf-8')).hexdigest()
    key = json.dumps([row['input_ids'], row['mask_index']], separators=(',', ':'))
    return hashlib.sha256(('tokens:' + key).encode()).hexdigest()


def audit_traces(rows: list[dict], *, allow_synthetic: bool = False) -> dict[str, Any]:
    if not rows:
        raise ValueError('no teacher traces')
    ids_by_split: dict[tuple[int, ...], str] = {}
    source_groups: dict[str, str] = {}
    seen: dict[tuple[tuple[int, ...], int], str] = {}
    counts = Counter()
    model_names = set()
    for row in rows:
        split = row.get('split')
        if split not in ('train', 'validation'):
            raise ValueError('every trace needs an explicit train/validation split')
        counts[split] += 1
        model = row.get('model')
        model_names.add(model)
        if not allow_synthetic and model != EXPECTED_TEACHER:
            raise ValueError('expected real ModernBERT traces; pass --allow-synthetic for fixture tests')
        exact = (tuple(row['input_ids']), row['mask_index'])
        if exact in seen:
            if seen[exact] != split:
                raise ValueError('split leakage: duplicate masked context across splits')
            raise ValueError('duplicate masked context in traces')
        seen[exact] = split
        # Different mask positions from one token sequence are still grouped.
        tokens = tuple(row['input_ids'])
        if tokens in ids_by_split and ids_by_split[tokens] != split:
            raise ValueError('split leakage: identical token sequence in both splits')
        ids_by_split[tokens] = split
        source = _content_key(row)
        if source in source_groups and source_groups[source] != split:
            raise ValueError('split leakage: identical source text in both splits')
        source_groups[source] = split
    if not counts['train'] or not counts['validation']:
        raise ValueError('both train and validation traces are required')
    return {'train': counts['train'], 'validation': counts['validation'],
            'teacher_models': sorted(str(v) for v in model_names),
            'holdout_warning': 'validation has fewer than 200 examples; report is exploratory'
                               if counts['validation'] < 200 else None}


def percentile_ms(samples_ns: list[int], percentile: int) -> float | None:
    if not samples_ns:
        return None
    ordered = sorted(samples_ns)
    # nearest-rank estimator, stable for small sample sizes
    index = max(0, math.ceil(percentile * len(ordered) / 100) - 1)
    return round(ordered[index] / 1_000_000, 5)


def latency_summary(samples_ns: list[int]) -> dict:
    return {'queries': len(samples_ns),
            'p50_ms': percentile_ms(samples_ns, 50),
            'p95_ms': percentile_ms(samples_ns, 95),
            'mean_ms': round(statistics.mean(samples_ns) / 1_000_000, 5) if samples_ns else None,
            'queries_per_s': round(1_000_000_000 * len(samples_ns) / sum(samples_ns), 3)
                             if samples_ns and sum(samples_ns) > 0 else None}


def evaluate_predictions(rows: list[dict], predictions: list[int | None]) -> dict:
    if len(rows) != len(predictions):
        raise ValueError('prediction count mismatch')
    correct = sum(p == teacher_top1(r) for p, r in zip(predictions, rows))
    covered = sum(p is not None for p in predictions)
    labels = [teacher_top1(r) for r in rows]
    return {
        'examples': len(rows), 'coverage': covered / len(rows) if rows else 0,
        'top1_agreement_all': correct / len(rows) if rows else None,
        'top1_agreement_covered': correct / covered if covered else None,
        'unanswered': len(rows) - covered,
        'distinct_teacher_labels': len(set(labels)),
    }


def majority_baseline(train: list[dict], validation: list[dict]) -> dict:
    common = Counter(teacher_top1(r) for r in train).most_common(1)[0][0]
    matches = sum(teacher_top1(r) == common for r in validation)
    return {'label': common, 'top1_agreement_all': matches / len(validation),
            'note': 'majority teacher token on training split; no student model'}

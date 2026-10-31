"""M7: lexical n-gram memory and teacher-supervised sparse fusion.

Student inference uses only dictionary postings and scalar additions. The
external text corpus is used offline for association construction, never
ModernBERT or neural operations during student prediction.
"""
from __future__ import annotations
from collections import Counter, defaultdict


def lexical_features(ids: list[int], pos: int) -> dict[tuple, float]:
    if not 0 <= pos < len(ids):
        raise ValueError("invalid mask position")
    out = {}
    left, right = ids[:pos], ids[pos+1:]
    if left:
        out[("L1", left[-1])] = 1.4
    if right:
        out[("R1", right[0])] = 1.4
    if len(left) >= 2:
        out[("L2", left[-2], left[-1])] = 3.0
    if len(right) >= 2:
        out[("R2", right[0], right[1])] = 3.0
    if len(left) >= 3:
        out[("L3", left[-3], left[-2], left[-1])] = 4.5
    if len(right) >= 3:
        out[("R3", right[0], right[1], right[2])] = 4.5
    if left and right:
        out[("LR", left[-1], right[0])] = 6.0
    if len(left) >= 2 and right:
        out[("LLR", left[-2], left[-1], right[0])] = 7.0
    if left and len(right) >= 2:
        out[("LRR", left[-1], right[0], right[1])] = 7.0
    return out


class LexicalBackoff:
    def __init__(self):
        self.postings = defaultdict(Counter)
        self.support = Counter()
        self.priors = Counter()
        self.seen = 0

    def add(self, ids: list[int], pos: int, label: int) -> None:
        if label < 0:
            raise ValueError("negative label")
        self.seen += 1
        self.priors[label] += 1
        for feature in lexical_features(ids, pos):
            self.postings[feature][label] += 1
            self.support[feature] += 1

    def score(self, ids: list[int], pos: int, *, smoothing: float = 5.0,
              specificity: float = 1.0, max_labels: int = 100) -> dict[int, float]:
        scores = defaultdict(float)
        for feature, strength in lexical_features(ids, pos).items():
            posting = self.postings.get(feature)
            if not posting:
                continue
            weight = strength ** specificity / (self.support[feature] + smoothing)
            entries = posting.most_common(max_labels) if len(posting) > max_labels else posting.items()
            for label, count in entries:
                scores[label] += weight * count
        return dict(scores)

    def predict(self, ids: list[int], pos: int, **kwargs) -> int | None:
        scored = self.score(ids, pos, **kwargs)
        if scored:
            return max(scored, key=lambda t: (scored[t], -t))
        if self.priors:
            return max(self.priors, key=lambda t: (self.priors[t], -t))
        return None

    def stats(self) -> dict:
        return {"observations": self.seen, "feature_types": len(self.postings),
                "label_postings": sum(len(p) for p in self.postings.values()),
                "vocabulary": len(self.priors)}


def sparse_scores(model, ids: list[int], mask: int, *, smoothing: float = 10.0,
                  exponent: float = 1.0, max_labels: int = 48) -> dict[int, float]:
    from spark.associative_classifier import context_features
    scores = defaultdict(float)
    for feature, strength in context_features(ids, mask).items():
        support = model.support.get(feature, 0)
        if support < 1:
            continue
        posting = model.postings[feature]
        entries = (sorted(posting.items(), key=lambda p: (-p[1], p[0]))[:max_labels]
                   if len(posting) > max_labels else posting.items())
        weight = strength / (support + smoothing) ** exponent
        for label, count in entries:
            scores[label] += weight * count
    return dict(scores)


def fused_predictions(lexical: dict[int, float], teacher: dict[int, float],
                      *, lexical_weight: float = 0.5, majority: int | None = None) -> int | None:
    if not 0 <= lexical_weight <= 1:
        raise ValueError("lexical_weight must be 0..1")
    lex_peak = max(lexical.values(), default=0.0)
    sup_peak = max(teacher.values(), default=0.0)
    candidates = set(lexical) | set(teacher)
    if not candidates:
        return majority
    scores = {}
    for token in candidates:
        lex = lexical.get(token, 0.0) / lex_peak if lex_peak else 0.0
        sup = teacher.get(token, 0.0) / sup_peak if sup_peak else 0.0
        scores[token] = lexical_weight * lex + (1 - lexical_weight) * sup
    return max(scores, key=lambda t: (scores[t], -t))

"""Teacher-supervised sparse associative masked-token baseline.

Inference uses feature-to-token postings and scalar additions; no matrices.
This is a quality baseline, not a Transformer compiler.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log1p, sqrt


def context_features(ids: list[int], mask: int) -> dict[str, float]:
    if not 0 <= mask < len(ids):
        raise ValueError("invalid mask position")
    out = {}
    for d in range(1, 6):
        if mask >= d:
            out[f"L{d}:{ids[mask-d]}"] = 3.0 / sqrt(d)
        if mask+d < len(ids):
            out[f"R{d}:{ids[mask+d]}"] = 3.0 / sqrt(d)
    if mask >= 2:
        out[f"LL:{ids[mask-2]}:{ids[mask-1]}"] = 4.5
    if mask+2 < len(ids):
        out[f"RR:{ids[mask+1]}:{ids[mask+2]}"] = 4.5
    if mask and mask+1 < len(ids):
        out[f"LR:{ids[mask-1]}:{ids[mask+1]}"] = 5.0
    for token in set(ids[max(0,mask-16):mask]+ids[mask+1:mask+17]):
        if token not in (50281, 50282, 50283, 50284):
            out[f"B:{token}"] = 0.4
    return out


@dataclass
class SparseAssoc:
    postings: dict[str,dict[int,float]]
    priors: Counter
    support: dict[str,int]

    @classmethod
    def fit(cls, rows: list[dict], *, topk_soft: bool = False) -> "SparseAssoc":
        postings = defaultdict(lambda: defaultdict(float))
        support = Counter()
        priors = Counter()
        for row in rows:
            if row["split"] != "train":
                raise ValueError("training leakage: non-train trace")
            labels = [(int(row["topk"][0]["token"]), 1.0)]
            if topk_soft:
                labels = [(int(x["token"]), float(x["prob"])) for x in row["topk"]]
            features = context_features(row["input_ids"],row["mask_index"])
            for token, prob in labels:
                priors[token] += prob
            for f in features:
                support[f] += 1
                for token, prob in labels:
                    postings[f][token] += prob
        return cls({f:dict(p) for f,p in postings.items()},priors,dict(support))

    def predict(self, ids: list[int], mask: int, *, smoothing: float=3.0,
                prior_weight:float=0.5, min_support:int=1,
                exponent:float=1.0, top_postings:int=48) -> int:
        if not self.priors:
            raise ValueError("empty training set")
        majority = max(self.priors,key=lambda t:(self.priors[t],-t))
        scores = defaultdict(float)
        scores[majority] = 0.0
        for feature, strength in context_features(ids,mask).items():
            df = self.support.get(feature,0)
            if df < min_support:
                continue
            posting = self.postings[feature]
            weight = strength / ((df+smoothing)**exponent)
            if len(posting)>top_postings:
                entries = sorted(posting.items(),key=lambda p:-p[1])[:top_postings]
            else:
                entries = posting.items()
            for token, count in entries:
                scores[token] += weight*count
        if prior_weight:
            for token in scores:
                scores[token] += prior_weight*log1p(self.priors[token])
        return max(scores,key=lambda t:(scores[t],-t))

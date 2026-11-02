"""Training/development-only reconstruction and held-out quality statistics."""
from __future__ import annotations
import numpy as np


def fit_candidates(train_features, train_target, dev_features, dev_target, output_weight, widths, ridge_factors):
    """Rank units on training contributions; select ridge only on validation."""
    h = np.asarray(train_features, dtype=np.float64); y = np.asarray(train_target, dtype=np.float64)
    hd = np.asarray(dev_features, dtype=np.float64); yd = np.asarray(dev_target, dtype=np.float64)
    if h.ndim != 2 or y.ndim != 2 or len(h) != len(y) or hd.shape[1] != h.shape[1] or yd.shape != (len(hd), y.shape[1]):
        raise ValueError('invalid reconstruction samples')
    importance = np.mean(h*h, axis=0) * np.sum(np.asarray(output_weight, dtype=np.float64)**2, axis=0)
    ranking = np.argsort(-importance, kind='stable')
    target_mean = y.mean(axis=0)
    scale = max(float(np.sqrt(np.mean((yd - target_mean)**2))), 1e-12)
    candidates = []
    for width in widths:
        if not 1 <= width <= h.shape[1]: raise ValueError('invalid candidate width')
        keep = ranking[:width]; x = h[:, keep]; mean = x.mean(axis=0); centered = x - mean
        gram = centered.T @ centered; cross = centered.T @ (y - target_mean)
        reference_scale = max(float(np.trace(gram) / width), 1e-12)
        trials = []
        for factor in ridge_factors:
            if not factor > 0: raise ValueError('ridge factors must be positive')
            coefficient = np.linalg.solve(gram + factor * reference_scale * np.eye(width), cross)
            # Actual deployed/live coefficients use source float32 storage.
            weight = coefficient.T.astype(np.float32)
            bias = (target_mean - mean @ coefficient).astype(np.float32)
            if not np.isfinite(weight).all() or not np.isfinite(bias).all() or max(np.max(np.abs(weight)), np.max(np.abs(bias))) > 1e4:
                raise ValueError('reconstruction coefficients exceed live update bounds')
            residual = hd[:, keep] @ weight.astype(np.float64).T + bias.astype(np.float64) - yd
            mse = float(np.mean(residual**2)); nrmse = float(np.sqrt(mse) / scale)
            trials.append({'ridge_factor': float(factor), 'dev_mse': mse, 'dev_nrmse': nrmse, 'weight': weight, 'bias': bias})
        best = min(trials, key=lambda row: (row['dev_nrmse'], row['ridge_factor']))
        candidates.append({'width': int(width), 'keep': keep.tolist(), **best,
                           'trials': [{k:v for k,v in t.items() if k not in ('weight','bias')} for t in trials]})
    return candidates


def correlation_prune(features,target,output_weight,width,ridge_factor):
    """Training-only conditional-covariance pruning with readout compensation."""
    h=np.asarray(features,dtype=np.float64);y=np.asarray(target,dtype=np.float64);w=np.asarray(output_weight,dtype=np.float64).copy()
    if h.ndim!=2 or w.shape!=(y.shape[1],h.shape[1]) or not 1<=width<h.shape[1] or ridge_factor<=0:
        raise ValueError('invalid correlation-aware pruning inputs')
    mean=h.mean(axis=0);target_mean=y.mean(axis=0);centered=h-mean
    gram=(centered.T@centered)/len(h);scale=max(float(np.trace(gram)/len(gram)),1e-12)
    precision=np.linalg.inv(gram+ridge_factor*scale*np.eye(len(gram)))
    active=list(range(h.shape[1]));removed=[]
    while len(active)>width:
        diagonal=np.diag(precision)
        if np.any(diagonal<=0) or not np.isfinite(precision).all():raise ValueError('unstable regularized covariance')
        costs=np.sum(w*w,axis=0)/diagonal;j=int(np.argmin(costs))
        mask=np.arange(len(active))!=j;remaining=np.flatnonzero(mask);row=precision[j,remaining]/precision[j,j]
        w=w[:,remaining]-w[:,j,None]*row[None,:]
        column=precision[remaining,j].copy();pjj=precision[j,j]
        precision=precision[np.ix_(remaining,remaining)]-np.outer(column,column)/pjj
        removed.append({'unit':active[j],'training_cost':float(costs[j])});active=[a for i,a in enumerate(active) if i!=j]
    weight=w.astype(np.float32);bias=(target_mean-mean[active]@w.T).astype(np.float32)
    if max(np.max(np.abs(weight)),np.max(np.abs(bias)))>1e4 or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('compensated coefficients exceed runtime bounds')
    return {'keep':active,'weight':weight,'bias':bias,'removed':removed,'ridge_factor':float(ridge_factor)}


def log_probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=1, keepdims=True)
    return values - maximum - np.log(np.exp(values - maximum).sum(axis=1, keepdims=True))


def quality_rows(reference, actual, labels, articles):
    reference = np.asarray(reference, dtype=np.float64); actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape or reference.ndim != 2 or len(labels) != len(reference) or len(articles) != len(labels):
        raise ValueError('quality arrays do not align')
    p = log_probabilities(reference); q = log_probabilities(actual)
    original = reference.argmax(axis=1); predicted = actual.argmax(axis=1); labels = np.asarray(labels)
    kl = np.maximum(0., np.sum(np.exp(p) * (p - q), axis=1))
    return [{'article': str(article), 'teacher_match': bool(a == b), 'baseline_correct': bool(a == gold),
             'correct': bool(b == gold), 'nll': float(-q[i, gold]), 'baseline_nll': float(-p[i, gold]), 'teacher_kl': float(kl[i])}
            for i, (article, a, b, gold) in enumerate(zip(articles, original, predicted, labels))]


def summarize_quality(rows, *, seed=20261102, samples=2000):
    if not rows: raise ValueError('no quality observations')
    def metrics(data):
        mean = lambda key: float(np.mean([r[key] for r in data]))
        return {'teacher_top1_agreement': mean('teacher_match'), 'gold_accuracy': mean('correct'),
                'baseline_gold_accuracy': mean('baseline_correct'), 'gold_accuracy_delta': mean('correct') - mean('baseline_correct'),
                'mean_nll': mean('nll'), 'mean_baseline_nll': mean('baseline_nll'), 'mean_teacher_kl_nats': mean('teacher_kl')}
    groups = {}
    for row in rows: groups.setdefault(row['article'], []).append(row)
    names = list(groups); rng = np.random.default_rng(seed)
    results = metrics(rows); boot = {k: [] for k in ('teacher_top1_agreement','gold_accuracy_delta','mean_teacher_kl_nats')}
    for _ in range(samples):
        data = [row for i in rng.integers(0, len(names), size=len(names)) for row in groups[names[int(i)]]]
        values = metrics(data)
        for key in boot: boot[key].append(values[key])
    results.update(queries=len(rows), articles=len(groups), confidence_intervals_95={k: np.quantile(v, [.025,.975]).tolist() for k,v in boot.items()})
    return results


def quality_passes(result, gates):
    return bool(result['teacher_top1_agreement'] >= gates['minimum_teacher_top1_agreement'] and
                result['gold_accuracy_delta'] >= -gates['maximum_gold_accuracy_drop'] and
                result['mean_teacher_kl_nats'] <= gates['maximum_mean_teacher_kl_nats'])


def paired_speedup(baseline, candidate, *, seed=20261102, samples=2000):
    base = np.asarray(baseline, dtype=float); modified = np.asarray(candidate, dtype=float)
    if base.shape != modified.shape or base.ndim != 1 or not len(base) or np.any(base <= 0) or np.any(modified <= 0):
        raise ValueError('invalid paired timings')
    ratio = base / modified; rng = np.random.default_rng(seed)
    bootstrap = np.median(ratio[rng.integers(0, len(ratio), size=(samples, len(ratio)))], axis=1)
    interval = np.quantile(bootstrap, [.025,.975]).tolist()
    return {'median_paired_speedup': float(np.median(ratio)), 'confidence_interval_95': interval,
            'speedup_demonstrated': bool(interval[0] > 1.), 'paired_rounds': len(base)}

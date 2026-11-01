"""M11: structural search for compact shared-gain, multiregister equations.

The *program* is a DAG of monomial instructions synthesized by structural
mutations, sparse regression and guard selection.  Its only adaptive numeric
values are small rational constants; there are no learned tensor weights.

Synthetic research grammar, not a general symbolic AI or Transformer competitor.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

try:
    import numpy as np
except ImportError:  # compiled / Python inference needs no numpy
    np = None

FORMAT = "spark-evolved-program-v1"
MAX_REGISTERS = 8
MAX_GENERATIONS = 4
MAX_TERMS = 12


def expand_topologies(registers: int, *, depth: int = 3, seed: int = 11,
                      fitness_rows=None, population_limit: int = 48):
    """Create a bounded population using multiply-by-variable mutations.

    No handcoded degree-3 basis list is supplied. Depth controls the number of
    graph expansion generations; mutation produces new structural motifs.
    Candidate nodes compete for survival using residual prediction fitness
    across training regimes. Pruning never uses the external validation set.
    """
    if not (1 <= registers <= MAX_REGISTERS and 1 <= depth <= MAX_GENERATIONS):
        raise ValueError("invalid topology search shape")
    width = registers + 2  # state registers, input x, shared global gain w
    zero = (0,) * width
    frontier = {zero}
    population = {zero}
    rng = random.Random(seed)
    generations = []
    for generation in range(1, depth + 1):
        mutated = set()
        for parent in sorted(frontier):
            order = list(range(width)); rng.shuffle(order)
            for index in order:
                child = list(parent); child[index] += 1
                mutated.add(tuple(child))
        frontier = mutated - population
        if fitness_rows is not None and len(frontier) > population_limit:
            if np is None:
                raise ImportError("training-only fitness pruning requires numpy")
            # Remove trends already explained by surviving topology nodes.
            previous = sorted(population)
            X = np.asarray([[_value(p,s,x,w) for p in previous]
                            for s,x,w,_ in fitness_rows])
            Y = np.asarray([y for *_,y in fitness_rows])
            explained = np.linalg.lstsq(X,Y,rcond=None)[0]
            residual = Y - X @ explained
            candidate_scores = []
            for candidate in sorted(frontier):
                v = np.asarray([_value(candidate,s,x,w) for s,x,w,_ in fitness_rows])
                norm = max(np.linalg.norm(v),1e-12)
                score = float(np.max(np.abs(v @ residual)) / norm)
                candidate_scores.append((score,candidate))
            candidate_scores.sort(key=lambda item:(-item[0],item[1]))
            frontier = {candidate for _,candidate in candidate_scores[:population_limit]}
        population |= frontier
        generations.append(len(frontier))
    # Fixed canonical output makes codegen and search reproducible.
    return sorted(population, key=lambda p: (sum(p), p)), generations


def _value(powers, state, x, gain):
    factors = [*state, x, gain]
    result = 1.0
    for a, exponent in zip(factors, powers):
        result *= float(a) ** exponent
    return result


def _pred_terms(terms, state, x, gain):
    return sum((num / den) * _value(tuple(p), state, x, gain) for p, num, den in terms)


def predict(program, state, x, gain):
    validate(program)
    if len(state) != program['registers'] or not all(math.isfinite(float(z)) for z in (*state, x, gain)):
        raise ValueError("invalid model input")
    branch = 'reset' if program['guard'] is not None and x == program['guard'] else 'outputs'
    if branch == 'reset':
        return tuple(0.0 for _ in state)
    return tuple(_pred_terms(terms, state, x, gain) for terms in program['outputs'])


def rollout(program, state, xs, gain):
    state = tuple(state)
    for x in xs:
        state = predict(program, state, x, gain)
    return state


def validate(program):
    if not isinstance(program, dict) or set(program) != {'format', 'registers', 'generations', 'guard', 'outputs'}:
        raise ValueError("invalid program keys")
    n = program['registers']
    if program['format'] != FORMAT or type(n) is not int or not 1 <= n <= MAX_REGISTERS:
        raise ValueError("invalid program format or register count")
    if type(program['generations']) is not int or not 1 <= program['generations'] <= MAX_GENERATIONS:
        raise ValueError("invalid expression depth")
    guard = program['guard']
    if guard is not None and (type(guard) not in (int, float) or not math.isfinite(guard) or abs(guard) > 100):
        raise ValueError("invalid event guard")
    outputs = program['outputs']
    if not isinstance(outputs, list) or len(outputs) != n:
        raise ValueError("incorrect number of register updates")
    for terms in outputs:
        if not isinstance(terms, list) or len(terms) > MAX_TERMS:
            raise ValueError("too many program instructions")
        seen = set()
        for term in terms:
            if not isinstance(term, list) or len(term) != 3:
                raise ValueError("invalid program term")
            powers, num, den = term
            if (not isinstance(powers, list) or len(powers) != n + 2 or
                any(type(v) is not int or not 0 <= v <= MAX_GENERATIONS for v in powers)
                or sum(powers) > program['generations'] or tuple(powers) in seen or
                type(num) is not int or abs(num) > 1024 or num == 0 or
                type(den) is not int or not 1 <= den <= 128):
                raise ValueError("invalid rational expression")
            seen.add(tuple(powers))


def _fit_sparse(X, y, powers, max_terms=8, tolerance=1e-14):
    """Greedy structural selection with rational-valued coefficient fitting."""
    residual = y.copy()
    norms = np.maximum(np.linalg.norm(X, axis=0), 1e-12)
    selected = []
    best = (float(np.mean(y*y)), [])
    for _ in range(max_terms):
        correlations = np.abs(X.T @ residual) / norms
        correlations[selected] = -1
        index = int(np.argmax(correlations))
        if correlations[index] < 1e-9:
            break
        selected.append(index)
        coefs = np.linalg.lstsq(X[:, selected], y, rcond=None)[0]
        rational = [Fraction(float(v)).limit_denominator(32) for v in coefs]
        compiled = [[list(powers[k]), v.numerator, v.denominator]
                    for k, v in zip(selected, rational) if v]
        predicted = X[:, selected] @ np.asarray([float(v) for v in rational])
        mse = float(np.mean((y - predicted) ** 2))
        if (mse, len(compiled)) < (best[0], len(best[1])):
            best = (mse, compiled)
        residual = y - X[:, selected] @ coefs
        if mse <= tolerance:
            break
    return best[1], best[0]


def _fit(rows, powers, registers, max_terms):
    X = np.asarray([[_value(p, s, x, gain) for p in powers]
                    for s, x, gain, _ in rows], dtype=float)
    targets = np.asarray([y for _, _, _, y in rows], dtype=float)
    result = []
    for target in targets.T:
        terms, _ = _fit_sparse(X, target, powers, max_terms=max_terms)
        result.append(terms)
    return result


def _score(rows, outputs, guard):
    errors = []
    for state, x, gain, truth in rows:
        pred = ([0.] * len(state) if guard is not None and x == guard else
                [_pred_terms(terms, state, x, gain) for terms in outputs])
        errors.extend((a-b)**2 for a, b in zip(truth, pred))
    return sum(errors) / len(errors)


def _complexity(outputs, guard):
    distinct = {tuple(p) for row in outputs for p, _, _ in row}
    links = sum(len(row) for row in outputs)
    return len(distinct) + links + (3 if guard is not None else 0)


@dataclass(frozen=True)
class EvolvedProgram:
    payload: dict
    train_mse: float
    inner_validation_mse: float
    topology_mutations: tuple[int, ...]
    complexity: int
    def step(self, state, x, gain):
        return predict(self.payload, state, x, gain)
    def rollout(self, state, xs, gain):
        return rollout(self.payload, state, xs, gain)


def evolve(rows, *, seed=11, max_generations=3, max_terms=8,
           min_guard_examples=22, complexity_weight=1e-9):
    """Choose graph depth, structural modules and guard using TRAIN-ONLY CV.

    Evaluates programs across all training gain values, preferring simpler
    accurate programs; outer OOD gains never influence search.
    """
    if np is None:
        raise ImportError("NumPy required only during symbolic synthesis")
    if len(rows) < 160 or not 1 <= max_generations <= MAX_GENERATIONS or not 1 <= max_terms <= MAX_TERMS:
        raise ValueError("insufficient training examples or invalid search budget")
    n = len(rows[0][0])
    if not 1 <= n <= MAX_REGISTERS:
        raise ValueError("invalid number of state registers")
    if any(len(s)!=n or len(y)!=n or any(not math.isfinite(float(v)) for v in (*s, x, gain, *y))
           for s, x, gain, y in rows):
        raise ValueError("invalid training observation")
    # Split examples deterministically, stratified by gain and input event. It is
    # internal CV; the public external validation set is never consulted.
    rng = random.Random(seed)
    grouped = {}
    for row in rows:
        grouped.setdefault((row[2], row[1]), []).append(row)
    train, tune = [], []
    for group in grouped.values():
        copy = group.copy(); rng.shuffle(copy)
        cut = max(1, len(copy)//5) if len(copy) >= 5 else 0
        tune.extend(copy[:cut]); train.extend(copy[cut:])
    if len(tune) < 25:
        raise ValueError("insufficient internal holdout observations")
    guards = [None]
    counts = {}
    for _, x, _, _ in train:
        counts[x] = counts.get(x, 0) + 1
    guards += [x for x, freq in sorted(counts.items()) if freq >= min_guard_examples
               and freq <= len(train)-min_guard_examples and x == int(x)]
    best = None
    audit = []
    for depth in range(1, max_generations + 1):
        for guard in guards:
            fit = [r for r in train if guard is None or r[1] != guard]
            if len(fit) < 150:
                continue
            powers, generations = expand_topologies(n, depth=depth, seed=seed,
                                                    fitness_rows=fit)
            # All outputs must reuse a common *topology population*; regression
            # selects the useful structural nodes independently per register.
            outputs = _fit(fit, powers, n, max_terms)
            mse = _score(tune, outputs, guard)
            cost = _complexity(outputs, guard)
            quality = mse + complexity_weight*cost
            audit.append((depth, guard, mse, cost, quality))
            if best is None or (quality, cost, depth) < best[0]:
                best = ((quality, cost, depth), depth, guard, generations)
    if best is None:
        raise ValueError("unable to synthesize program")
    _, depth, guard, generations = best
    fit = [r for r in rows if guard is None or r[1] != guard]
    powers, _ = expand_topologies(n, depth=depth, seed=seed, fitness_rows=fit)
    outputs = _fit(fit, powers, n, max_terms)
    model = {'format':FORMAT, 'registers':n, 'generations':depth,
             'guard':guard, 'outputs':outputs}
    validate(model)
    return EvolvedProgram(model, _score(rows, outputs, guard),
                          min(entry[2] for entry in audit if entry[0]==depth and entry[1]==guard),
                          tuple(generations), _complexity(outputs, guard))


def cpp_source(model):
    validate(model)
    n = model['registers']
    unique = sorted({tuple(p) for out in model['outputs'] for p,_,_ in out}, key=lambda p:(sum(p),p))
    names = {p:f't{i}' for i,p in enumerate(unique)}
    def term_cpp(p):
        factors = [f's[{i}]' for i in range(n)] + ['x','gain']
        parts = [factors[i] for i,count in enumerate(p) for _ in range(count)]
        return ' * '.join(parts) if parts else '1.0'
    lines = ['// Learned SPARK expression DAG: no neural weight tensors or lookup memory.',
             '#include <cstddef>',
             'extern "C" void spark_evolved_step_v1(const double* s,double x,double gain,double* out) {']
    if model['guard'] is not None:
        lines.append(f'  if (x == {float(model["guard"]):.17g}) {{')
        lines.extend(f'    out[{j}] = 0.0;' for j in range(n))
        lines.extend(['    return;','  }'])
    lines += [f'  const double {names[p]} = {term_cpp(p)};' for p in unique]
    for j, terms in enumerate(model['outputs']):
        parts=[f'({num}.0/{den}.0)*{names[tuple(p)]}' for p,num,den in terms]
        lines.append(f'  out[{j}] = '+(' + '.join(parts) if parts else '0.0')+';')
    lines += ['}',
              'extern "C" void spark_evolved_rollout_v1(double* s,const double* xs,std::size_t n,double gain) {',
              f'  double out[{n}];',
              '  for(std::size_t i=0;i<n;++i) {',
              '    spark_evolved_step_v1(s,xs[i],gain,out);',
              f'    for(std::size_t j=0;j<{n};++j) s[j]=out[j];',
              '  }','}','']
    return '\n'.join(lines)


def save(path: Path, model: EvolvedProgram):
    validate(model.payload)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(model.payload,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')


def load(path: Path):
    data=json.loads(path.read_text(encoding='utf-8'))
    validate(data)
    return EvolvedProgram(data,float('nan'),float('nan'),(),_complexity(data['outputs'],data['guard']))

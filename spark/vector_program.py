"""M10: learned coupled vector-state mathematical programs, compiled without matrices.

Synthesis: sparse rational polynomial search, optionally with a learned input
exception branch. Shared monomials are calculated once and reused by outputs.
Only training requires numpy; inference needs the Python stdlib or compiled C++.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

FORMAT = 'spark-vector-symbolic-v1'
MAX_REGISTERS = 8
MAX_TERMS = 16


def basis(n: int, degree: int = 2) -> list[tuple[int, ...]]:
    if not 1 <= n <= MAX_REGISTERS or degree not in (1, 2):
        raise ValueError('invalid register count or polynomial degree')
    m = n + 1
    exps = [tuple(0 for _ in range(m))]
    for i in range(m):
        e = [0] * m; e[i] = 1; exps.append(tuple(e))
    if degree == 2:
        for i in range(m):
            for j in range(i, m):
                e = [0] * m; e[i] += 1; e[j] += 1
                exps.append(tuple(e))
    return exps


def feature_values(state: tuple[float, ...] | list[float], x: float,
                   powers: list[tuple[int, ...]]) -> list[float]:
    values = list(state) + [x]
    return [math.prod(v ** p for v, p in zip(values, exponents)) for exponents in powers]


def _design(rows: list[tuple[tuple[float, ...], float, tuple[float, ...]]],
            powers: list[tuple[int, ...]]):
    return np.asarray([feature_values(s, x, powers) for s,x,_ in rows], dtype=np.float64)


def _fit_output(X, y, powers, *, max_terms: int = 7, denom: int = 16, tolerance=1e-18):
    """Greedy sparse model selection; each selected coefficient becomes a rational constant."""
    chosen = []
    residual = y.copy()
    norms = np.maximum(np.linalg.norm(X, axis=0), 1e-15)
    best = None
    for _ in range(max_terms):
        corr = np.abs(X.T @ residual) / norms
        corr[chosen] = -1
        k = int(np.argmax(corr))
        if corr[k] < 1e-10:
            break
        chosen.append(k)
        coefficients = np.linalg.lstsq(X[:, chosen], y, rcond=None)[0]
        rational = [Fraction(float(v)).limit_denominator(denom) for v in coefficients]
        terms = [[list(powers[i]), c.numerator, c.denominator]
                 for i,c in zip(chosen, rational) if c]
        approx = np.asarray([float(q) for q in rational])
        err = float(np.mean((X[:, chosen] @ approx - y) ** 2))
        residual = y - X[:, chosen] @ coefficients
        if best is None or (err, len(terms)) < (best[0],len(best[1])):
            best = (err,terms)
        if err < tolerance:
            break
    if best is None:
        best = (float(np.mean(y*y)), [])
    return best[1], best[0]


def _fit(rows, powers, *, max_terms=7, denom=16):
    if len(rows) < len(powers):
        raise ValueError('too few training observations')
    X = _design(rows, powers)
    ys = np.asarray([target for _,_,target in rows], dtype=np.float64)
    outputs = []
    for i in range(ys.shape[1]):
        terms, _ = _fit_output(X, ys[:,i], powers, max_terms=max_terms, denom=denom)
        outputs.append(terms)
    return outputs


def _eval_terms(terms, features, index):
    return sum((num/den) * features[index[tuple(p)]] for p,num,den in terms)


def _eval_program(program: dict, state, x):
    selected = 'then' if program['guard'] is not None and x == program['guard'] else 'else'
    outputs = program[selected]
    powers = [tuple(p) for p in program['powers']]
    index = {p:i for i,p in enumerate(powers)}
    features = feature_values(state, x, powers)
    return tuple(_eval_terms(terms, features, index) for terms in outputs)


def validate(program: dict) -> None:
    if not isinstance(program, dict) or set(program) != {'format','registers','degree','powers','guard','then','else'}:
        raise ValueError('invalid vector program fields')
    n=program['registers']; d=program['degree']
    if type(n) is not int or not 1 <= n <= MAX_REGISTERS or type(d) is not int or d not in (1,2):
        raise ValueError('invalid vector model dimension or degree')
    canonical = basis(n,d)
    if program['powers'] != [list(p) for p in canonical]:
        raise ValueError('noncanonical basis')
    guard=program['guard']
    if guard is not None and (type(guard) not in (int,float) or not math.isfinite(guard) or abs(guard)>1000):
        raise ValueError('invalid guard')
    for name in ('then','else'):
        rows=program[name]
        if not isinstance(rows,list) or len(rows)!=n:
            raise ValueError('incorrect number of register outputs')
        for terms in rows:
            if not isinstance(terms,list) or len(terms)>MAX_TERMS:
                raise ValueError('invalid expression terms')
            seen=set()
            for entry in terms:
                if not isinstance(entry,list) or len(entry)!=3:
                    raise ValueError('invalid expression term')
                powers,num,den=entry
                if (not isinstance(powers,list) or tuple(powers) not in canonical or
                    tuple(powers) in seen or type(num) is not int or type(den) is not int or
                    not 1 <= den <= 128 or abs(num)>1000000):
                    raise ValueError('invalid polynomial term')
                seen.add(tuple(powers))
    if program['format'] != FORMAT:
        raise ValueError('incorrect format')


@dataclass(frozen=True)
class VectorProgram:
    payload: dict
    train_mse: float
    tune_mse: float
    def step(self, state, x):
        if len(state)!=self.payload['registers']:
            raise ValueError('wrong state size')
        return _eval_program(self.payload,state,float(x))
    def rollout(self, state, xs):
        value=tuple(state)
        for x in xs:
            value=self.step(value,x)
        return value


def _score(rows, program):
    return sum(sum((a-b)**2 for a,b in zip(_eval_program(program,s,x),y))
               for s,x,y in rows)/(len(rows)*len(rows[0][0]))


def _cost(program):
    return sum(len(out) for group in ('then','else') for out in program[group])


def synthesize(rows: list[tuple[tuple[float, ...], float, tuple[float, ...]]],
               *, degree=2, denominator_limit=16, max_terms=7, min_branch=28) -> VectorProgram:
    if np is None:
        raise ImportError('numpy required only for offline equation discovery')
    if len(rows)<160 or not 1<=max_terms<=MAX_TERMS:
        raise ValueError('need >=160 observations and bounded complexity')
    n=len(rows[0][0]); powers=basis(n,degree)
    if not all(len(s)==len(y)==n and len(s)>0 and
               all(math.isfinite(float(v)) for v in (*s,float(x),*y)) for s,x,y in rows):
        raise ValueError('inconsistent or nonfinite observations')
    # Train-only deterministic inner holdout; never use externally supplied OOD tests.
    order=list(range(len(rows)))
    rng=np.random.default_rng(30410); rng.shuffle(order)
    split=max(40,len(order)//5)
    tune=[rows[i] for i in order[:split]]; fit=[rows[i] for i in order[split:]]
    candidate_guards=[None]
    counts={}
    for _,x,_ in fit:
        if int(x)==x:
            counts[x]=counts.get(x,0)+1
    for value,count in sorted(counts.items()):
        if min_branch<=count<=len(fit)-min_branch:
            candidate_guards.append(value)
    best=None
    for guard in candidate_guards:
        if guard is None:
            outs=_fit(fit,powers,max_terms=max_terms,denom=denominator_limit)
            program={'format':FORMAT,'registers':n,'degree':degree,'powers':[list(p) for p in powers],
                     'guard':None,'then':outs,'else':outs}
        else:
            yes=[r for r in fit if r[1]==guard]; no=[r for r in fit if r[1]!=guard]
            if min(len(yes),len(no))<max(min_branch,len(powers)):
                continue
            out_yes=_fit(yes,powers,max_terms=max_terms,denom=denominator_limit)
            out_no=_fit(no,powers,max_terms=max_terms,denom=denominator_limit)
            program={'format':FORMAT,'registers':n,'degree':degree,'powers':[list(p) for p in powers],
                     'guard':guard,'then':out_yes,'else':out_no}
        mse=_score(tune,program)
        # Prefer the smallest program within numerical recovery error.
        rank=(0 if mse<1e-14 else 1, 0.0 if mse<1e-14 else mse,
              (_cost(program) if guard is not None else _cost(program)//2), guard is not None)
        if best is None or rank<best[0]:
            best=(rank,guard,mse)
    if best is None:
        raise ValueError('could not select a valid program')
    guard=best[1]
    if guard is None:
        out=_fit(rows,powers,max_terms=max_terms,denom=denominator_limit)
        program={'format':FORMAT,'registers':n,'degree':degree,'powers':[list(p) for p in powers],
                 'guard':None,'then':out,'else':out}
    else:
        yes=[r for r in rows if r[1]==guard];no=[r for r in rows if r[1]!=guard]
        program={'format':FORMAT,'registers':n,'degree':degree,'powers':[list(p) for p in powers],
                 'guard':guard,'then':_fit(yes,powers,max_terms=max_terms,denom=denominator_limit),
                 'else':_fit(no,powers,max_terms=max_terms,denom=denominator_limit)}
    validate(program)
    return VectorProgram(program,_score(rows,program),best[2])


def save(path:Path,program:VectorProgram):
    validate(program.payload)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(program.payload,sort_keys=True,separators=(',',':'))+'\n')


def load(path:Path)->VectorProgram:
    doc=json.loads(path.read_text());validate(doc)
    return VectorProgram(doc,float('nan'),float('nan'))


def _cpp_expression(exponents):
    factors=[]
    for i,count in enumerate(exponents):
        factors.extend([f's[{i}]' if i<len(exponents)-1 else 'x']*count)
    return '*'.join(factors) if factors else '1.0'


def cpp_source(program:dict)->str:
    validate(program)
    n=program['registers']
    use={tuple(p) for branch in ('then','else') for output in program[branch] for p,_,_ in output}
    powers=[tuple(p) for p in program['powers']]
    names={p:f't{i}' for i,p in enumerate(powers) if p in use}
    body=[f'    const double {names[p]} = {_cpp_expression(p)};' for p in powers if p in names]
    def outputs(kind):
        lines=[]
        for i,terms in enumerate(program[kind]):
            summands=[f'({num}.0/{den}.0)*{names[tuple(p)]}' for p,num,den in terms]
            lines.append(f'    out[{i}] = '+(' + '.join(summands) if summands else '0.0')+';')
        return lines
    # Branch-specific optimized arithmetic, all outputs read the same old state.
    if program['guard'] is None:
        body += outputs('else')
    else:
        body += [f'    if (x == {float(program["guard"]):.17g}) {{']
        body += ['    '+line for line in outputs('then')]
        body += ['    } else {']
        body += ['    '+line for line in outputs('else')]
        body += ['    }']
    return ('// Generated from learned SPARK vector expression AST.\n'
            '#include <cstddef>\n'
            'extern "C" void spark_vector_step_v1(const double* s, double x, double* out) {\n'
            +'\n'.join(body)+'\n}\n'
            'extern "C" void spark_vector_rollout_v1(double* s, const double* xs, std::size_t length) {\n'
            f'  double next[{n}];\n'
            '  for (std::size_t i=0; i<length; ++i) {\n'
            '    spark_vector_step_v1(s,xs[i],next);\n'
            f'    for (std::size_t k=0; k<{n}; ++k) s[k]=next[k];\n'
            '  }\n}\n')

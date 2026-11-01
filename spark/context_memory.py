"""SPARK M18: discover context-dependent memory writes from delayed read errors.

The learner receives only raw observations (a, b, key, value), read requests,
and their eventual targets. There is NO relevance bit or write-instruction label.
A small, validated symbolic-grammar search chooses a compiled routing equation.

Research prototype: finite hand-designed grammar, explicit key/read API, and
supervised delayed queries. Not autonomous language-level relevance learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random

FORMAT = "spark-context-memory-v1"
FEATURES = ("a", "b", "sum", "difference", "product", "square_difference")
CUTS = (-0.25, 0.0, 0.25)
COMPARISONS = ("gt", "lt")


@dataclass(frozen=True)
class Observation:
    key: int
    value: float
    a: float
    b: float
    query: bool = False
    target: float = 0.0


@dataclass(frozen=True)
class Expression:
    feature: str
    comparison: str
    threshold: float

    def validate(self):
        if self.feature not in FEATURES or self.comparison not in COMPARISONS or type(self.threshold) not in (int, float):
            raise ValueError("invalid expression operation")
        if not math.isfinite(self.threshold) or float(self.threshold) not in CUTS:
            raise ValueError("unapproved threshold")

    def evaluate(self, a: float, b: float) -> bool:
        self.validate()
        t = {"a": a, "b": b, "sum": a+b, "difference": a-b,
             "product": a*b, "square_difference": a*a-b*b}[self.feature]
        return t > self.threshold if self.comparison == "gt" else t < self.threshold

    @property
    def complexity(self) -> int:
        return 2 if self.feature in ("a", "b") else 3 if self.feature in ("sum", "difference") else 4

    def to_json(self):
        return {"feature": self.feature, "comparison": self.comparison, "threshold": self.threshold}


def candidates(*, linear_only=False):
    fs=FEATURES[:4] if linear_only else FEATURES
    return [Expression(f,comparison,t) for f in fs for t in CUTS for comparison in COMPARISONS]


class ContextRouter:
    def __init__(self, expression: Expression, *, keys: int = 8):
        expression.validate()
        if type(keys) is not int or not 1 <= keys <= 64:
            raise ValueError("invalid memory key capacity")
        self.expression=expression
        self.keys=keys

    def replay(self, sequence, *, return_reads=False):
        memory=[0.0]*self.keys
        hits=reads=0
        errors=0.0
        predictions=[]
        for e in sequence:
            if (type(e.key) is not int or not 0<=e.key<self.keys or
                not all(math.isfinite(v) for v in (e.a,e.b,e.value,e.target))):
                raise ValueError("invalid context-memory observation")
            if e.query:
                pred=memory[e.key]
                predictions.append(pred)
                errors+=(pred-e.target)**2
                hits+=abs(pred-e.target)<1e-10
                reads+=1
            elif self.expression.evaluate(e.a,e.b):
                memory[e.key]=e.value
        if return_reads:
            return errors,hits,reads,predictions
        return errors,hits,reads

    def save(self,path):
        p=Path(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({"format": FORMAT,"keys": self.keys,
                                 "expression": self.expression.to_json()},sort_keys=True)+"\n")

    @classmethod
    def load(cls,path):
        doc=json.loads(Path(path).read_text())
        if set(doc)!={"format","keys","expression"} or doc["format"]!=FORMAT:
            raise ValueError("unknown program serialization")
        if not isinstance(doc["expression"],dict) or set(doc["expression"])!={"feature","comparison","threshold"}:
            raise ValueError("unknown routing expression")
        return cls(Expression(**doc["expression"]),keys=doc["keys"])

    def cpp_source(self):
        """Compile the DISCOVERED equation, not a hardcoded relevance rule."""
        ex=self.expression
        ex.validate()
        expr={"a":"a","b":"b","sum":"(a+b)","difference":"(a-b)",
              "product":"(a*b)","square_difference":"(a*a-b*b)"}[ex.feature]
        sign=">" if ex.comparison=="gt" else "<"
        return f'''// SPARK M18: learned context-memory routing equation.
// No relevance indicator, task-specific lookup table or neural matrix.
#include <cmath>
#include <cstddef>
struct SparkContextMemory {{ double slots[{self.keys}]{{}}; }};
extern "C" void spark_context_reset_v1(SparkContextMemory* m) {{
  if (!m) return;
  for(int i=0;i<{self.keys};++i)m->slots[i]=0.0;
}}
extern "C" int spark_context_event_v1(SparkContextMemory* m,int key,double a,double b,double value,int query,double* result) {{
  if(!m||!result||key<0||key>={self.keys}||!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(value))return 0;
  if(!query && ({expr} {sign} {ex.threshold:.17g}))m->slots[key]=value;
  *result=m->slots[key];return 1;
}}
'''


def fit(episodes, *, keys=8, linear_only=False, penalty=0.0001):
    """Fit only from the error at eventual read events, never from write labels.

    Candidate grammar is deliberately finite, making it possible to audit that
    every training signal comes from query targets. Candidate selection is
    train-only. Held-out queries must never be supplied to this function.
    """
    if not episodes or len(episodes)<8:
        raise ValueError("need independent training episodes")
    ranked=[]
    for expr in candidates(linear_only=linear_only):
        router=ContextRouter(expr,keys=keys)
        total=reads=0
        for sequence in episodes:
            e,_,n=router.replay(sequence)
            total+=e;reads+=n
        if reads<1:raise ValueError("episodes contain no read supervision")
        loss=total/reads
        ranked.append((loss+penalty*expr.complexity,loss,expr.complexity,expr.feature,expr.comparison,expr.threshold,expr))
    best=min(ranked)
    return ContextRouter(best[-1],keys=keys),{"train_query_mse":best[1],"candidates":len(ranked),
                                               "selection_cost":best[0],"description_cost":best[2]}


def _context(rng: random.Random, *, scale=1.0):
    # Balanced sign quadrants and independently sampled magnitudes. No direct
    # relevance bit is presented to the learner. The generator alone knows
    # the target behavior that determines query answers.
    sa=rng.choice((-1.,1.));sb=rng.choice((-1.,1.))
    return (sa*rng.uniform(.12,scale),sb*rng.uniform(.12,scale))


def episodes(seed, count, *, keys=4, distractors=(5,18), scale=1.0, queries=3,
             mode="same_sign"):
    """Generate streaming queries from an unknown nonlinear context rule.

    The episode construction must know a latent rule to calculate true query
    targets, but learners get only the Observation stream and read targets.
    """
    if mode not in ("same_sign","opposite_sign"):
        raise ValueError("invalid synthetic hidden mode")
    if type(keys) is not int or not 1<=keys<=64 or not (0<distractors[0]<=distractors[1]) or queries<1:
        raise ValueError("invalid data-generation configuration")
    rng=random.Random(seed)
    results=[]
    for _ in range(count):
        memory=[0.0]*keys;sequence=[]
        def emit(key,value,a,b):
            # The simulator's rule is NEVER added as a model feature or label.
            relevant=(a*b>0) if mode=="same_sign" else (a*b<0)
            if relevant:memory[key]=value
            sequence.append(Observation(key,value,a,b))
        for k in range(keys):
            # No relevance cue: initialize through the same raw-context API.
            # Draw random contexts until an event happens to be relevant.
            for _attempt in range(30):
                a,b=_context(rng,scale=scale)
                if ((a*b>0) if mode=="same_sign" else (a*b<0)):
                    emit(k,rng.uniform(-1.5,1.5),a,b)
                    break
            else:raise RuntimeError("could not sample initial context")
        for _ in range(queries):
            for _ in range(rng.randint(*distractors)):
                key=rng.randrange(keys)
                a,b=_context(rng,scale=scale)
                emit(key,rng.uniform(-1.5,1.5),a,b)
            key=rng.randrange(keys)
            sequence.append(Observation(key,0.0,0.0,0.0,True,memory[key]))
        results.append(sequence)
    return results


def metrics(router,streams):
    total=hits=reads=0
    for seq in streams:
        err,h,n=router.replay(seq)
        total+=err;hits+=h;reads+=n
    return {"episodes":len(streams),"reads":reads,"correct":hits,
            "query_mse":total/max(reads,1),"accuracy":hits/max(reads,1)}


def adapt_prequential(start: ContextRouter, mode_stages, *, window=24, refresh_every=6):
    """Adapt the *program structure* using only PAST read feedback.

    Each new episode is predicted first. Its read targets are added to the
    rolling fit window only after scoring. No mode labels go to the learner.
    """
    policy=start
    history=[];stage_reports=[]
    for stage,streams in enumerate(mode_stages):
        errors=[];correct=[];exprs=[]
        for seq in streams:
            err,h,n=policy.replay(seq)
            errors.append(err/max(n,1));correct.append(h/max(n,1))
            exprs.append(policy.expression.to_json())
            history.append(seq)
            if len(history)>window:history.pop(0)
            if len(history)>=12 and len(errors)%refresh_every==0:
                policy,_=fit(history,keys=policy.keys)
        stage_reports.append({"stage":stage,"examples":len(streams),
          "first_12_mse":sum(errors[:12])/12,"last_24_mse":sum(errors[-24:])/24,
          "last_24_accuracy":sum(correct[-24:])/24,
          "final_expression":policy.expression.to_json(),
          "first_expression":exprs[0]})
    return stage_reports

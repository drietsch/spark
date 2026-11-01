"""M19: search reusable Boolean compositions of mathematical memory predicates.

Training receives sequences and delayed read targets, never write/relevance labels.
The grammar and expression-depth bound are explicit limitations.
Inference uses a validated expression AST compiled to C++ scalar control flow.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from spark.context_memory import Observation, Expression, candidates, ContextRouter

FORMAT = 'spark-composed-memory-v1'

@dataclass(frozen=True)
class Program:
    op: str
    left: Expression
    right: Expression | None = None

    def validate(self):
        self.left.validate()
        if self.op == 'atom':
            if self.right is not None:raise ValueError('invalid atomic program')
        elif self.op in ('and','or'):
            if not isinstance(self.right,Expression):raise ValueError('invalid compound program')
            self.right.validate()
            if self.left.to_json()==self.right.to_json():raise ValueError('duplicate condition')
        else:raise ValueError('unsupported program operation')

    def eval(self,a,b):
        if self.op=='atom':return self.left.evaluate(a,b)
        if self.op=='and':return self.left.evaluate(a,b) and self.right.evaluate(a,b)
        return self.left.evaluate(a,b) or self.right.evaluate(a,b)

    @property
    def complexity(self):
        return self.left.complexity+(0 if self.right is None else self.right.complexity+2)

    def json(self):
        self.validate()
        return {'op':self.op,'left':self.left.to_json(),'right':None if self.right is None else self.right.to_json()}

    @classmethod
    def from_json(cls,raw):
        if not isinstance(raw,dict) or set(raw)!={'op','left','right'}:raise ValueError('invalid AST fields')
        def leaf(p):
            if not isinstance(p,dict) or set(p)!={'feature','comparison','threshold'}:raise ValueError('invalid atom')
            e=Expression(**p);e.validate();return e
        result=cls(raw['op'],leaf(raw['left']),None if raw['right'] is None else leaf(raw['right']))
        result.validate()
        return result


def _score(data,program,keys):
    error=reads=hits=0
    for episode in data:
        memory=[0.]*keys
        for e in episode:
            if e.query:
                delta=memory[e.key]-e.target
                error+=delta*delta;reads+=1;hits+=abs(delta)<1e-10
            elif program.eval(e.a,e.b):memory[e.key]=e.value
    if not reads:raise ValueError('no read targets')
    return error/reads,hits,reads


def search(data,*,keys=8,beam=36,penalty=1e-5,composition=True):
    """Rank atoms by delayed read loss; combine them under AND/OR.

    Beam-size=36 includes all atomic predicates and provides exhaustive one-step
    Boolean composition, not arbitrary program synthesis. No relevance labels.
    """
    if not data or len(data)<8 or type(keys) is not int or not 1<=keys<=64:
        raise ValueError('invalid program-discovery data')
    atoms=candidates()
    singles=[]
    for a in atoms:
        p=Program('atom',a);loss,_,_=_score(data,p,keys)
        singles.append((loss+penalty*p.complexity,p.complexity,a.feature,a.comparison,a.threshold,p))
    singles.sort(key=lambda r:r[:-1]);short=[s[-1].left for s in singles[:beam]]
    rankings=singles[:]
    if composition:
        for i in range(len(short)):
            for j in range(i+1,len(short)):
                for op in ('and','or'):
                    p=Program(op,short[i],short[j]);loss,_,_=_score(data,p,keys)
                    rankings.append((loss+penalty*p.complexity,p.complexity,
                                     p.left.feature+':'+p.right.feature,p.op,
                                     p.left.threshold+p.right.threshold,p))
    rankings.sort(key=lambda r:r[:-1]);winner=rankings[0][-1]
    return ComposedRouter(winner,keys),{'candidates':len(rankings),'atomic_candidates':len(singles),
        'train_loss':_score(data,winner,keys)[0],'complexity':winner.complexity,
        'program':winner.json()}


@dataclass
class ComposedRouter:
    program: Program
    keys: int
    def __post_init__(self):
        self.program.validate()
        if type(self.keys) is not int or not 1<=self.keys<=64:raise ValueError('invalid capacity')

    def evaluate(self,data):
        e,h,n=_score(data,self.program,self.keys)
        return {'reads':n,'correct':h,'query_mse':e,'accuracy':h/n}

    def save(self,path):
        Path(path).write_text(json.dumps({'format':FORMAT,'keys':self.keys,'program':self.program.json()},sort_keys=True)+'\n')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if not isinstance(d,dict) or set(d)!={'format','keys','program'} or d['format']!=FORMAT:
            raise ValueError('invalid serialization')
        return cls(Program.from_json(d['program']),d['keys'])

    def cpp(self):
        self.program.validate()
        def leaf(p):
            s={'a':'a','b':'b','sum':'(a+b)','difference':'(a-b)','product':'(a*b)',
               'square_difference':'(a*a-b*b)'}[p.feature]
            op='>' if p.comparison=='gt' else '<'
            return f'({s}{op}{p.threshold:.17g})'
        expr=leaf(self.program.left)
        if self.program.right:
            expr=f'({expr} '+('&&' if self.program.op=='and' else '||')+f' {leaf(self.program.right)})'
        return f'''// SPARK M19 learned compound mathematical memory rule.
#include <cmath>
#include <cstddef>
struct M19State{{double slots[{self.keys}]{{}};}};
extern "C" void m19_reset(M19State* m){{if(m)for(int i=0;i<{self.keys};++i)m->slots[i]=0.;}}
extern "C" int m19_event(M19State* m,int k,double a,double b,double v,int query,double* out){{
 if(!m||!out||k<0||k>={self.keys}||!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(v))return 0;
 if(!query&&{expr})m->slots[k]=v;
 *out=m->slots[k];return 1;
}}
'''


def generate(seed,count,*,keys=4,distractors=(5,18),scale=1.0,queries=3,mode='compound'):
    """Hidden target exists in simulator only; data has no relevance label."""
    if mode not in ('compound','reverse') or not 1<=keys<=64 or not 1<=distractors[0]<=distractors[1]:
        raise ValueError('invalid episode parameters')
    rng=random.Random(seed);episodes=[]
    for _ in range(count):
        truth=[0.]*keys;events=[]
        def append(k,val,a,b):
            selected=(a*b>0 and a+b>0.25)
            if mode=='reverse':selected=not selected
            if selected:truth[k]=val
            events.append(Observation(k,val,a,b))
        for k in range(keys):
            # Guaranteed initial retained value, no significance bit in stream.
            append(k,rng.uniform(-1.4,1.4),1.0,1.0) if mode=='compound' else append(k,rng.uniform(-1.4,1.4),-1.0,1.0)
        for _ in range(queries):
            for _ in range(rng.randint(*distractors)):
                k=rng.randrange(keys)
                a=rng.choice((-1.,1.))*rng.uniform(.03, min(.2,scale) if rng.random()<.4 else scale)
                b=rng.choice((-1.,1.))*rng.uniform(.03, min(.2,scale) if rng.random()<.4 else scale)
                append(k,rng.uniform(-1.4,1.4),a,b)
            k=rng.randrange(keys)
            events.append(Observation(k,0.,0.,0.,True,truth[k]))
        episodes.append(events)
    return episodes

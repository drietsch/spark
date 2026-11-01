"""SPARK M21: joint address/write program search and online neural value semantics.

No all-write calibration and no write/address labels. A single delayed query loss
selects address and Boolean program; a mutable tiny neural function interprets
what a stored observation means, trained only after a read query is answered.
This is a finite-grammar synthetic experiment, not open-ended intelligence.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from spark.addressed_program import (
    Address, AddressedProgram, Observation, PROJECTIONS, ATOMS, discover_policy,
    eval_node, generate as base_generate,
)
from spark.plastic_function import PlasticFunction

FORMAT = 'spark-joint-plastic-memory-v1'


def generate(seed: int, episodes: int, *, keys: int = 4, distractors=(5, 18),
             queries=4, scale=1.3, mode: int = 1):
    """Synthesizes query targets only; does not return write labels or identities.

    The hidden generator implements a three-condition rule and a nonlinear
    payload. Neither appears in the learner's search/training code.
    """
    if mode not in (-1, 1):
        raise ValueError('invalid hidden experiment mode')
    raw = base_generate(seed, episodes, keys=keys, distractors=distractors,
                        queries=queries, scale=scale, calibration=False)
    results = []
    for seq in raw:
        truth = [0.] * keys
        altered = []
        for e in seq:
            key = round(e.u - e.v)
            if e.query:
                altered.append(Observation(e.u,e.v,0.,e.a,e.b,True,truth[key]))
            else:
                # Ground truth ONLY belongs to the synthetic task generator.
                relevant = (e.a * e.b > 0 and e.a + e.b > 0.25 and e.a - e.b < 0.25)
                if relevant:
                    truth[key] = e.value + mode * 0.17 * math.tanh(
                        1.4 * e.value + 0.35 * e.a - 0.25 * e.b)
                altered.append(Observation(e.u,e.v,e.value,e.a,e.b))
        results.append(altered)
    return results


def _validate_streams(streams, capacity):
    if not streams or len(streams) < 4 or type(capacity) is not int or not 1 <= capacity <= 64:
        raise ValueError('invalid streams/capacity')
    reads = 0
    for seq in streams:
        if not seq: raise ValueError('empty sequence')
        for e in seq:
            if not isinstance(e, Observation) or not all(map(math.isfinite,
                    (e.u,e.v,e.value,e.a,e.b,e.target))):
                raise ValueError('invalid observation')
            if e.query:
                reads += 1
                if e.value != 0.: raise ValueError('read observations must not contain write values')
            elif e.target != 0.:
                raise ValueError('write events must not contain future query targets')
    if reads == 0:raise ValueError('no delayed query targets')


def joint_search(streams, *, capacity=8, coarse_count=32, finalist_count=2, beam=90):
    """Joint structure selection from one delayed read-loss objective.

    First fit *each address paired with a write expression*, using a small
    training-only coarse slice. Refine the most promising address/program
    combinations on the entire training set with a deeper expression search.
    No known-valid-writes phase or address oracle enters any stage.
    """
    _validate_streams(streams, capacity)
    if not 12 <= coarse_count <= len(streams) or not 1<=finalist_count<=len(PROJECTIONS):
        raise ValueError('invalid joint-search budget')
    rankings=[];coarse_programs=0
    for cu,cv in PROJECTIONS:
        address = Address(cu,cv)
        model, info = discover_policy(streams[:coarse_count], address, capacity,
                                      depth=2, beam=beam)
        # Score a joint candidate on all training queries. Ranking never uses
        # any future held-out stream or separate all-write calibration corpus.
        fit = replay_joint(streams,model.address,model.policy,capacity)
        rankings.append((fit['mse'], info['complexity'], abs(cu)+abs(cv),cu,cv))
        coarse_programs+=info['searched']
    rankings.sort()
    finalists=[]
    for _,_,_,cu,cv in rankings[:finalist_count]:
        model,info=discover_policy(streams,Address(cu,cv),capacity,depth=3,beam=beam)
        scored=replay_joint(streams,model.address,model.policy,capacity)
        finalists.append((scored['mse'],info['complexity'],cu,cv,model,info))
    finalists.sort(key=lambda r:(r[0]+1e-6*r[1],r[1],r[2],r[3]))
    mse,cost,cu,cv,chosen,info=finalists[0]
    return chosen,{'strategy':'joint address+policy query-loss, coarse-to-fine',
       'calibration_episodes':0,'coarse_candidates':len(PROJECTIONS),
       'coarse_programs_searched':coarse_programs,'full_programs_searched':sum(r[5]['searched'] for r in finalists),
       'selected_address':{'cu':cu,'cv':cv},'selection_mse':mse,
       'cost':cost,'policy':chosen.policy,'coarse_ranking':[
           {'address':[r[3],r[4]],'loss':r[0]} for r in rankings]}


def replay_joint(streams, address, policy, capacity):
    """Score a candidate from delayed targets using raw payloads only.

    This supplies the joint search's surrogate before neural residual learning.
    """
    address.validate()
    sq=hits=reads=0
    for seq in streams:
        slots=[0.]*capacity
        for e in seq:
            k=address.get(e,capacity)
            if e.query:
                pred=slots[k] if k>=0 else 0.
                error=pred-e.target
                sq+=error*error;hits+=abs(error)<1e-8;reads+=1
            elif k>=0 and eval_node(policy,e.a,e.b):
                slots[k]=e.value
    return {'reads':reads,'exact':hits,'mse':sq/reads}


@dataclass
class JointMemory:
    address: Address
    policy: dict
    capacity: int
    function: PlasticFunction
    steps: int=0

    def __post_init__(self):
        AddressedProgram(self.address,self.policy,self.capacity)
        self.function.validate()
        if type(self.steps) is not int or self.steps < 0:raise ValueError('invalid online step count')

    def process(self, episode, *, learn=False, lr=.04):
        """Strict prequential evaluation: read prediction before feedback update."""
        slots=[None]*self.capacity
        errors=[];predictions=[]
        for e in episode:
            if not isinstance(e, Observation) or not all(math.isfinite(z) for z in
                (e.u,e.v,e.a,e.b,e.value,e.target)):
                raise ValueError('invalid observation')
            k=self.address.get(e,self.capacity)
            if e.query:
                if k < 0 or slots[k] is None:
                    pred=0.
                else:
                    z=slots[k]
                    correction,cache=self.function.forward(z)
                    pred=z[0]+correction
                    if learn:
                        self.function.update(z,pred-e.target,lr,cache)
                        self.steps+=1
                predictions.append(pred)
                errors.append((pred-e.target)**2)
            elif k>=0 and eval_node(self.policy,e.a,e.b):
                slots[k]=(e.value,e.a,e.b)
        return predictions,errors

    def evaluate(self, streams, *, learn=False, lr=.04):
        _validate_streams(streams,self.capacity)
        losses=[];predictions=[]
        for seq in streams:
            pr,err=self.process(seq,learn=learn,lr=lr)
            predictions.extend(pr);losses.extend(err)
        return {'reads':len(losses),'mse':sum(losses)/len(losses),
                'last_100_mse':sum(losses[-100:])/len(losses[-100:]),
                'predictions':predictions}

    def save(self,path):
        self.function.validate()
        obj={'format':FORMAT,'address':self.address.json(),'policy':self.policy,
             'capacity':self.capacity,'active':self.function.active,
             'units':self.function.units,'bias':self.function.bias,'steps':self.steps}
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')

    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_text())
        if not isinstance(obj,dict) or set(obj)!={'format','address','policy','capacity','active','units','bias','steps'} or obj['format']!=FORMAT:
            raise ValueError('unsupported joint memory format')
        addr=obj['address']
        if not isinstance(addr,dict) or set(addr)!={'cu','cv'}:raise ValueError('invalid address')
        func=PlasticFunction(len(obj['units']),obj['active'],obj['units'],obj['bias'])
        return cls(Address(**addr),obj['policy'],obj['capacity'],func,obj['steps'])

    def cpp(self):
        """C++17 fixed arithmetic/AST + *live* gated-dense neural parameters.

        The trained neural weights and active-unit count are runtime data, not
        inlined numerical constants. Online gradient updates run natively.
        """
        from spark.addressed_program import ATOMS
        cap=self.capacity; width=self.function.capacity
        def node_cpp(n):
            if n['op']=='atom':
                p=ATOMS[n['index']]
                z={'a':'a','b':'b','sum':'(a+b)','difference':'(a-b)',
                   'product':'(a*b)','square_difference':'(a*a-b*b)'}[p.feature]
                return f'({z} '+('>' if p.comparison=='gt' else '<')+f' {p.threshold:.17g})'
            return f'({node_cpp(n["left"])} '+('&&' if n['op']=='and' else '||')+f' {node_cpp(n["right"])})'
        compiled_policy=node_cpp(self.policy)
        cu,cv=self.address.cu,self.address.cv
        return f'''// SPARK M21: validated symbolic address/policy with LIVE neural value function.
// No hardcoded neural coefficients and no oracle write/address labels.
#include <cmath>
#include <cstddef>
#include <new>
#include <algorithm>
struct M21 {{double slots[{cap}][3]{{}};bool valid[{cap}]{{}};
 double w[{width}][9]{{}};double bias{{}};int active{{1}};}};
static inline double sg(double x) {{if(x>=0)return 1./(1.+std::exp(-x));
 const double e=std::exp(x);return e/(1.+e);}}
static inline int adr(double u,double v) {{
 if(!std::isfinite(u)||!std::isfinite(v))return -1;
 const double k={cu}*u+{cv}*v;
 if(!std::isfinite(k)||k<-.5||k>={cap}-.5)return -1;
 return static_cast<int>(std::floor(k+.5));}}
static inline bool keep(double a,double b){{return {compiled_policy};}}
extern "C" void* m21_create() {{return new(std::nothrow) M21();}}
extern "C" void m21_destroy(void* p) {{delete static_cast<M21*>(p);}}
extern "C" std::size_t m21_parameters(){{return {width}*9+1;}}
extern "C" int m21_set(void* p,const double* f,std::size_t n,int active) {{
 if(!p||!f||n!={width}*9+1||active<1||active>{width})return 0;
 for(std::size_t i=0;i<n;++i)if(!std::isfinite(f[i])||std::abs(f[i])>8.)return 0;
 M21& s=*static_cast<M21*>(p);
 for(int i=0;i<{width};++i)for(int j=0;j<9;++j)s.w[i][j]=f[i*9+j];
 s.bias=f[{width}*9];s.active=active;return 1;}}
extern "C" int m21_get(void* p,double* f,std::size_t n) {{
 if(!p||!f||n!={width}*9+1)return 0;
 M21& s=*static_cast<M21*>(p);
 for(int i=0;i<{width};++i)for(int j=0;j<9;++j)f[i*9+j]=s.w[i][j];
 f[{width}*9]=s.bias;return 1;}}
extern "C" void m21_reset(void* p){{if(!p)return;M21& m=*static_cast<M21*>(p);
 for(int i=0;i<{cap};++i)m.valid[i]=false;}}
extern "C" int m21_event(void* p,double u,double v,double a,double b,
 double value,int query,double target,int learn,double lr,double* out) {{
 if(!p||!out||!std::isfinite(u)||!std::isfinite(v)||!std::isfinite(a)||
 !std::isfinite(b)||!std::isfinite(value)||!std::isfinite(target)||
 (learn&&(!std::isfinite(lr)||lr<=0||lr>.15)))return 0;
 M21& m=*static_cast<M21*>(p);const int key=adr(u,v);
 if(!query) {{
   if(key>=0&&keep(a,b)){{m.slots[key][0]=value;m.slots[key][1]=a;
   m.slots[key][2]=b;m.valid[key]=true;}}
   *out=0.;return 1;
 }}
 if(key<0||!m.valid[key]){{*out=0.;return 1;}}
 const double* z=m.slots[key];double h[{width}],g[{width}];double q=m.bias;
 for(int j=0;j<m.active;++j){{
   double* w=m.w[j];double aa=w[3],bb=w[7];
   for(int i=0;i<3;++i){{aa+=w[i]*z[i];bb+=w[4+i]*z[i];}}
   h[j]=std::tanh(aa);g[j]=sg(bb);q+=w[8]*h[j]*g[j];
 }}
 *out=z[0]+q;
 if(learn){{
   const double gradient=std::clamp(*out-target,-2.,2.);
   for(int j=0;j<m.active;++j){{
     double* w=m.w[j];const double old=w[8];
     const double da=gradient*old*g[j]*(1-h[j]*h[j]);
     const double dg=gradient*old*h[j]*g[j]*(1-g[j]);
     for(int i=0;i<3;++i)w[i]=std::clamp(w[i]-lr*da*z[i],-8.,8.);
     w[3]=std::clamp(w[3]-lr*da,-8.,8.);
     for(int i=0;i<3;++i)w[4+i]=std::clamp(w[4+i]-lr*dg*z[i],-8.,8.);
     w[7]=std::clamp(w[7]-lr*dg,-8.,8.);
     w[8]=std::clamp(w[8]-lr*gradient*h[j]*g[j],-8.,8.);
   }}
   m.bias=std::clamp(m.bias-lr*gradient,-8.,8.);
 }}
 return 1;
}}
'''

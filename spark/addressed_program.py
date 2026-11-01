"""M20: learn address expression and nested write program from delayed read errors.

Synthetic research-only grammar. The learner receives raw observations with no
integer address and no relevance/write labels. Training identifies an address
function from calibration query losses, then composes a separate write program
from query losses. Native C++ executes both discovered mathematical functions.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from spark.context_memory import candidates

FORMAT = 'spark-addressed-program-v1'
ATOMS = candidates()
# Candidate mathematical address projections, NOT per-object lookup entries.
PROJECTIONS = ((1,0),(0,1),(1,-1),(1,1),(2,-1),(-1,1),(2,1),(1,-2))

@dataclass(frozen=True)
class Observation:
    u: float
    v: float
    value: float
    a: float
    b: float
    query: bool=False
    target: float=0.0

@dataclass(frozen=True)
class Address:
    cu: int
    cv: int
    def validate(self):
        if (type(self.cu) is not int or type(self.cv) is not int or
            (self.cu,self.cv) not in PROJECTIONS):
            raise ValueError('unsupported mathematical address expression')
    def get(self, e:Observation,capacity:int):
        raw=self.cu*e.u+self.cv*e.v
        if not math.isfinite(raw):raise ValueError('invalid address observation')
        idx=math.floor(raw+0.5)
        return idx if 0<=idx<capacity else -1
    def json(self):
        self.validate();return {'cu':self.cu,'cv':self.cv}

# A program is a validated tree: each leaf is an atom from the M18 grammar;
# the topology is learned and may consist of three (or fewer) predicates.
def validate_node(node,depth=0):
    if depth>4 or not isinstance(node,dict):raise ValueError('invalid expression AST')
    if node.get('op')=='atom':
        if set(node)!={'op','index'} or type(node['index']) is not int or not 0<=node['index']<len(ATOMS):
            raise ValueError('invalid atomic predicate')
    elif node.get('op') in ('and','or'):
        if set(node)!={'op','left','right'}:raise ValueError('invalid Boolean composition')
        validate_node(node['left'],depth+1);validate_node(node['right'],depth+1)
    else:raise ValueError('unsupported program operation')

def atom(i):return {'op':'atom','index':i}
def combine(op,l,r):return {'op':op,'left':l,'right':r}

def eval_node(node,a,b):
    op=node['op']
    if op=='atom':return ATOMS[node['index']].evaluate(a,b)
    if op=='and':return eval_node(node['left'],a,b) and eval_node(node['right'],a,b)
    return eval_node(node['left'],a,b) or eval_node(node['right'],a,b)

def node_cost(node):
    return 1 if node['op']=='atom' else 2+node_cost(node['left'])+node_cost(node['right'])

def _check(episodes,capacity):
    if not isinstance(episodes,list) or len(episodes)<8 or not 1<=capacity<=64:
        raise ValueError('insufficient episodes or invalid capacity')
    for seq in episodes:
        if not seq:raise ValueError('empty sequence')
        for e in seq:
            if not isinstance(e,Observation) or not all(math.isfinite(v) for v in
                    (e.u,e.v,e.value,e.a,e.b,e.target)):
                raise ValueError('malformed or nonfinite observation')

def replay(episodes,address,policy,capacity,*,always=False):
    if policy is not None:validate_node(policy)
    address.validate();_check(episodes,capacity)
    loss=hits=reads=0
    for seq in episodes:
        slots=[0.]*capacity
        for e in seq:
            idx=address.get(e,capacity)
            if e.query:
                pred=slots[idx] if idx>=0 else 0.
                delta=pred-e.target;loss+=delta*delta
                hits+=abs(delta)<1e-9;reads+=1
            elif idx>=0 and (always or (policy is not None and eval_node(policy,e.a,e.b))):
                slots[idx]=e.value
    if reads==0:raise ValueError('query targets required')
    return {'reads':reads,'correct':hits,'mse':loss/reads,'accuracy':hits/reads}


def learn_address(calibration,capacity=8):
    """All calibration observations are valid updates; no hidden key labels.

    The calibration protocol is distinct from write-relevance training. Only
    subsequent query targets determine the best projection. All candidate
    projections are small explicit arithmetic functions, no dictionary lookup.
    """
    _check(calibration,capacity)
    ranked=[]
    for cu,cv in PROJECTIONS:
        candidate=Address(cu,cv)
        result=replay(calibration,candidate,None,capacity,always=True)
        ranked.append((result['mse'],abs(cu)+abs(cv),cu,cv,result))
    ranked.sort(key=lambda t:t[:4])
    r=ranked[0]
    return Address(r[2],r[3]),{'candidates':len(ranked),'train_mse':r[0],
                              'train_correct':r[4]['correct'],'train_reads':r[4]['reads']}


def _encoded(episodes,address,capacity):
    """Compile training observations into address and predicate truth bitsets.

    The search never receives ground-truth write labels or hidden addresses.
    """
    encoded=[]
    for seq in episodes:
        items=[]
        for e in seq:
            idx=address.get(e,capacity)
            mask=0
            if not e.query:
                for j,p in enumerate(ATOMS):
                    z={'a':e.a,'b':e.b,'sum':e.a+e.b,'difference':e.a-e.b,
                       'product':e.a*e.b,'square_difference':e.a*e.a-e.b*e.b}[p.feature]
                    if (z>p.threshold if p.comparison=='gt' else z<p.threshold):mask|=1<<j
            items.append((idx,e.value,e.query,e.target,mask))
        encoded.append(items)
    return encoded


def _score_encoded(data, predicate, *, max_loss=float('inf')):
    # predicate(mask) is a tiny closure over one to three atomic masks.
    err=reads=hits=0
    capacity=data.capacity
    for seq in data.events:
        slots=[0.]*capacity
        for idx,value,query,target,mask in seq:
            if query:
                pred=slots[idx] if idx>=0 else 0.
                delta=pred-target;err+=delta*delta
                hits+=abs(delta)<1e-9;reads+=1
            elif idx>=0 and predicate(mask):slots[idx]=value
        # Cumulative error is nonnegative; prune hopeless candidates.
        if err>max_loss:return float('inf'),0
    return err/data.reads,hits

class Encoded:
    def __init__(self,events,capacity):
        self.events=events;self.capacity=capacity
        self.reads=sum(q for seq in events for _,_,q,_,_ in seq)
        if not self.reads:raise ValueError('no query targets')

def discover_policy(episodes,address,capacity=8,*,depth=3,beam=100,penalty=1e-6):
    """Bounded three-atom structural beam, fitted only on delayed read loss.

    Enumerates all 1/2-atom programs, then combines top diverse pairs with a
    third atom. The exact final program is not inserted as a search candidate.
    """
    if depth not in (1,2,3) or not 1<=beam<=300:raise ValueError('invalid structure search limits')
    _check(episodes,capacity);address.validate()
    data=Encoded(_encoded(episodes,address,capacity),capacity)
    n=len(ATOMS);record=[];two=[];count=0
    def consider(node,pred,store=None):
        nonlocal count
        mse,hits=_score_encoded(data,pred)
        count+=1
        cost=node_cost(node)
        rank=(mse+penalty*cost,cost)
        result=(rank,mse,hits,node)
        record.append(result)
        if store is not None:store.append(result)
    for i in range(n):
        flag=1<<i
        consider(atom(i),lambda mask,f=flag:bool(mask&f))
    if depth>=2:
        for i in range(n):
            for j in range(i+1,n):
                a=1<<i;b=1<<j
                consider(combine('and',atom(i),atom(j)),lambda mask,a=a,b=b:bool(mask&a and mask&b),two)
                consider(combine('or',atom(i),atom(j)),lambda mask,a=a,b=b:bool(mask&a or mask&b),two)
    if depth>=3:
        # Preserve candidates from both families. Solely choosing lowest-loss
        # pairs can prematurely discard a useful intermediate conjunction.
        selected=[]
        for op in ('and','or'):
            family=sorted((r for r in two if r[3]['op']==op),key=lambda t:t[0])
            selected.extend(family[:beam])
        for _,_,_,pair in selected:
            i=pair['left']['index'];j=pair['right']['index']
            pa=1<<i;pb=1<<j
            for k in range(n):
                if k==i or k==j:continue
                pc=1<<k
                for root in ('and','or'):
                    node=combine(root,pair,atom(k))
                    if pair['op']=='and':
                        pred=(lambda m,a=pa,b=pb,c=pc:bool((m&a and m&b) and m&c)) if root=='and' else (lambda m,a=pa,b=pb,c=pc:bool((m&a and m&b) or m&c))
                    else:
                        pred=(lambda m,a=pa,b=pb,c=pc:bool((m&a or m&b) and m&c)) if root=='and' else (lambda m,a=pa,b=pb,c=pc:bool((m&a or m&b) or m&c))
                    consider(node,pred)
    record.sort(key=lambda t:t[0]);winner=record[0]
    model=AddressedProgram(address,winner[3],capacity)
    return model,{'searched':count,'depth':depth,'pair_beam_per_operator':beam,
                  'train_mse':winner[1],'train_correct':winner[2],'train_reads':data.reads,
                  'complexity':node_cost(winner[3]),'program':winner[3]}

@dataclass(frozen=True)
class AddressedProgram:
    address:Address
    policy:dict
    capacity:int=8
    def __post_init__(self):
        self.address.validate();validate_node(self.policy)
        if type(self.capacity) is not int or not 1<=self.capacity<=64:raise ValueError('invalid memory capacity')
    def evaluate(self,episodes):return replay(episodes,self.address,self.policy,self.capacity)
    def save(self,path):
        Path(path).write_text(json.dumps({'format':FORMAT,'address':self.address.json(),
                            'policy':self.policy,'capacity':self.capacity},sort_keys=True)+'\n')
    @classmethod
    def load(cls,path):
        o=json.loads(Path(path).read_text())
        if not isinstance(o,dict) or set(o)!={'format','address','policy','capacity'} or o['format']!=FORMAT:
            raise ValueError('unsupported model serialization')
        a=o['address']
        if not isinstance(a,dict) or set(a)!={'cu','cv'}:raise ValueError('bad address')
        return cls(Address(**a),o['policy'],o['capacity'])
    def cpp(self):
        def emit(n):
            if n['op']=='atom':
                p=ATOMS[n['index']]
                v={'a':'a','b':'b','sum':'(a+b)','difference':'(a-b)',
                   'product':'(a*b)','square_difference':'(a*a-b*b)'}[p.feature]
                return f'({v} '+('>' if p.comparison=='gt' else '<')+f' {p.threshold:.17g})'
            return f'({emit(n["left"])} '+('&&' if n['op']=='and' else '||')+f' {emit(n["right"])})'
        expression=emit(self.policy)
        addr=f'({self.address.cu}*u+{self.address.cv}*v)'
        return f'''// SPARK M20 learned address and reusable, composed write subprograms.
// No fixed object IDs, no oracle relevance bit, no neural weight tensors.
#include <cmath>
#include <cstddef>
struct M20Memory{{double slots[{self.capacity}]{{}};}};
static inline int m20_address(double u,double v) {{
 if(!std::isfinite(u)||!std::isfinite(v))return -1;
 const double w={addr};
 if(!std::isfinite(w)||w<-.5||w>={self.capacity}-.5)return -1;
 return static_cast<int>(std::floor(w+.5));
}}
static inline bool m20_keep(double a,double b) {{return {expression};}}
extern "C" void m20_reset(M20Memory* mem){{if(mem)for(int i=0;i<{self.capacity};++i)mem->slots[i]=0.;}}
extern "C" int m20_event(M20Memory* mem,double u,double v,double a,double b,double value,int query,double* out){{
 if(!mem||!out||!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(value))return 0;
 const int key=m20_address(u,v);
 if(!query&&key>=0&&m20_keep(a,b))mem->slots[key]=value;
 *out=(key>=0)?mem->slots[key]:0.;return 1;
}}
'''


def _identity(rng,key):
    # Neither key nor a relevance indicator is part of the returned observation.
    nuisance=rng.uniform(-2.4,2.4)
    return (float(key)+nuisance,nuisance)

def _context(rng,scale):
    upper=min(.20,scale) if rng.random()<.55 else scale
    return (rng.choice((-1.,1.))*rng.uniform(.04,upper),
            rng.choice((-1.,1.))*rng.uniform(.04,upper))

def generate(seed,count,*,keys=4,distractors=(4,15),queries=3,scale=1.,calibration=False):
    if type(keys) is not int or not 1<=keys<=64 or count<1 or queries<1 or distractors[0]<1 or distractors[0]>distractors[1]:
        raise ValueError('invalid synthetic task configuration')
    rng=random.Random(seed);result=[]
    for _ in range(count):
        truth=[0.]*keys;episode=[]
        def emit(key,a,b):
            value=rng.uniform(-1.5,1.5);u,v=_identity(rng,key)
            # This hidden rule is ONLY used to produce supervision targets.
            relevant=calibration or (a*b>0 and a+b>.25 and a-b<.25)
            if relevant:truth[key]=value
            episode.append(Observation(u,v,value,a,b))
        for k in range(keys):emit(k,1.,1.)
        for _ in range(queries):
            for _ in range(rng.randint(*distractors)):
                a,b=_context(rng,scale)
                emit(rng.randrange(keys),a,b)
            key=rng.randrange(keys);u,v=_identity(rng,key)
            episode.append(Observation(u,v,0.,0.,0.,True,truth[key]))
        result.append(episode)
    return result

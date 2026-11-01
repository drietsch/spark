"""M17: learn sparse write/retain decisions from delayed read losses.

A memory bank is a mathematical operation, not a token retrieval database.
The routing function is trained from query outcomes without write labels.
A fixed-capacity native implementation preserves the learned binary policy.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random


def sigmoid(x: float) -> float:
    if x >= 0: return 1. / (1. + math.exp(-x))
    e=math.exp(x)
    return e/(1.+e)


@dataclass(frozen=True)
class Event:
    signal: float
    key: int
    value: float
    query: bool = False
    target: float = 0.0


def episodes(seed: int, count: int, *, keys=4, distractors=(8,28),
             value_scale=1.0, queries=3):
    """Independent episodes. No write labels ever passed to the learner.

    The generator uses a concealed relevance rule: signal=1 updates a key;
    signal=0 is a distractor. Each query has a target but does not update state.
    """
    r=random.Random(seed)
    result=[]
    for _ in range(count):
        state=[0.0]*keys
        stream=[]
        # Ensure each episode exercises both relevant and irrelevant writes.
        for k in range(keys):
            value=r.uniform(-value_scale,value_scale)
            state[k]=value
            stream.append(Event(1.0,k,value))
        for q in range(queries):
            for _ in range(r.randint(*distractors)):
                key=r.randrange(keys)
                cue=r.random()<0.19
                value=r.uniform(-value_scale,value_scale)
                if cue:state[key]=value
                stream.append(Event(float(cue),key,value))
            key=r.randrange(keys)
            stream.append(Event(0.0,key,0.,True,state[key]))
        result.append(stream)
    return result


@dataclass
class MemoryRouter:
    """Sigmoid gate trained by differentiating through writes to future reads.

    Deployment chooses a hard write/retain decision from learned gate score.
    The 0.5 threshold is generic machinery; no cue-specific policy is coded.
    """
    theta: list[float]
    keys: int = 4

    def __post_init__(self):
        if len(self.theta)!=3 or not all(math.isfinite(v) for v in self.theta):
            raise ValueError('invalid policy coefficients')
        if not 1<=self.keys<=64:raise ValueError('invalid memory capacity')

    def score(self,e:Event):
        return sigmoid(self.theta[0]+self.theta[1]*e.signal+self.theta[2]*e.value)

    def evaluate_episode(self,seq,*,hard=True,derivative=False):
        memory=[0.]*self.keys
        jac=[[0.,0.,0.] for _ in range(self.keys)]
        loss=hits=reads=0
        grad=[0.,0.,0.]
        for e in seq:
            if not 0<=e.key<self.keys or not all(math.isfinite(v) for v in (e.signal,e.value,e.target)):
                raise ValueError('invalid memory event')
            k=e.key
            if e.query:
                error=memory[k]-e.target
                loss+=error*error
                hits+=abs(error)<1e-8
                reads+=1
                if derivative:
                    for j in range(3):grad[j]+=2.*error*jac[k][j]
            else:
                g=self.score(e)
                f=[1.,e.signal,e.value]
                old=memory[k]
                if hard:
                    if g>.5:memory[k]=e.value
                else:
                    memory[k]=(1.-g)*old+g*e.value
                    if derivative:
                        jac[k]=[(1-g)*jac[k][j]+g*(1-g)*(e.value-old)*f[j] for j in range(3)]
        return (loss/max(reads,1),hits,reads,grad)

    def fit(self,data,*,epochs=45,rate=.24,seed=77):
        """Adam on downstream read loss; train-only data; no teacher write labels."""
        r=random.Random(seed)
        first=[0.]*3;second=[0.]*3;step=0
        history=[]
        for epoch in range(epochs):
            indices=list(range(len(data)));r.shuffle(indices)
            total=0.
            for idx in indices:
                loss,_,_,gradient=self.evaluate_episode(data[idx],hard=False,derivative=True)
                total+=loss
                norm=max(1.,math.sqrt(sum(g*g for g in gradient))/2.)
                step+=1
                for j in range(3):
                    g=gradient[j]/norm
                    first[j]=.9*first[j]+.1*g
                    second[j]=.999*second[j]+.001*g*g
                    self.theta[j]-=rate*(first[j]/(1-.9**step))/(math.sqrt(second[j]/(1-.999**step))+1e-8)
                    self.theta[j]=max(-18.,min(18.,self.theta[j]))
            history.append(total/len(data))
        return history

    def cpp(self):
        """Native bounded memory, learned policy coefficients passed as live data."""
        return f'''// M17 learned read-supervised write policy, mutable parameter ABI.
#include <cmath>
#include <cstddef>
struct SparkLearnedMemory {{double slots[{self.keys}]{{}};double theta[3]{{}};}};
extern "C" void spark_memory_reset_v2(SparkLearnedMemory* m,const double* policy) {{
 for(int i=0;i<{self.keys};++i)m->slots[i]=0.;
 for(int i=0;i<3;++i)m->theta[i]=policy[i];
}}
extern "C" double spark_memory_signal_v2(SparkLearnedMemory* m,int key,double signal,double value,int query) {{
 if(!m||key<0||key>={self.keys})return 0.;
 if(query)return m->slots[key];
 const double s=m->theta[0]+m->theta[1]*signal+m->theta[2]*value;
 if(s>0.)m->slots[key]=value;
 return m->slots[key];
}}
'''

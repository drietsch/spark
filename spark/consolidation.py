"""M15: neural-behavior consolidation with independent acceptance and rollback.

Synthesizes rational sparse polynomial alternatives to an observed neural
function; candidates MUST beat a validation gate and preserve recurrent rollout
behavior before atomic replacement. Network fallback is retained on rejection.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
import math
import numpy as np


def features(z,degree=3):
    z=list(z); out=[((),1.)]
    for d in range(1,degree+1):
        for inds in combinations_with_replacement(range(len(z)),d):
            value=math.prod(z[i] for i in inds)
            out.append((inds,value))
    return out


def evaluate(terms,z):
    return sum((n/d)*math.prod(z[i] for i in inds) for inds,n,d in terms)


def synthesize(observations,*,degree=3,denominator=256,max_terms=12):
    if len(observations)<50 or not 1<=degree<=3 or len(observations[0][0])>4:
        raise ValueError('invalid synthesis observations')
    powers=[p for p,_ in features(observations[0][0],degree)]
    X=np.asarray([[v for _,v in features(z,degree)] for z,_ in observations]);
    y=np.asarray([v for _,v in observations]);res=y.copy();chosen=[];best=None
    for _ in range(max_terms):
        norms=np.maximum(np.linalg.norm(X,axis=0),1e-12)
        corr=np.abs(X.T@res)/norms
        corr[chosen]=-1
        idx=int(np.argmax(corr))
        if corr[idx]<1e-9:break
        chosen.append(idx)
        coeff=np.linalg.lstsq(X[:,chosen],y,rcond=None)[0]
        rational=[Fraction(float(v)).limit_denominator(denominator) for v in coeff]
        terms=[(powers[i],v.numerator,v.denominator) for i,v in zip(chosen,rational) if v]
        mse=float(np.mean((np.asarray([evaluate(terms,z) for z,_ in observations])-y)**2))
        res=y-X[:,chosen]@coeff
        if best is None or (mse,len(terms))<(best[0],len(best[1])):best=(mse,terms)
    if best is None:raise ValueError('could not synthesize expression')
    return best[1]


@dataclass
class ConsolidatedFunction:
    neural:object
    symbolic:list|None=None
    accepted:int=0
    rejected:int=0

    def __call__(self,z):return evaluate(self.symbolic,z) if self.symbolic is not None else self.neural(z)

    def propose(self,train,validation,rollouts,*,absolute_mse=1e-5,
                max_relative=1.05,rollout_tolerance=0.05,max_terms=12):
        """Independent validation and multi-step rollout must pass; otherwise rollback."""
        if not train or not validation or not rollouts:raise ValueError('separate evidence required')
        terms=synthesize(train,max_terms=max_terms)
        val_mse=sum((evaluate(terms,z)-target)**2 for z,target in validation)/len(validation)
        neural_mse=sum((self.neural(z)-target)**2 for z,target in validation)/len(validation)
        # Rollouts consist of (initial_state, inputs, target_final), where
        # s_(t+1) = 0.6*s_t + learned_function(s_t, input_t).
        max_error=0.
        for init,xs,expected in rollouts:
            s=float(init)
            for x in xs:s=.6*s+evaluate(terms,(s,x))
            max_error=max(max_error,abs(s-expected))
        passed=(val_mse<=absolute_mse and val_mse<=max_relative*neural_mse+1e-12
                and max_error<=rollout_tolerance)
        if passed:
            self.symbolic=terms
            self.accepted+=1
        else:self.rejected+=1
        return {'accepted':passed,'validation_mse':val_mse,'neural_validation_mse':neural_mse,
                'rollout_max_error':max_error,'terms':len(terms)}

    def rollback(self):self.symbolic=None

    def cpp(self):
        if self.symbolic is None:raise ValueError('no validated symbolic replacement')
        terms=[]
        for inds,num,den in self.symbolic:
            expr='*'.join(f'z[{i}]' for i in inds) if inds else '1.0'
            terms.append(f'(({num}.0/{den}.0)*({expr}))')
        return ('// Validated SPARK M15 mathematical-function consolidation.\n'
                '#include <cstddef>\n'
                'extern "C" double spark_consolidated(const double* z) {\n'
                ' return '+' + '.join(terms)+';\n}\n')


class ConsolidationManager:
    """Evidence-triggered promotion and rollback of symbolic neural replacements.

    Inputs are observed (function_argument, target) samples, with a reserved
    chronological validation window. Calls are automatic once enough evidence
    arrives; rollouts are held-out sequences supplied separately.
    """
    def __init__(self, neural, *, min_samples=300, interval=100,
                 absolute_mse=5e-4, drift_mse=0.02):
        if min_samples<80 or interval<1:raise ValueError('invalid consolidation budget')
        self.function=ConsolidatedFunction(neural)
        self.samples=[]
        self.min_samples=min_samples
        self.interval=interval
        self.absolute_mse=absolute_mse
        self.drift_mse=drift_mse
        self.decisions=[]
        self.errors=[]

    def observe(self,z,target,*,rollouts=()):
        z=tuple(map(float,z));target=float(target)
        if not math.isfinite(target) or not all(math.isfinite(v) for v in z):
            raise ValueError('nonfinite observation')
        predicted=self.function(z)
        error=(predicted-target)**2
        self.samples.append((z,target))
        self.errors.append(error)
        # Detect environment change from recent prediction failure and
        # restore the mutable neural function; do not train it here.
        if self.function.symbolic is not None and len(self.errors)>=20 and \
           sum(self.errors[-20:])/20>self.drift_mse:
            self.function.rollback()
            self.decisions.append({'event':'rollback','observations':len(self.samples)})
        count=len(self.samples)
        if count>=self.min_samples and count%self.interval==0 and rollouts:
            # Validation is chronologically newer than training; no random
            # mixing between candidate fitting and acceptance evidence.
            cut=int(count*.8)
            fitting=[(z,self.function.neural(z)) for z,_ in self.samples[:cut]]
            validation=self.samples[cut:]
            proposal=self.function.propose(fitting,validation,rollouts,
                 absolute_mse=self.absolute_mse,max_relative=1.25,
                 rollout_tolerance=.12,max_terms=9)
            self.decisions.append({'event':'accept' if proposal['accepted'] else 'reject',
                                   'observations':count,'metrics':proposal})
        return predicted

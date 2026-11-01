"""M14: observation-only predictive state learning and native compilation.

Learns an autoregressive controlled state from observation histories. No true
physical/hidden state labels are provided. Inference is scalar recurrence, with
optional online adaptation of its coefficients from prediction error.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np

MAX_ORDER = 6


def _window(sequence, order):
    y,u=sequence
    for i in range(order-1,len(y)-1):
        if not all(math.isfinite(v) for v in y[i-order+1:i+2]):
            continue
        yield [*reversed(y[i-order+1:i+1]),u[i],1.0],y[i+1]


def fit(sequences, *, max_order=4, ridge=1e-6):
    """Fit ARX predictors, choose memory order on TRAIN-only group holdout."""
    if not 1<=max_order<=MAX_ORDER or len(sequences)<8:raise ValueError('invalid training configuration')
    n=max(2,len(sequences)//5)
    train,tune=sequences[:-n],sequences[-n:]
    best=None
    for order in range(1,max_order+1):
        samples=[(x,v) for seq in train for x,v in _window(seq,order)]
        if len(samples)<10*(order+2):raise ValueError('insufficient observations')
        X=np.array([a for a,b in samples]);Y=np.array([b for a,b in samples])
        regularizer=np.diag([ridge]*(order+1)+[0.])
        coef=np.linalg.solve(X.T@X+regularizer,X.T@Y)
        val=[(a,b) for seq in tune for a,b in _window(seq,order)]
        loss=float(np.mean([(float(np.dot(coef,a))-b)**2 for a,b in val]))
        if best is None or (loss+1e-9*order,order)<(best[0],best[1]):best=(loss+1e-9*order,order)
    order=best[1]
    examples=[(a,b) for seq in sequences for a,b in _window(seq,order)]
    X=np.asarray([a for a,b in examples]);Y=np.asarray([b for a,b in examples])
    coef=np.linalg.solve(X.T@X+np.diag([ridge]*(order+1)+[0.]),X.T@Y)
    return PredictiveState(order,[float(x) for x in coef],float(best[0]-1e-9*order))


@dataclass
class PredictiveState:
    order:int
    coefficients:list[float]
    train_only_validation_mse:float=math.nan

    def validate(self):
        if not 1<=self.order<=MAX_ORDER or len(self.coefficients)!=self.order+2:
            raise ValueError('invalid learned state geometry')
        if not all(math.isfinite(v) and abs(v)<100 for v in self.coefficients):
            raise ValueError('nonfinite or unsafe learned coefficient')

    def predict(self,history,u):
        self.validate()
        if len(history)!=self.order:raise ValueError('wrong recurrent state order')
        return sum(a*b for a,b in zip(self.coefficients,[*history,u,1.]))

    def online(self,history,u,observed, *,lr=0.0):
        """Prequential: predict first, then optionally update from observation."""
        prediction=self.predict(history,u)
        if not math.isfinite(observed):return prediction
        if not 0<=lr<=0.1:raise ValueError('invalid learning rate')
        inputs=[*history,u,1.]
        if lr:
            err=prediction-observed
            normalizer=1+sum(v*v for v in inputs)
            self.coefficients=[max(-99.,min(99.,w-lr*err*x/normalizer))
                               for w,x in zip(self.coefficients,inputs)]
        return prediction

    def rollout(self,initial,inputs,observed=None,*,lr=0.):
        """Predict before observing; missing observations use model predictions."""
        state=list(initial)
        if len(state)!=self.order:raise ValueError('invalid initial observed history')
        ys=[]
        if observed is None:observed=[None]*len(inputs)
        if len(observed)!=len(inputs):raise ValueError('mismatched feedback')
        for u,y in zip(inputs,observed):
            pred=self.predict(state,u)
            ys.append(pred)
            if y is not None and math.isfinite(y):
                self.online(state,u,y,lr=lr)
                new=y
            else:new=pred
            state=[new]+state[:-1]
        return ys

    def save(self,path):
        self.validate()
        Path(path).write_text(json.dumps({'format':'spark-predictive-state-v1',
                'order':self.order,'coefficients':self.coefficients},sort_keys=True)+'\n')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if set(d)!={'format','order','coefficients'} or d['format']!='spark-predictive-state-v1':
            raise ValueError('invalid model serialization')
        obj=cls(d['order'],d['coefficients']);obj.validate();return obj

    def cpp(self):
        """Compiled recurrence, mutable coefficients, feedback updates; no matrices."""
        self.validate();n=self.order
        return f'''// SPARK M14 learned predictive program, mutable learned coefficients.
#include <cmath>
#include <cstddef>
struct SparkPredictive {{ double weights[{n+2}]; }};
extern "C" void spark_predictive_init(SparkPredictive* m,const double* w) {{
 for(int i=0;i<{n+2};++i)m->weights[i]=w[i];
}}
extern "C" double spark_predictive_step(SparkPredictive* m,const double* history,double u,
 double observation,double lr,int has_observation) {{
 double v[{n+2}];
 for(int i=0;i<{n};++i)v[i]=history[i];
 v[{n}]=u;v[{n+1}]=1.;
 double p=0.,norm=1.;
 for(int i=0;i<{n+2};++i){{p+=m->weights[i]*v[i];norm+=v[i]*v[i];}}
 if(has_observation && lr>0.) {{
  double err=p-observation;
  for(int i=0;i<{n+2};++i){{
   double t=m->weights[i]-lr*err*v[i]/norm;
   m->weights[i]=std::fmax(-99.,std::fmin(99.,t));
  }}
 }}
 return p;
}}
'''

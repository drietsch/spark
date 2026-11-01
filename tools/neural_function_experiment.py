#!/usr/bin/env python3
"""M12: learn adaptive gated-dense function, compose with symbolic recurrence, compile."""
from __future__ import annotations
import argparse
import ctypes
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from spark.neural_primitive import (fit_neural,build_hybrid,save,cpp_source,
                                    load,neural_scalar)


def rule(s,x,context):
    d=s[0]-s[1]
    q=.44*math.tanh(1.25*d+.4*x)+.20*math.tanh(1.8*context*x)
    return (float(x),s[0],.6*s[2]+q,.5*s[3]+.2*s[2]+q),q


def samples(seed,n,*,ood=False):
    rng=random.Random(seed)
    contexts=[-.8,.15,.55,1.65] if ood else [-1.25,-.25,.75,1.25]
    scale=1.48 if ood else 1.1
    xs=[]
    for _ in range(n):
        s=tuple(rng.uniform(-scale,scale) for _ in range(4))
        x=rng.uniform(-1.45,1.45) if ood else rng.uniform(-1.1,1.1)
        c=float(rng.choice(contexts))
        target,q=rule(s,x,c)
        xs.append((s,x,c,target,q))
    return xs


def poly_features(z):
    a,b,c=z
    return [1.,a,b,c,a*a,b*b,c*c,a*b,a*c,b*c]


def polynomial_baseline(train):
    z=[(s[0]-s[1],x,c) for s,x,c,_,_ in train]
    A=np.asarray([poly_features(r) for r in z],dtype=float)
    y=np.asarray([q for *_,q in train])
    coef=np.linalg.lstsq(A,y,rcond=None)[0]
    return lambda s,x,c:float(np.dot(coef,poly_features((s[0]-s[1],x,c))))


def affine_baseline(train):
    A=np.asarray([(*s,x,c,1.) for s,x,c,_,_ in train])
    Y=np.asarray([y for _,_,_,y,_ in train])
    coefs=np.linalg.lstsq(A,Y,rcond=None)[0]
    return lambda s,x,c:tuple(np.asarray([*s,x,c,1.])@coefs)



def full_dense_baseline(train, *,width=16,epochs=1100,lr=.012):
    """Independent tiny dense tanh transition net; no symbolic recurrence."""
    Z=np.asarray([[*s,x,c] for s,x,c,_,_ in train],dtype=np.float64)
    Y=np.asarray([y for _,_,_,y,_ in train],dtype=np.float64)
    rng=np.random.default_rng(900)
    p=dict(w=rng.normal(0,.32,(6,width)),b=np.zeros(width),
           v=rng.normal(0,.2,(width,4)),d=np.zeros(4))
    first={k:np.zeros_like(v) for k,v in p.items()}
    second={k:np.zeros_like(v) for k,v in p.items()}
    for step in range(1,epochs+1):
        h=np.tanh(Z@p['w']+p['b'])
        pred=h@p['v']+p['d']
        delta=(pred-Y)*(2.0/(len(Z)*4))
        dh=(delta@p['v'].T)*(1.-h*h)
        grad={'w':Z.T@dh,'b':dh.sum(axis=0),
              'v':h.T@delta,'d':delta.sum(axis=0)}
        for name in p:
            first[name]=.9*first[name]+.1*grad[name]
            second[name]=.999*second[name]+.001*grad[name]*grad[name]
            p[name]-=lr*(first[name]/(1-.9**step))/(np.sqrt(second[name]/(1-.999**step))+1e-8)
    def predict(s,x,c):
        v=np.asarray([*s,x,c])
        return tuple(np.tanh(v@p['w']+p['b'])@p['v']+p['d'])
    return predict,6*width+width+width*4+4


def baseline_step(q_fn,s,x,c):
    q=q_fn(s,x,c)
    return (float(x),s[0],.6*s[2]+q,.5*s[3]+.2*s[2]+q)


def score(rows,predict):
    sq=[];absolute=[]
    for s,x,c,target,_ in rows:
        actual=predict(s,x,c)
        for a,b in zip(actual,target):
            sq.append((a-b)**2)
            absolute.append(abs(a-b))
    return {'examples':len(rows),'mse':sum(sq)/len(sq),
            'max_absolute_error':max(absolute)}


def native_compile(model,out):
    cpp=out.with_suffix('.cpp');cpp.write_text(cpp_source(model.payload))
    compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise RuntimeError('requires a C++17 compiler')
    subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(out)],check=True)
    lib=ctypes.CDLL(str(out.resolve()))
    ptr=ctypes.POINTER(ctypes.c_double)
    lib.spark_hybrid_step_v1.argtypes=(ptr,ctypes.c_double,ctypes.c_double,ptr)
    lib.spark_hybrid_step_v1.restype=None
    lib.spark_hybrid_rollout_v1.argtypes=(ptr,ptr,ptr,ctypes.c_size_t)
    lib.spark_hybrid_rollout_v1.restype=None
    return lib


def native_step(lib,s,x,c):
    inp=(ctypes.c_double*4)(*s);out=(ctypes.c_double*4)()
    lib.spark_hybrid_step_v1(inp,x,c,out)
    return tuple(out)


def native_rollout(lib,s,xs,contexts):
    s=(ctypes.c_double*4)(*s)
    xs=(ctypes.c_double*len(xs))(*xs)
    cs=(ctypes.c_double*len(contexts))(*contexts)
    lib.spark_hybrid_rollout_v1(s,xs,cs,len(xs))
    return tuple(s)


def experiment(out,*,epochs=900):
    out.mkdir(parents=True,exist_ok=True)
    train=samples(121,3000)
    holdout=samples(918,600,ood=True)
    z=np.asarray([(s[0]-s[1],x,c) for s,x,c,_,_ in train],dtype=np.float64)
    qs=np.asarray([q for _,_,_,_,q in train],dtype=np.float64)
    model=build_hybrid(fit_neural(z,qs,width=12,epochs=epochs,seed=15,lr=.015))
    save(out/'hybrid.program.json',model)
    assert load(out/'hybrid.program.json').payload==model.payload
    lib=native_compile(model,out/'hybrid.so')
    poly=polynomial_baseline(train)
    affine=affine_baseline(train)
    dense,dense_parameters=full_dense_baseline(train)
    neural_result=score(holdout,model.step)
    native_result=score(holdout,lambda s,x,c:native_step(lib,s,x,c))
    polynomial_result=score(holdout,lambda s,x,c:baseline_step(poly,s,x,c))
    affine_result=score(holdout,affine)
    dense_result=score(holdout,dense)
    max_parity=max(abs(a-b) for s,x,c,_,_ in holdout
                   for a,b in zip(model.step(s,x,c),native_step(lib,s,x,c)))
    # Stability and native parity for long, never-trained recurrent sequences.
    rng=random.Random(951)
    lengths=(32,128,1024,4096)
    rollout_count=0
    native_parity=0
    true_mse=[]
    max_rollout_difference=0.
    for n in lengths:
        for _ in range(8):
            s0=tuple(rng.uniform(-1.,1.) for _ in range(4))
            xs=[rng.uniform(-1.2,1.2) for _ in range(n)]
            cs=[rng.choice([-.8,.15,.55,1.65]) for _ in range(n)]
            ground=s0
            for x,c in zip(xs,cs):ground=rule(ground,x,c)[0]
            python_pred=model.rollout(s0,xs,cs)
            native_pred=native_rollout(lib,s0,xs,cs)
            delta=max(abs(a-b) for a,b in zip(python_pred,native_pred))
            max_rollout_difference=max(max_rollout_difference,delta)
            native_parity+=delta<1e-9
            true_mse.extend((a-b)**2 for a,b in zip(ground,native_pred))
            rollout_count+=1
    # Native benchmark excludes Python FFI overhead by doing 4096 steps per call.
    n=4096
    xs=[rng.uniform(-1.,1.) for _ in range(n)]
    cs=[rng.choice([-1.,0.,1.]) for _ in range(n)]
    for _ in range(5):native_rollout(lib,(0.,0.,0.,0.),xs,cs)
    t=perf_counter()
    for _ in range(60):native_rollout(lib,(0.,0.,0.,0.),xs,cs)
    duration=perf_counter()-t
    report={'format':'spark-m12-neural-function-v1',
        'training_examples':len(train),'held_out_examples':len(holdout),
        'held_out_contexts':[-.8,.15,.55,1.65],
        'training_contexts':[-1.25,-.25,.75,1.25],
        'neural_width':12,'neural_parameter_count':model.parameter_count,
        'neural_and_symbolic':neural_result,'neural_and_symbolic_native':native_result,
        'symbolic_quadratic_baseline':polynomial_result,
        'affine_baseline':affine_result,
        'dense_only_baseline':dense_result,'dense_only_parameter_count':dense_parameters,
        'native_max_step_disagreement':max_parity,
        'native_long_rollouts_matching_python':native_parity,
        'long_rollout_count':rollout_count,
        'longest_rollout':max(lengths),
        'native_max_rollout_disagreement':max_rollout_difference,
        'native_long_rollout_ground_truth_mse':sum(true_mse)/len(true_mse),
        'native_steps_per_second':round(n*60/duration,1),
        'native_library_bytes':(out/'hybrid.so').stat().st_size,
        'constraints':[
            'The four-register symbolic scaffold and neural inputs are prescribed, not evolved.',
            'The synthetic target neural subfunction is supervised directly during training.',
            'Dense weights are learned and embedded as numerical constants in generated C++.',
            'No Transformer baseline or natural-language reasoning claim.',
            'The quadratic baseline shares the same symbolic scaffold.',
            'Native throughput is for four-register recurrent steps, not language tokens.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('neural_parameter_count','neural_and_symbolic','symbolic_quadratic_baseline','dense_only_baseline','affine_baseline','native_max_step_disagreement','native_long_rollouts_matching_python','native_steps_per_second')},indent=2))
    assert max_parity<1e-10,report
    assert native_parity==rollout_count,report
    return report


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,default=Path('artifacts/m12'))
    a.add_argument('--epochs',type=int,default=900)
    args=a.parse_args()
    experiment(args.output,epochs=args.epochs)

#!/usr/bin/env python3
"""M10: learn coupled equations, compile C++17 and test OOD transitions/rollouts."""
from __future__ import annotations
import argparse
import ctypes
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.vector_program import cpp_source,save,synthesize
import numpy as np


def hidden_rule(s,x,reset):
    a,b,c,d=s
    if reset and x==-2:
        return (0.,0.,0.,0.)
    q=(a-b)*x
    return (float(x),a,c+q,d+c+q)


def observations(seed,count,reset,wide=False):
    rng=random.Random(seed)
    examples=[]
    for _ in range(count):
        s=tuple(rng.uniform(-3,3) if wide else rng.uniform(-1,1) for _ in range(4))
        x=float(rng.choice([-5,-4,-3,-2,-1,0,1,2,3,4,5] if wide else
                           ([-2,-1,0,1,2,3] if reset else [-1,0,1,2])))
        examples.append((s,x,hidden_rule(s,x,reset)))
    return examples


def affine_baseline(train,validation):
    X=np.asarray([[*s,x,1.0] for s,x,_ in train]);Y=np.asarray([y for _,_,y in train])
    coefs=np.linalg.lstsq(X,Y,rcond=None)[0]
    return lambda s,x:tuple(np.asarray([*s,x,1.0])@coefs)


def score(rows,predict):
    differences=[abs(a-b) for s,x,y in rows for a,b in zip(predict(s,x),y)]
    return {'examples':len(rows),'mse':sum(v*v for v in differences)/len(differences),
            'max_abs_error':max(differences),'exact_fraction':sum(v<1e-8 for v in differences)/len(differences)}


def native_compile(model,out):
    path=out.with_suffix('.cpp');path.write_text(cpp_source(model.payload))
    compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise RuntimeError('requires C++17 compiler')
    subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC',str(path),'-o',str(out)],check=True)
    lib=ctypes.CDLL(str(out.resolve()))
    ptr=ctypes.POINTER(ctypes.c_double)
    lib.spark_vector_step_v1.argtypes=(ptr,ctypes.c_double,ptr)
    lib.spark_vector_step_v1.restype=None
    lib.spark_vector_rollout_v1.argtypes=(ptr,ptr,ctypes.c_size_t)
    lib.spark_vector_rollout_v1.restype=None
    return lib


def native_step(lib,state,x):
    s=(ctypes.c_double*len(state))(*state)
    out=(ctypes.c_double*len(state))()
    lib.spark_vector_step_v1(s,x,out)
    return tuple(out)


def native_rollout(lib,state,xs):
    s=(ctypes.c_double*len(state))(*state)
    x=(ctypes.c_double*len(xs))(*xs)
    lib.spark_vector_rollout_v1(s,x,len(xs))
    return tuple(s)


def experiment(reset,out):
    tag='reset_coupled' if reset else 'coupled'
    train=observations(242,950,reset)
    holdout=observations(330,350,reset,wide=True)
    model=synthesize(train,degree=2)
    assert model.train_mse<1e-15,(tag,model.train_mse)
    assert model.payload['guard']==(-2.0 if reset else None),model.payload['guard']
    save(out/(tag+'.program.json'),model)
    lib=native_compile(model,out/(tag+'.so'))
    native=score(holdout,lambda s,x:native_step(lib,s,x))
    student=score(holdout,model.step)
    baseline=score(holdout,affine_baseline(train,holdout))
    native_delta=max(abs(a-b) for s,x,_ in holdout
                     for a,b in zip(model.step(s,x),native_step(lib,s,x)))
    rng=random.Random(102)
    lengths=(32,128,1024,4096)
    exact_python=exact_native=0
    worst_native=worst_python=0.
    total=0
    for length in lengths:
        for _ in range(12):
            xs=[float(rng.choice([-2,-1,0,1,2] if reset else [-1,0,1,2]))
                for _ in range(length)]
            init=tuple(float(rng.randrange(-2,3)) for _ in range(4))
            expected=init
            for x in xs:expected=hidden_rule(expected,x,reset)
            student_roll=model.rollout(init,xs)
            native_roll=native_rollout(lib,init,xs)
            ep=max(abs(a-b) for a,b in zip(expected,student_roll))
            en=max(abs(a-b) for a,b in zip(expected,native_roll))
            worst_python=max(worst_python,ep);worst_native=max(worst_native,en)
            exact_python+=ep<1e-8;exact_native+=en<1e-8
            total+=1
    n=4096;inputs=[float(rng.randrange(-1,2)) for _ in range(n)]
    for _ in range(5):native_rollout(lib,(0.,0.,0.,0.),inputs)
    begin=perf_counter()
    for _ in range(150):native_rollout(lib,(0.,0.,0.,0.),inputs)
    elapsed=perf_counter()-begin
    used={tuple(p) for kind in ('then','else') for terms in model.payload[kind] for p,_,_ in terms}
    appearances=sum(len(terms) for terms in model.payload['else'])
    summary={'training_transitions':len(train),'held_out_transitions':len(holdout),
             'learned_guard':model.payload['guard'],'training_mse':model.train_mse,
             'train_only_tuning_mse':model.tune_mse,'ood':student,'native_ood':native,
             'affine_ood':baseline,'native_max_difference':native_delta,
             'native_rollouts_exact':exact_native,'python_rollouts_exact':exact_python,
             'rollout_sequences':total,'longest_rollout_steps':max(lengths),
             'max_native_rollout_error':worst_native,'max_python_rollout_error':worst_python,
             'distinct_shared_basis_expressions':len(used),
             'output_term_references':appearances,
             'native_library_bytes':(out/(tag+'.so')).stat().st_size,
             'native_rollout_steps_per_second':round(150*n/elapsed,2)}
    assert student['exact_fraction']==1,summary
    assert native['exact_fraction']==1,summary
    assert exact_native==exact_python==48,summary
    assert baseline['mse']>student['mse']+1e-4,summary
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('artifacts/m10'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result={'format':'spark-m10-vector-symbolic-v1',
            'model':'Learned rational multiregister programs with optional guard; shared expressions compiled once',
            'limitations':['Synthetic deterministic rules in a constrained polynomial/guard grammar.',
                           'Supervised transition training with fully observable state.',
                           'Affine baseline is not a Transformer or a modern state-space model.',
                           'This does not demonstrate general language modeling or Transformer superiority.']}
    for reset in (False,True):
        result['reset_coupled' if reset else 'coupled']=experiment(reset,args.output)
    (args.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({tag:{'ood':result[tag]['ood'],'baseline':result[tag]['affine_ood'],
        'guard':result[tag]['learned_guard'],'native_rollouts_exact':result[tag]['native_rollouts_exact']}
        for tag in ('coupled','reset_coupled')},indent=2))

if __name__=='__main__':main()

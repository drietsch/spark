#!/usr/bin/env python3
"""M11: independent multi-gain teacher-free task, structural search and native rollout."""
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
from spark.evolved_program import evolve, cpp_source, save


def transition(s,x,w,reset=False):
    if reset and x==-2:
        return (0.,0.,0.,0.)
    a,b,c,d=s
    q=w*x*(a-b)
    return (float(x),a,c+q,d+c+q)


def observations(seed,count, gains,*,reset=False,wide=False):
    rng=random.Random(seed)
    xs=[-2,-1,0,1,2,3] if reset else [-1,0,1,2]
    if wide:
        xs=([-5,-4,-3,-2,-1,0,1,2,3,4,5] if reset else [-5,-3,-1,0,1,3,5])
    rows=[]
    for i in range(count):
        state=tuple(rng.uniform(-1,1) if not wide else rng.uniform(-2.0,2.0) for _ in range(4))
        x=float(xs[i % len(xs)])
        gain=float(gains[(i//len(xs)) % len(gains)])
        rows.append((state,x,gain,transition(state,x,gain,reset)))
    rng.shuffle(rows)
    return rows


def measure(rows,predict):
    diffs=[abs(a-b) for s,x,w,expected in rows for a,b in zip(expected,predict(s,x,w))]
    return {'examples':len(rows),'mse':sum(v*v for v in diffs)/len(diffs),
            'max_abs_error':max(diffs),'exact_fraction':sum(v<1e-8 for v in diffs)/len(diffs)}


def affine_baseline(train):
    import numpy as np  # training-only; native/test imports need no NumPy
    X=np.asarray([[*s,x,gain,1.] for s,x,gain,_ in train])
    Y=np.asarray([y for _,_,_,y in train]); coefs=np.linalg.lstsq(X,Y,rcond=None)[0]
    return lambda s,x,w:tuple(np.asarray([*s,x,w,1.])@coefs)


def build_native(model,out):
    compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise RuntimeError('C++17 compiler required')
    out.with_suffix('.cpp').write_text(cpp_source(model.payload))
    subprocess.run([compiler,'-std=c++17','-O3','-shared','-fPIC',
                    str(out.with_suffix('.cpp')),'-o',str(out)],check=True)
    library=ctypes.CDLL(str(out.resolve()))
    ptr=ctypes.POINTER(ctypes.c_double)
    library.spark_evolved_step_v1.argtypes=(ptr,ctypes.c_double,ctypes.c_double,ptr)
    library.spark_evolved_step_v1.restype=None
    library.spark_evolved_rollout_v1.argtypes=(ptr,ptr,ctypes.c_size_t,ctypes.c_double)
    library.spark_evolved_rollout_v1.restype=None
    return library


def native_step(library,state,x,w):
    src=(ctypes.c_double*len(state))(*state)
    dst=(ctypes.c_double*len(state))()
    library.spark_evolved_step_v1(src,x,w,dst)
    return tuple(dst)


def native_rollout(library,state,xs,w):
    src=(ctypes.c_double*len(state))(*state)
    ins=(ctypes.c_double*len(xs))(*xs)
    library.spark_evolved_rollout_v1(src,ins,len(xs),w)
    return tuple(src)


def run(task,out):
    reset=(task=='reset')
    training_gains=(0.5,1.0,1.5)
    novel_gains=(-0.75,0.25,2.25)
    train=observations(1205,1050,training_gains,reset=reset)
    holdout=observations(433,390,novel_gains,reset=reset,wide=True)
    model=evolve(train,seed=12,max_generations=3,max_terms=8,complexity_weight=1e-9)
    save(out/(task+'.program.json'),model)
    lib=build_native(model,out/(task+'.so'))
    python=measure(holdout,model.step)
    native=measure(holdout,lambda s,x,w:native_step(lib,s,x,w))
    affine=measure(holdout,affine_baseline(train))
    rng=random.Random(543)
    rollout_checks=0
    worst=0.0
    # Some nonlinear recurrences can exponentially diverge in long rollouts;
    # use stable input zero / occasional small pulses to test propagation and
    # long state trajectories, not unstable chaotic amplification.
    for length in (32,128,1024,4096):
        for _ in range(8):
            gain=float(rng.choice(novel_gains))
            state=(rng.uniform(-.2,.2),rng.uniform(-.2,.2),rng.uniform(-.2,.2),rng.uniform(-.2,.2))
            xs=[float(-2 if reset and i%97==0 else 0.0) for i in range(length)]
            expected=state
            for x in xs:expected=transition(expected,x,gain,reset)
            a=model.rollout(state,xs,gain)
            b=native_rollout(lib,state,xs,gain)
            diff=max(abs(u-v) for u,v in zip(expected,b))
            worst=max(worst,diff)
            if max(abs(u-v) for u,v in zip(expected,a))<1e-8 and diff<1e-8:
                rollout_checks+=1
    # Check a far longer unrolled native sequence under controlled stable inputs.
    xs=[.02,-.01,0.0,.015]*1024
    for _ in range(3):native_rollout(lib,(0,0,0,0),xs,0.25)
    begin=perf_counter()
    for _ in range(120):native_rollout(lib,(0,0,0,0),xs,0.25)
    speed=4096*120/(perf_counter()-begin)
    result={'training':len(train),'validation':len(holdout),
            'training_gains':training_gains,'unseen_gains':novel_gains,
            'topology_generations':model.payload['generations'],
            'mutation_population_per_generation':model.topology_mutations,
            'learned_guard':model.payload['guard'],
            'compiled_program_complexity':model.complexity,
            'active_shared_expressions':len({tuple(p) for row in model.payload['outputs'] for p,_,_ in row}),
            'train_mse':model.train_mse,'internal_holdout_mse':model.inner_validation_mse,
            'held_out':python,'native_held_out':native,'affine_held_out':affine,
            'native_long_rollouts_exact':rollout_checks,'long_rollout_cases':32,
            'longest_rollout':4096,'native_worst_rollout_error':worst,
            'native_steps_per_s':round(speed,1),
            'native_library_bytes':(out/(task+'.so')).stat().st_size}
    assert python['exact_fraction']==native['exact_fraction']==1.0,result
    assert python['mse']<1e-18,result
    assert rollout_checks==32,result
    assert affine['mse']>python['mse']+1e-5,result
    assert model.payload['guard']==(-2.0 if reset else None),result
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('artifacts/symbolic-m11'))
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    report={'format':'spark-m11-evolutionary-programs-v1',
            'limitations':['Synthetic fully observable recurrence family inside the mutation grammar.',
                           'Input gain is a single externally provided shared scalar, not a trained parameter matrix.',
                           'Structural mutation search uses train-only cross-validation and numerical coefficient fitting.',
                           'No Transformer/SSM benchmark or claim of general intelligence.',
                           'Long-sequence tests use stable zero-input traces, not chaotic rollouts.']}
    for name in ('coupled','reset'):
        report[name]=run(name,args.output)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{k:report[name][k] for k in ('topology_generations','learned_guard','held_out','native_held_out','affine_held_out','native_long_rollouts_exact')} for name in ('coupled','reset')},indent=2))


if __name__=='__main__':main()

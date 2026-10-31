#!/usr/bin/env python3
"""M8: teach SPARK to DISCOVER and COMPILE state transition equations.

Synthetic task is intentionally interpretable: no claim of language-model or
Transformer superiority. Independent in-range, extrapolation and rollout tests.
"""
import argparse
import ctypes
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spark.symbolic_program import (affine_baseline, cpp_source, evaluate,
                                    serialize, synthesize)


def hidden_physics(state, inp):
    # Ground truth is used exclusively by the DATA GENERATOR, never the solver.
    return state + 0.5 * inp * (1.0 - state * state)


def make_examples(count, seed, s_range, x_range):
    rng = random.Random(seed)
    return [(s, x, hidden_physics(s,x))
            for s,x in ((rng.uniform(*s_range),rng.uniform(*x_range)) for _ in range(count))]


def mse(examples, predict):
    return sum((predict(s,x)-y)**2 for s,x,y in examples) / len(examples)


def cpp_compile(source, output: Path):
    compiler = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        raise RuntimeError('C++17 compiler not installed')
    output.parent.mkdir(parents=True,exist_ok=True)
    source_file = output.with_suffix('.cpp')
    source_file.write_text(source)
    subprocess.run([compiler,'-std=c++17','-O3','-fPIC','-shared',str(source_file),'-o',str(output)], check=True)
    lib = ctypes.CDLL(str(output.resolve()))
    lib.spark_expr_step_v1.argtypes = [ctypes.c_double,ctypes.c_double]
    lib.spark_expr_step_v1.restype = ctypes.c_double
    lib.spark_expr_rollout_v1.argtypes = [ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]
    lib.spark_expr_rollout_v1.restype = ctypes.c_double
    return lib


def physics_experiment(output: Path):
    train = make_examples(600, 2468, (-0.8,0.8),(-0.5,0.5))
    test = make_examples(300, 9001, (-0.8,0.8),(-0.5,0.5))
    extrap = make_examples(300, 9011, (-1.25,1.25),(-0.95,0.95))
    learned = synthesize(train,max_degree=3)
    a,b,c=affine_baseline(train)
    affine=lambda s,x:a+b*s+c*x
    model_file=output/'physics.expr.json'
    serialize(model_file,learned.expression,{'task':'nonlinear-state-update','train':len(train),
                                              'scope':'synthetic interpretable benchmark'})
    lib=cpp_compile(cpp_source(learned.expression),output/'physics.so')
    assert max(abs(lib.spark_expr_step_v1(s,x)-learned.step(s,x)) for s,x,_ in extrap)<1e-11

    rng = random.Random(1883)
    trajectories=[]
    for length in (32,128,512,1024):
        for _ in range(10):
            init=rng.uniform(-0.5,0.5)
            steps=[rng.uniform(-0.18,0.18) for _ in range(length)]
            s=init
            for x in steps: s=hidden_physics(s,x)
            model_s=init
            for x in steps: model_s=learned.step(model_s,x)
            array=(ctypes.c_double*length)(*steps)
            native_s=lib.spark_expr_rollout_v1(init,array,length)
            trajectories.append({'length':length,'student_error':abs(s-model_s),
                                 'native_error':abs(s-native_s)})

    rng=random.Random(66)
    xs=(ctypes.c_double*10000)(*(rng.uniform(-0.1,0.1) for _ in range(10000)))
    iterations=300
    start=perf_counter()
    acc=0.0
    for i in range(iterations):acc+=lib.spark_expr_rollout_v1(0.1,xs,10000)
    runtime=perf_counter()-start
    assert math.isfinite(acc)
    return {'task':'nonlinear-state-transition','train':600,'iid_test':300,
            'extrapolation_test':300,'polynomial_search_max_degree':3,
            'learned_degree':learned.degree,
            'expression':learned.expression, 'terms':[(i,j,str(v)) for i,j,v in learned.terms],
            'estimated_description_cost':learned.description_cost,
            'train_mse':learned.train_mse,
            'iid_mse':mse(test,learned.step),'affine_iid_mse':mse(test,affine),
            'extrapolation_mse':mse(extrap,learned.step),
            'affine_extrapolation_mse':mse(extrap,affine),
            'max_long_rollout_error':max(d['student_error'] for d in trajectories),
            'max_native_rollout_error':max(d['native_error'] for d in trajectories),
            'longest_rollout_steps':1024,'native_library_bytes':(output/'physics.so').stat().st_size,
            'native_rollout_calls':iterations*10000,
            'native_rollout_seconds':runtime,
            'native_rollout_steps_per_s':iterations*10000/runtime,
            'native_python_max_abs_difference':max(abs(lib.spark_expr_step_v1(s,x)-learned.step(s,x)) for s,x,_ in extrap)}


def parity_experiment(output: Path):
    rng=random.Random(44)
    train=[]
    for i in range(200):
        s=rng.randint(0,1); x=rng.randint(0,1)
        train.append((s,x, s^x))
    learned=synthesize(train,max_degree=2,square_free=True)
    serialize(output/'parity.expr.json',learned.expression,{'task':'unbounded-parity-state-machine'})
    lib=cpp_compile(cpp_source(learned.expression),output/'parity.so')
    checks=[]
    for length in (16,128,1024,4096):
        for _ in range(8):
            bits=[rng.randint(0,1) for _ in range(length)]
            expected=sum(bits)%2
            s=0.0
            for x in bits:s=learned.step(s,x)
            native=lib.spark_expr_rollout_v1(0.0,(ctypes.c_double*length)(*bits),length)
            checks.append((length,abs(s-expected),abs(native-expected)))
    return {'task':'parity-state-machine','train':len(train),'degree':learned.degree,
            'expression':learned.expression,'terms':[(i,j,str(v)) for i,j,v in learned.terms],
            'rollout_sequences':len(checks),'longest_rollout_steps':4096,
            'python_exact_sequences':sum(v==0 for _,v,_ in checks),
            'native_exact_sequences':sum(v==0 for _,_,v in checks),
            'native_library_bytes':(output/'parity.so').stat().st_size}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('artifacts/symbolic-m8'))
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report={'format':'spark-m8-symbolic-discovery-v1',
            'source':'synthetic known-law tasks (NOT natural-language benchmarking)',
            'model':'learned typed expression tree, rational constants, recurrent state',
            'physics':physics_experiment(args.output),
            'parity':parity_experiment(args.output),
            'limitations':['Polynomial grammar is specified by experimenter and is not universal program synthesis.',
                           'Program coefficients are learned numerical constants; no conventional neural weight matrices.',
                           'No evidence of superiority over a trained Transformer or real language-model quality.',
                           'Timing is for a simple native loop on one CPU, not a transformer comparison.']}
    p=args.output/'report.json'; p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__':main()

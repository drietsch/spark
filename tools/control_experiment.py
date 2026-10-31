#!/usr/bin/env python3
"""M9: discover rules, compile conditional programs and assess OOD rollouts."""
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
from spark.symbolic_control import discover, compiled_source, save
from spark.symbolic_program import affine_baseline, synthesize

def reset_counter(s,x):
    return 0.0 if x==-2 else (s if x==-1 else s+x)

def clamp_counter(s,x):
    return float(max(-3,min(3,s+x)))

def nonlinear_reset(s,x):
    return 0.0 if x==-2 else s+0.25*x*(1-s*s)

def compile_native(tree,out):
    compiler=shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:raise RuntimeError("C++17 compiler required")
    out.with_suffix(".cpp").write_text(compiled_source(tree))
    subprocess.run([compiler,"-std=c++17","-O3","-shared","-fPIC",
        str(out.with_suffix(".cpp")),"-o",str(out)],check=True)
    lib=ctypes.CDLL(str(out.resolve()))
    lib.spark_control_step_v1.argtypes=(ctypes.c_double,ctypes.c_double)
    lib.spark_control_step_v1.restype=ctypes.c_double
    lib.spark_control_rollout_v1.argtypes=(ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t)
    lib.spark_control_rollout_v1.restype=ctypes.c_double
    return lib

def experiment(name,rule,train,ood,output,seed):
    learned=discover(train,max_degree=3 if name=="nonlinear" else 1,
                     max_depth=1 if name=="nonlinear" else 2,min_branch=24)
    a,b,c=affine_baseline(train)
    affine=lambda s,x:a+b*s+c*x
    cubic=synthesize(train,max_degree=3)
    def stats(rows,predict):
        errors=[abs(predict(s,x)-y) for s,x,y in rows]
        return {"mse":sum(v*v for v in errors)/len(rows),
                "exact_fraction":sum(v<1e-8 for v in errors)/len(rows)}
    assert learned.train_mse < 1e-18, (name,learned)
    save(output/(name+".program.json"),learned.tree,
         {"task":name,"train_examples":len(train)})
    lib=compile_native(learned.tree,output/(name+".so"))
    native_delta=max(abs(lib.spark_control_step_v1(s,x)-learned.step(s,x)) for s,x,_ in ood)
    assert native_delta<1e-10
    rng=random.Random(seed)
    rollouts=[]
    for length in (32,128,1024,4096):
        for _ in range(12):
            init=float(rng.uniform(-.8,.8) if name=="nonlinear" else rng.randrange(-3,4))
            xs=[rng.choice([-2,-1,0,1]) if name=="nonlinear"
                else rng.choice([-4,-3,-2,-1,0,1,2]) if name=="reset"
                else rng.randrange(-2,3) for i in range(length)]
            expected=init
            for x in xs:expected=rule(expected,x)
            predicted=init
            for x in xs:predicted=learned.step(predicted,x)
            native=lib.spark_control_rollout_v1(init,(ctypes.c_double*length)(*xs),length)
            rollouts.append((length,abs(expected-predicted),abs(expected-native)))
    n=4096
    xs=(ctypes.c_double*n)(*([1.0,0.0,-1.0,2.0]*(n//4)))
    for _ in range(4):lib.spark_control_rollout_v1(0.0,xs,n)
    iterations=250
    start=perf_counter()
    for _ in range(iterations):lib.spark_control_rollout_v1(0.0,xs,n)
    secs=perf_counter()-start
    return {"train":len(train),"ood_test":len(ood),
      "branches":learned.branches,"complexity":learned.complexity,
      "learned_tree":learned.tree,"training_mse":learned.train_mse,
      "ood":stats(ood,learned.step),"affine_ood":stats(ood,affine),
      "unbranched_degree3_ood":stats(ood,cubic.step),
      "native_max_step_difference":native_delta,"rollouts":len(rollouts),
      "longest_rollout":4096,
      "python_exact_rollouts":sum(e<1e-8 for _,e,_ in rollouts),
      "native_exact_rollouts":sum(e<1e-8 for _,_,e in rollouts),
      "native_library_bytes":(output/(name+".so")).stat().st_size,
      "native_rollout_steps_per_s":round(n*iterations/secs,1)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=Path("artifacts/m9"))
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rng=random.Random(41)
    reset_train=[(float(rng.randrange(-7,8)),
        float(rng.choice([-4,-3,-2,-1,0,1,2]))) for _ in range(700)]
    reset_train=[(s,x,reset_counter(s,x)) for s,x in reset_train]
    reset_test=[(float(rng.randrange(-12,13)),
        float(rng.choice([-6,-5,-4,-3,-2,-1,0,1,2,3,4]))) for _ in range(300)]
    reset_test=[(s,x,reset_counter(s,x)) for s,x in reset_test]
    clamp_train=[(float(rng.randrange(-7,8)),float(rng.randrange(-4,5)))
        for _ in range(1300)]
    clamp_train=[(s,x,clamp_counter(s,x)) for s,x in clamp_train]
    clamp_test=[(float(rng.randrange(-15,16)),float(rng.randrange(-8,9)))
        for _ in range(300)]
    clamp_test=[(s,x,clamp_counter(s,x)) for s,x in clamp_test]
    nonlinear_train=[(rng.uniform(-.8,.8),
        float(rng.choice([-4,-3,-2,-1,0,1,2]))) for _ in range(650)]
    nonlinear_train=[(s,x,nonlinear_reset(s,x)) for s,x in nonlinear_train]
    nonlinear_test=[(rng.uniform(-1.2,1.2),
        float(rng.choice([-5,-4,-3,-2,-1,0,1,2,3,4,5]))) for _ in range(300)]
    nonlinear_test=[(s,x,nonlinear_reset(s,x)) for s,x in nonlinear_test]
    tasks={"reset":(reset_counter,reset_train,reset_test,123),
           "clamp":(clamp_counter,clamp_train,clamp_test,124),
           "nonlinear":(nonlinear_reset,nonlinear_train,nonlinear_test,125)}
    report={"format":"spark-m9-conditional-program-v1",
      "limitations":["Synthetic deterministic tasks with rules inside fixed grammar.",
       "Training uses numpy; compiled scalar inference is tensor-free.",
       "No evidence of superior general reasoning or Transformer replacement."]}
    for name,(rule,train,ood,seed) in tasks.items():
        report[name]=experiment(name,rule,train,ood,args.output,seed)
    (args.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({name:{
      "branches":report[name]["branches"],"ood":report[name]["ood"],
      "affine_ood":report[name]["affine_ood"],
      "unbranched_degree3_ood":report[name]["unbranched_degree3_ood"],
      "native_exact_rollouts":report[name]["native_exact_rollouts"]}
      for name in tasks},indent=2))

if __name__=="__main__":main()

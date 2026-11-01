#!/usr/bin/env python3
"""M13: verify live neural function adaptation through changing environments."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spark.plastic_function import PlasticFunction,PlasticProgram,cpp_source


def truth(s,x,c,sign):
    d=s[0]-s[1]
    q=sign*(.72*math.tanh(1.2*d+.45*x+.65*c)+.15*math.tanh(c*x))
    return (float(x),s[0],.6*s[2]+q,.5*s[3]+.2*s[2]+q)


def sample(seed,n):
    r=random.Random(seed)
    for i in range(n):
        s=tuple(r.uniform(-.8,.8) for _ in range(4))
        x=r.uniform(-.9,.9);c=r.uniform(-1.,1.)
        yield s,x,c


def score(model,examples,sign):
    errors=[]
    for s,x,c in examples:
        pred=model.step(s,x,c);y=truth(s,x,c,sign)
        errors.extend((a-b)**2 for a,b in zip(pred,y))
    return sum(errors)/len(errors)


class Native:
    def __init__(self,model:PlasticProgram,path:Path):
        compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if not compiler:raise RuntimeError('C++17 compiler required')
        path.parent.mkdir(parents=True,exist_ok=True)
        src=path.with_suffix('.cpp');src.write_text(cpp_source(model.program,capacity=model.function.capacity))
        subprocess.run([compiler,'-std=c++17','-O3','-shared','-fPIC',str(src),'-o',str(path)],check=True)
        self.lib=ctypes.CDLL(str(path.resolve()))
        p=ctypes.c_void_p;v=ctypes.POINTER(ctypes.c_double);sz=ctypes.c_size_t
        self.lib.spark_plastic_create_v1.argtypes=[]
        self.lib.spark_plastic_create_v1.restype=p
        self.lib.spark_plastic_destroy_v1.argtypes=[p]
        self.lib.spark_plastic_set_v1.argtypes=[p,v,sz,ctypes.c_int]
        self.lib.spark_plastic_set_v1.restype=ctypes.c_int
        self.lib.spark_plastic_get_v1.argtypes=[p,v,sz]
        self.lib.spark_plastic_get_v1.restype=ctypes.c_int
        self.lib.spark_plastic_resize_v1.argtypes=[p,ctypes.c_int]
        self.lib.spark_plastic_resize_v1.restype=ctypes.c_int
        self.lib.spark_plastic_active_v1.argtypes=[p]
        self.lib.spark_plastic_active_v1.restype=ctypes.c_int
        self.lib.spark_plastic_step_v1.argtypes=[p,v,ctypes.c_double,ctypes.c_double,v]
        self.lib.spark_plastic_step_v1.restype=ctypes.c_int
        self.lib.spark_plastic_observe_v1.argtypes=[p,v,ctypes.c_double,ctypes.c_double,v,ctypes.c_double]
        self.lib.spark_plastic_observe_v1.restype=ctypes.c_int
        self.lib.spark_plastic_native_stream_v1.argtypes=[p,v,v,v,v,sz,ctypes.c_double]
        self.lib.spark_plastic_native_stream_v1.restype=ctypes.c_double
        self.handle=self.lib.spark_plastic_create_v1()
        if not self.handle:raise RuntimeError('native allocation failed')
        self.sync_from_python(model)
        self.capacity=model.function.capacity
    def close(self):
        if self.handle:self.lib.spark_plastic_destroy_v1(self.handle);self.handle=None
    def sync_from_python(self,model):
        flat=model.function.flatten()
        arr=(ctypes.c_double*len(flat))(*flat)
        if not self.lib.spark_plastic_set_v1(self.handle,arr,len(flat),model.function.active):raise RuntimeError('invalid native parameter load')
    def read_params(self):
        n=self.capacity*9+1
        arr=(ctypes.c_double*n)()
        assert self.lib.spark_plastic_get_v1(self.handle,arr,n)
        return tuple(arr)
    def set_active(self,count):
        if not self.lib.spark_plastic_resize_v1(self.handle,count):raise ValueError('invalid native active width')
    def step(self,s,x,c):
        out=(ctypes.c_double*4)()
        if not self.lib.spark_plastic_step_v1(self.handle,(ctypes.c_double*4)(*s),x,c,out):raise ValueError('native step rejected')
        return tuple(out)
    def observe(self,s,x,c,target,lr):
        if not self.lib.spark_plastic_observe_v1(self.handle,(ctypes.c_double*4)(*s),x,c,
                       (ctypes.c_double*4)(*target),lr):raise ValueError('native update rejected')
    def stream(self,examples,lr=0.0):
        n=len(examples)
        ss=(ctypes.c_double*(n*4))(*(v for s,x,c,y in examples for v in s))
        xx=(ctypes.c_double*n)(*(x for s,x,c,y in examples))
        cc=(ctypes.c_double*n)(*(c for s,x,c,y in examples))
        yy=(ctypes.c_double*(n*4))(*(v for s,x,c,y in examples for v in y))
        return self.lib.spark_plastic_native_stream_v1(self.handle,ss,xx,cc,yy,n,lr)


def experiment(out,*,train_steps=1100,drift_steps=850,rate=.06):
    out.mkdir(parents=True,exist_ok=True)
    net=PlasticFunction.initialize(capacity=12,active=8,seed=37)
    model=PlasticProgram(net)
    native=Native(model,out/'live_function.so')
    initial=model.function.flatten()
    probes=list(sample(702,220))
    pre=list(sample(107,train_steps));post=list(sample(501,drift_steps))
    # Environment A: online learning of the normal mapping; no offline training.
    for s,x,c in pre:
        y=truth(s,x,c,1)
        model.observe(s,x,c,y,lr=rate)
        native.observe(s,x,c,y,rate)
    stable_mse=score(model,probes,1)
    frozen=PlasticProgram(PlasticFunction.initialize(capacity=12,active=8,seed=37))
    frozen.function.units=[w.copy() for w in model.function.units]
    frozen.function.bias=model.function.bias
    before_drift=score(model,probes,-1)
    frozen_mse=score(frozen,probes,-1)
    # Isolate parameter plasticity from the change in active width.
    fixed_width_online=PlasticProgram(PlasticFunction.initialize(capacity=12,active=8,seed=37))
    fixed_width_online.function.units=[w.copy() for w in model.function.units]
    fixed_width_online.function.bias=model.function.bias
    frozen_expanded=PlasticProgram(PlasticFunction.initialize(capacity=12,active=12,seed=37))
    frozen_expanded.function.units=[w.copy() for w in model.function.units]
    frozen_expanded.function.bias=model.function.bias
    frozen_expanded_mse=score(frozen_expanded,probes,-1)
    # Structural runtime adaptation: make more hidden units active while the
    # environment changes. No C++ recompilation or parameter-literal patching.
    model.function.set_active(12)
    native.set_active(12)
    mid_error=None
    for t,(s,x,c) in enumerate(post):
        y=truth(s,x,c,-1)
        model.observe(s,x,c,y,lr=rate)
        fixed_width_online.observe(s,x,c,y,lr=rate)
        native.observe(s,x,c,y,rate)
        if t==drift_steps//2:mid_error=score(model,probes,-1)
    after_drift=score(model,probes,-1)
    after_fixed_width=score(fixed_width_online,probes,-1)
    native_delta=max(abs(a-b) for s,x,c in probes for a,b in zip(model.step(s,x,c),native.step(s,x,c)))
    parameter_delta=max(abs(a-b) for a,b in zip(initial,model.function.flatten()))
    native_params_diff=max(abs(a-b) for a,b in zip(model.function.flatten(),native.read_params()))
    # Fresh independent stream used only for runtime throughput. No statistics
    # from it contribute to validation or training quality.
    speed=[(s,x,c,truth(s,x,c,-1)) for s,x,c in sample(812,2048)]
    native.stream(speed[:32],lr=0)
    t=perf_counter()
    repetitions=30
    for _ in range(repetitions):native.stream(speed,lr=0)
    inference_seconds=perf_counter()-t
    t=perf_counter()
    for _ in range(repetitions):native.stream(speed,lr=rate)
    adapt_seconds=perf_counter()-t
    model.save(out/'live_function.checkpoint.json')
    reloaded=PlasticProgram.load(out/'live_function.checkpoint.json')
    checkpoint_error=max(abs(a-b) for s,x,c in probes for a,b in zip(model.step(s,x,c),reloaded.step(s,x,c)))
    report={
        'format':'spark-m13-live-plastic-neural-function-v1',
        'online_only':True, 'neural_initial_active':8,'neural_final_active':12,
        'neural_capacity':12,'mutable_scalar_parameters':model.function.parameter_count,
        'observations_environment_a':len(pre),'observations_after_abrupt_drift':len(post),
        'held_out_probes':len(probes),
        'environment_a_mse_after_online_learning':stable_mse,
        'environment_b_mse_before_adaptation':before_drift,
        'environment_b_frozen_mse':frozen_mse,
        'environment_b_frozen_expanded_width_mse':frozen_expanded_mse,
        'environment_b_online_fixed_width_mse':after_fixed_width,
        'environment_b_mse_halfway':mid_error,
        'environment_b_mse_after_live_adaptation':after_drift,
        'environment_b_adaptation_improvement_factor':frozen_mse/after_drift,
        'max_initial_to_final_parameter_change':parameter_delta,
        'max_python_native_step_difference':native_delta,
        'max_python_native_parameter_difference':native_params_diff,
        'max_checkpoint_output_difference':checkpoint_error,
        'native_steps_per_second_static':len(speed)*repetitions/inference_seconds,
        'native_steps_per_second_with_online_learning':len(speed)*repetitions/adapt_seconds,
        'limitations':['Synthetic supervised transitions with directly observed next state.',
                       'The surrounding symbolic state equation and 3 neural inputs are prescribed.',
                       'Online learning requires feedback; this is not unsupervised continuous adaptation.',
                       'The number of hidden units can change within a preallocated capacity; no arbitrary topology growth or automatic structure discovery.',
                       'Throughput reports batch C++ transitions, not guaranteed hard real-time latency.',
                       'No language or Transformer quality benchmark.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    native.close()
    print(json.dumps(report,indent=2,sort_keys=True))
    assert stable_mse<.006, report
    assert after_drift<frozen_mse*.10,report
    assert after_fixed_width<frozen_mse*.10,report
    assert after_drift<frozen_expanded_mse*.10,report
    assert after_drift<before_drift*.10,report
    assert after_drift<.007,report
    assert native_delta<1e-9,report
    assert native_params_diff<1e-9,report
    assert checkpoint_error==0,report
    assert parameter_delta>.005,report
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('artifacts/m13'))
    args=ap.parse_args()
    experiment(args.output)

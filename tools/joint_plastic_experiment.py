#!/usr/bin/env python3
"""M21 reproducible joint program/memory learning and live native neural function."""
from __future__ import annotations
import argparse
import copy
import ctypes
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.joint_plastic_memory import JointMemory,generate,joint_search,replay_joint
from spark.plastic_function import PlasticFunction
from spark.addressed_program import AddressedProgram,discover_policy


def native_compile(program,out):
    compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise RuntimeError('C++17 compiler required')
    cpp=out.with_suffix('.cpp');cpp.write_text(program.cpp())
    subprocess.run([compiler,'-std=c++17','-O3','-fPIC','-shared',str(cpp),'-o',str(out)],check=True)
    lib=ctypes.CDLL(str(out.resolve()))
    void=ctypes.c_void_p; ptr=ctypes.POINTER(ctypes.c_double)
    lib.m21_create.argtypes=();lib.m21_create.restype=void
    lib.m21_destroy.argtypes=(void,);lib.m21_destroy.restype=None
    lib.m21_set.argtypes=(void,ptr,ctypes.c_size_t,ctypes.c_int);lib.m21_set.restype=ctypes.c_int
    lib.m21_get.argtypes=(void,ptr,ctypes.c_size_t);lib.m21_get.restype=ctypes.c_int
    lib.m21_parameters.argtypes=();lib.m21_parameters.restype=ctypes.c_size_t
    lib.m21_reset.argtypes=(void,);lib.m21_reset.restype=None
    lib.m21_event.argtypes=(void,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,
      ctypes.c_double,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_double,ptr)
    lib.m21_event.restype=ctypes.c_int
    return lib


def native_evaluate(program,streams,so,*,learn=False,lr=.04):
    lib=native_compile(program,so)
    p=lib.m21_create()
    if not p:raise RuntimeError('native allocation failed')
    weights=program.function.flatten()
    size=len(weights)
    assert size==lib.m21_parameters(),(size,lib.m21_parameters())
    assert lib.m21_set(p,(ctypes.c_double*size)(*weights),size,program.function.active)==1
    py=copy.deepcopy(program)
    error=0.;queries=ops=0;max_delta=0.
    try:
        for seq in streams:
            lib.m21_reset(p)
            py_slots=[None]*program.capacity
            for e in seq:
                y=ctypes.c_double()
                ok=lib.m21_event(p,e.u,e.v,e.a,e.b,e.value,int(e.query),e.target,int(learn),lr,ctypes.byref(y))
                if ok!=1:raise RuntimeError('valid native observation rejected')
                k=program.address.get(e,program.capacity)
                if e.query:
                    if k<0 or py_slots[k] is None:
                        python_pred=0.
                    else:
                        z=py_slots[k];q,cache=py.function.forward(z)
                        python_pred=z[0]+q
                        if learn:py.function.update(z,python_pred-e.target,lr,cache)
                    max_delta=max(max_delta,abs(y.value-python_pred))
                    error+=(y.value-e.target)**2
                    queries+=1
                elif k>=0:
                    from spark.addressed_program import eval_node
                    if eval_node(program.policy,e.a,e.b):py_slots[k]=(e.value,e.a,e.b)
                ops+=1
        new=(ctypes.c_double*size)()
        assert lib.m21_get(p,new,size)==1
        param_delta=max(abs(a-b) for a,b in zip(new,py.function.flatten()))
    finally:
        lib.m21_destroy(p)
    return {'operations':ops,'reads':queries,'mse':error/queries,
            'max_python_native_prediction_delta':max_delta,
            'max_python_native_updated_parameter_delta':param_delta,
            'native_library_bytes':so.stat().st_size}


def run(out,*,train_count=100,test_count=36,coarse_count=35,beam=90):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train=generate(2021,train_count,keys=4,distractors=(6,18),queries=4,scale=1.3,mode=1)
    test=generate(3031,test_count,keys=8,distractors=(512,2048),queries=3,scale=2.5,mode=1)
    begin=perf_counter()
    symbolic,structure=joint_search(train,capacity=8,coarse_count=coarse_count,beam=beam)
    search_seconds=perf_counter()-begin
    shallow,shallow_info=discover_policy(train,symbolic.address,capacity=8,depth=2)
    network=PlasticFunction.initialize(capacity=12,active=8,seed=37)
    model=JointMemory(symbolic.address,symbolic.policy,8,network)
    # No direct neural subfunction labels. Only query errors of the whole system.
    before=model.evaluate(train,learn=False)
    train_prequential=model.evaluate(train,learn=True,lr=.07)
    # Second pass is still train-only; improves coverage of rare contexts.
    train_prequential_2=model.evaluate(train,learn=True,lr=.035)
    model.save(out/'learned_joint_program.json')
    loaded=JointMemory.load(out/'learned_joint_program.json')
    held=model.evaluate(test,learn=False)
    base_raw=replay_joint(test,symbolic.address,symbolic.policy,8)
    shallow_raw=replay_joint(test,symbolic.address,shallow.policy,8)
    def always_mse(streams):
        sse=0;reads=0
        for seq in streams:
            slots=[0.]*8
            for e in seq:
                k=symbolic.address.get(e,8)
                if e.query:
                    pred=slots[k] if k>=0 else 0.
                    sse+=(pred-e.target)**2; reads+=1
                elif k>=0: slots[k]=e.value
        return {'reads':reads,'mse':sse/reads}
    no_policy=always_mse(test)
    native=native_evaluate(model,test,out/'joint_memory.so')
    roundtrip=loaded.evaluate(test,learn=False)
    # Abrupt reversal: keep exactly the learned structure, change only the live
    # dense function from feedback. Evaluate predictions *before* updates.
    change=generate(4041,450,keys=4,distractors=(5,20),queries=3,scale=1.5,mode=-1)
    frozen=copy.deepcopy(model)
    adaptive=copy.deepcopy(model)
    shift_frozen=frozen.evaluate(change,learn=False)
    shift_live=adaptive.evaluate(change,learn=True,lr=.07)
    shift_native=native_evaluate(model,change[:16],out/'joint_memory.so',learn=True,lr=.07)
    def seg_errors(predictions,streams):
        targets=[e.target for seq in streams for e in seq if e.query]
        n=len(targets)
        return {'first_100_mse':sum((a-b)**2 for a,b in zip(predictions[:100],targets[:100]))/100,
                'last_200_mse':sum((a-b)**2 for a,b in zip(predictions[-200:],targets[-200:]))/200,
                'reads':n}
    drift_live=seg_errors(shift_live['predictions'],change)
    drift_frozen=seg_errors(shift_frozen['predictions'],change)
    report={'format':'spark-m21-joint-plastic-program-v1',
      'train_episodes':len(train),'train_keys':4,'test_episodes':len(test),'test_keys':8,
      'calibration_episodes':0,'heldout_distractors':[512,2048],
      'search_time_seconds':search_seconds,'joint_search':structure,
      'train_untrained_neural':{'mse':before['mse'],'reads':before['reads']},
      'train_neural_first_pass':{'mse':train_prequential['mse'],'last_100_mse':train_prequential['last_100_mse']},
      'train_neural_second_pass':{'mse':train_prequential_2['mse'],'last_100_mse':train_prequential_2['last_100_mse']},
      'learned_neural_parameters':model.function.parameter_count,'active_neural_units':model.function.active,
      'heldout_neuro_symbolic':{'reads':held['reads'],'mse':held['mse']},
      'heldout_no_neural':base_raw,'heldout_two_atom_no_neural':shallow_raw,
      'heldout_always_write_no_neural':no_policy,
      'native':native,'serialization_mse':roundtrip['mse'],
      'drift_live':drift_live,'drift_frozen':drift_frozen,
      'drift_native_online':shift_native,
      'limitations':[
        'Address and write program are co-selected from a bounded grammar, via a coarse-to-fine search.',
        'The nonlinear value function is learned after program selection using only delayed query targets: coordinate optimization, not joint differentiable end-to-end search.',
        'Explicit read requests, finite memory capacity, synthetic three-condition relevance and a known update grammar remain.',
        'On reads the stored raw observation is interpreted by the current live neural function; changing its weights can change how older stored observations are interpreted.',
        'Model selection and held-out evaluation do not use a separate all-write address calibration set.',
        'No Transformer benchmark, real-language learning or compute-matched GRU/SSM comparison.']}
    assert structure['selected_address']=={'cu':1,'cv':-1},report['joint_search']
    assert held['mse'] < base_raw['mse'],(held['mse'],base_raw['mse'])
    assert no_policy['mse']>held['mse'],(no_policy,held['mse'])
    assert native['max_python_native_prediction_delta']<1e-10,native
    assert native['max_python_native_updated_parameter_delta']<1e-10,native
    assert shift_native['max_python_native_updated_parameter_delta']<1e-10,shift_native
    assert drift_live['last_200_mse']<drift_frozen['last_200_mse']*.25,(drift_live,drift_frozen)
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('joint_search','heldout_neuro_symbolic','heldout_no_neural',
       'heldout_two_atom_no_neural','heldout_always_write_no_neural','native','drift_live','drift_frozen')},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('artifacts/m21'))
    p.add_argument('--train-count',type=int,default=100)
    p.add_argument('--test-count',type=int,default=36)
    p.add_argument('--coarse-count',type=int,default=35)
    p.add_argument('--beam',type=int,default=90)
    a=p.parse_args();run(a.output,train_count=a.train_count,test_count=a.test_count,
                           coarse_count=a.coarse_count,beam=a.beam)

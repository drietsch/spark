#!/usr/bin/env python3
"""M18: discover context-dependent relevance from delayed targets, compile and test."""
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
from spark.context_memory import ContextRouter,Expression,Observation,adapt_prequential,episodes,fit,metrics


def native_verify(router,streams,out):
    compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise RuntimeError('C++17 compiler is needed for native parity')
    cpp=out.with_suffix('.cpp')
    cpp.write_text(router.cpp_source())
    subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(out)],check=True)
    class State(ctypes.Structure):
        _fields_=[('slots',ctypes.c_double*router.keys)]
    lib=ctypes.CDLL(str(out.resolve()))
    ptr=ctypes.POINTER(State)
    lib.spark_context_reset_v1.argtypes=(ptr,)
    lib.spark_context_reset_v1.restype=None
    lib.spark_context_event_v1.argtypes=(ptr,ctypes.c_int,ctypes.c_double,ctypes.c_double,
                                         ctypes.c_double,ctypes.c_int,ctypes.POINTER(ctypes.c_double))
    lib.spark_context_event_v1.restype=ctypes.c_int
    mismatch=operations=reads=0
    native_hits=0
    for seq in streams:
        state=State();lib.spark_context_reset_v1(ctypes.byref(state))
        memory=[0.0]*router.keys
        for event in seq:
            ref=memory[event.key]
            if not event.query and router.expression.evaluate(event.a,event.b):
                memory[event.key]=event.value
                ref=event.value
            result=ctypes.c_double()
            ok=lib.spark_context_event_v1(ctypes.byref(state),event.key,event.a,event.b,
                                            event.value,int(event.query),ctypes.byref(result))
            if not ok:raise RuntimeError('native event rejected valid input')
            mismatch+=abs(result.value-ref)>1e-12
            if event.query:
                reads+=1
                native_hits+=abs(result.value-event.target)<1e-10
            operations+=1
    return {'operations':operations,'reads':reads,'correct':native_hits,
            'python_native_mismatches':mismatch,'native_library_bytes':out.stat().st_size}


def constant_policy(write,keys):
    # Always and never write are controls; not learned and not in candidate grammar.
    class Control:
        def replay(self,seq):
            slots=[0.0]*keys;error=0.0;hits=reads=0
            for e in seq:
                if e.query:
                    delta=slots[e.key]-e.target
                    error+=delta*delta;hits+=abs(delta)<1e-10;reads+=1
                elif write:slots[e.key]=e.value
            return error,hits,reads
    return Control()


def run(out,*,train_count=320,long_count=48):
    out.mkdir(parents=True,exist_ok=True)
    start=perf_counter()
    train=episodes(1818,train_count,keys=4,distractors=(4,18),queries=3)
    learned,fit_report=fit(train,keys=8)
    linear,linear_report=fit(train,keys=8,linear_only=True)
    training_time=perf_counter()-start
    # Completely disjoint episode seeds, longer delays, twice the key capacity,
    # and larger raw context magnitudes than in training.
    short=episodes(1991,80,keys=8,distractors=(6,30),scale=1.8,queries=3)
    long=episodes(1992,long_count,keys=8,distractors=(1024,4096),scale=2.4,queries=3)
    result={
        'format':'spark-m18-context-discovery-v1',
        'training_episodes':len(train),
        'training_keys':4,'test_keys':8,
        'training_distractors':[4,18], 'long_test_distractors':[1024,4096],
        'training_context_magnitude_max':1.0,'long_test_context_magnitude_max':2.4,
        'candidate_fit_seconds':training_time,
        'candidate_fit':fit_report,
        'learned_equation':learned.expression.to_json(),
        'short':metrics(learned,short),
        'long':metrics(learned,long),
        'linear_only_equation':linear.expression.to_json(),
        'linear_only_fit':linear_report,
        'linear_only_long':metrics(linear,long),
        'always_write_long':metrics(constant_policy(True,8),long),
        'never_write_long':metrics(constant_policy(False,8),long),
        'native':native_verify(learned,long,out/'context_memory.so'),
    }
    learned.save(out/'learned_context_program.json')
    result['loaded_program_long']=metrics(ContextRouter.load(out/'learned_context_program.json'),long)
    # Model never receives a regime label. Query feedback is added only after
    # predicting an episode. The old regime returns after an intervening shift.
    stages=[episodes(3001,48,keys=8,distractors=(8,30),scale=2.0,mode='same_sign'),
            episodes(3002,80,keys=8,distractors=(8,30),scale=2.0,mode='opposite_sign'),
            episodes(3003,80,keys=8,distractors=(8,30),scale=2.0,mode='same_sign')]
    result['adaptation']=adapt_prequential(learned,stages,window=24,refresh_every=6)
    result['limitations']=[
        'Synthetic relevance is an equation deliberately expressible within a fixed six-feature grammar.',
        'All candidate selection uses query targets, not direct relevance/write labels.',
        'Keys and read requests are still explicit; the environment is not natural language.',
        'Hard gate trained via discrete downstream loss, not differentiable end-to-end neural credit assignment.',
        'Test policy has access to raw context pairs, but no binary event relevance cue.',
        'Unseen keys/delays test generalization of shared symbolic policy, not open-world semantics.',
        'No compute-matched GRU/SSM/Transformer study, no general intelligence claim.'
    ]
    assert result['learned_equation']=={'feature':'product','comparison':'gt','threshold':0.0},result
    assert result['long']['correct']==result['long']['reads'],result
    assert result['linear_only_long']['correct']<result['long']['correct'],result
    assert result['always_write_long']['correct']<result['long']['correct'],result
    assert result['never_write_long']['correct']<result['long']['correct'],result
    assert result['native']['python_native_mismatches']==0,result
    assert result['native']['correct']==result['long']['correct'],result
    assert result['adaptation'][1]['last_24_accuracy']>.95,result
    assert result['adaptation'][2]['last_24_accuracy']>.95,result
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('learned_equation','long','linear_only_long','always_write_long','native','adaptation')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('artifacts/m18'))
    args=p.parse_args()
    run(args.output)

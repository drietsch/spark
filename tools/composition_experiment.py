#!/usr/bin/env python3
"""M19: learn Boolean composition from delayed memory read losses, compile C++."""
from __future__ import annotations
import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.composed_memory import ComposedRouter,generate,search,Program,Expression


def native_parity(router,episodes,out):
    compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise RuntimeError('C++17 compiler required')
    cpp=out.with_suffix('.cpp');cpp.write_text(router.cpp())
    subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(out)],check=True)
    class Memory(ctypes.Structure):
        _fields_=[('slots',ctypes.c_double*router.keys)]
    lib=ctypes.CDLL(str(out.resolve()));ptr=ctypes.POINTER(Memory)
    lib.m19_reset.argtypes=(ptr,);lib.m19_reset.restype=None
    lib.m19_event.argtypes=(ptr,ctypes.c_int,ctypes.c_double,ctypes.c_double,
                              ctypes.c_double,ctypes.c_int,ctypes.POINTER(ctypes.c_double))
    lib.m19_event.restype=ctypes.c_int
    mismatches=operations=correct=reads=0
    for seq in episodes:
        native=Memory();lib.m19_reset(ctypes.byref(native));reference=[0.]*router.keys
        for e in seq:
            if not e.query and router.program.eval(e.a,e.b):reference[e.key]=e.value
            output=ctypes.c_double()
            if lib.m19_event(ctypes.byref(native),e.key,e.a,e.b,e.value,int(e.query),ctypes.byref(output))!=1:
                raise RuntimeError('native rejected valid input')
            mismatches+=abs(output.value-reference[e.key])>1e-12
            if e.query:
                reads+=1;correct+=abs(output.value-e.target)<1e-10
            operations+=1
    return {'operations':operations,'reads':reads,'correct':correct,
            'python_native_mismatches':mismatches,'native_library_bytes':out.stat().st_size}


def constant_memory_score(episodes,keys,*,write):
    hits=reads=0
    loss=0.0
    for seq in episodes:
        slots=[0.0]*keys
        for e in seq:
            if e.query:
                delta=slots[e.key]-e.target
                loss+=delta*delta
                hits+=abs(delta)<1e-10
                reads+=1
            elif write:slots[e.key]=e.value
    return {'reads':reads,'correct':hits,'query_mse':loss/reads,'accuracy':hits/reads}


def run(out,train_count=160,test_count=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train=generate(1919,train_count,keys=4,distractors=(5,18),queries=3,scale=1.0)
    start=perf_counter()
    learned,fit=search(train,keys=8)
    atomic,afit=search(train,keys=8,composition=False)
    duration=perf_counter()-start
    # Test on independent seeds, twice as many keys, much longer delays,
    # and BOTH near-threshold and wider context magnitudes.
    holdout=generate(2727,test_count,keys=8,distractors=(1024,4096),queries=3,scale=2.4)
    learned.save(out/'learned_composed_program.json')
    loaded=ComposedRouter.load(out/'learned_composed_program.json')
    report={'format':'spark-m19-compositional-routing-v1',
            'train_episodes':len(train),'train_keys':4,'heldout_episodes':len(holdout),
            'heldout_keys':8,'train_delay':[5,18],'heldout_delay':[1024,4096],
            'training_search_seconds':duration,'search':fit,
            'atomic_search':afit,'composed':learned.evaluate(holdout),
            'single_predicate':atomic.evaluate(holdout),
            'always_write':constant_memory_score(holdout,8,write=True),
            'never_write':constant_memory_score(holdout,8,write=False),
            'loaded':loaded.evaluate(holdout),
            'native':native_parity(learned,holdout,out/'composed_memory.so'),
            'limitations':[
              'Synthetic hidden relevance lies within an explicit 2-atom AND/OR grammar.',
              'Search is exhaustive over 1296 candidate programs; not unrestricted evolution.',
              'Keys and read operations are given; only event relevance is inferred.',
              'The dataset generator includes hidden rule; training sees only queries/targets, no write labels.',
              'No neural co-training, no general language tasks, no Transformer superiority claim.'
            ]}
    assert report['search']['candidates']==1296
    assert learned.program.op in ('and','or'),report
    assert report['composed']['correct']==report['composed']['reads'],report
    assert report['single_predicate']['correct']<report['composed']['correct'],report
    assert report['always_write']['correct']<report['composed']['correct'],report
    assert report['never_write']['correct']<report['composed']['correct'],report
    assert report['native']['python_native_mismatches']==0,report
    assert report['native']['correct']==report['composed']['correct'],report
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('search','composed','single_predicate','always_write','never_write','native')},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/m19'))
    p.add_argument('--train-count',type=int,default=160)
    p.add_argument('--test-count',type=int,default=42)
    args=p.parse_args();run(args.output,args.train_count,args.test_count)

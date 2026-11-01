#!/usr/bin/env python3
"""M20: delayed-supervision discovery of address and 3-step write programs."""
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
from spark.addressed_program import (Address,AddressedProgram,Observation,
    generate,learn_address,discover_policy,replay,eval_node)


def native_check(model,streams,path):
    compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise RuntimeError('C++17 compiler is required')
    path=Path(path);path.with_suffix('.cpp').write_text(model.cpp())
    subprocess.run([compiler,'-std=c++17','-O3','-shared','-fPIC',
                    str(path.with_suffix('.cpp')),'-o',str(path)],check=True)
    n=model.capacity
    class State(ctypes.Structure):
        _fields_=[('slots',ctypes.c_double*n)]
    lib=ctypes.CDLL(str(path.resolve()))
    ptr=ctypes.POINTER(State)
    lib.m20_reset.argtypes=(ptr,);lib.m20_reset.restype=None
    lib.m20_event.argtypes=(ptr,ctypes.c_double,ctypes.c_double,
        ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.POINTER(ctypes.c_double))
    lib.m20_event.restype=ctypes.c_int
    reads=correct=ops=disagreements=0
    for episode in streams:
        state=State();lib.m20_reset(ctypes.byref(state));reference=[0.]*n
        for e in episode:
            idx=model.address.get(e,n)
            if not e.query and idx>=0 and eval_node(model.policy,e.a,e.b):
                reference[idx]=e.value
            y=ctypes.c_double()
            ok=lib.m20_event(ctypes.byref(state),e.u,e.v,e.a,e.b,e.value,int(e.query),ctypes.byref(y))
            if ok!=1:raise RuntimeError('native runtime rejected a valid event')
            prediction=reference[idx] if idx>=0 else 0.
            disagreements+=abs(y.value-prediction)>1e-10
            if e.query:
                reads+=1;correct+=abs(y.value-e.target)<1e-9
            ops+=1
    return {'operations':ops,'reads':reads,'correct':correct,
            'python_native_disagreements':disagreements,'library_bytes':path.stat().st_size}


def run(out,*,train_count=110,test_count=48,beam=90):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Calibration: no address labels; every event is valid, and only a delayed
    # query gives a target. Learned independently of relevance policy.
    calibration=generate(2019,55,keys=4,distractors=(4,12),queries=3,calibration=True)
    start=perf_counter()
    address,address_fit=learn_address(calibration,capacity=8)
    train=generate(2020,train_count,keys=4,distractors=(5,18),queries=4,scale=1.3)
    model,fit=discover_policy(train,address,8,depth=3,beam=beam)
    shallow,shallow_fit=discover_policy(train,address,8,depth=2)
    search_seconds=perf_counter()-start
    # Independent unseen object IDs 4..7, different nuisance values for writes
    # and reads, larger context magnitudes, and much longer distractor streams.
    heldout=generate(9090,test_count,keys=8,distractors=(1024,4096),queries=3,scale=2.5)
    model.save(out/'learned_addressed_program.json')
    loaded=AddressedProgram.load(out/'learned_addressed_program.json')
    compiled=native_check(model,heldout,out/'addressed_memory.so')
    # Ablations preserve policy while replacing learned mathematical address,
    # or preserve address while omitting structural program search.
    wrong_address=AddressedProgram(Address(1,0),model.policy,8)
    always=replay(heldout,address,None,8,always=True)
    report={'format':'spark-m20-addressed-compiled-program-v1',
        'calibration_episodes':len(calibration),'train_episodes':len(train),
        'train_keys':4,'holdout_episodes':len(heldout),'holdout_keys':8,
        'training_distractors':[5,18],'heldout_distractors':[1024,4096],
        'address_search':address_fit,'learned_address':address.json(),
        'program_search':fit,'two_atom_search':shallow_fit,
        'train_search_seconds':search_seconds,
        'heldout':model.evaluate(heldout),'two_atom_baseline':shallow.evaluate(heldout),
        'raw_first_feature_address_baseline':wrong_address.evaluate(heldout),
        'always_write_baseline':always,'serialized_parity':loaded.evaluate(heldout),
        'native':compiled,
        'limitations':[
          'Synthetic deterministic target rule is contained in a finite three-atom grammar.',
          'Address learning is limited to eight preselected linear projections with round-to-integer indexing.',
          'Address calibration has a known all-write phase; no integer keys or address labels are supplied.',
          'Read-request events and target values are explicit supervision; memory capacity is allocated in advance.',
          'The learned routing and address programs are separate stages, not end-to-end joint learning.',
          'No live neural primitive is jointly trained in this experiment and no Transformer comparison is made.']}
    assert fit['complexity']>=7,fit
    assert report['heldout']['correct']==report['heldout']['reads'],report
    assert report['native']['python_native_disagreements']==0,report
    assert report['native']['correct']==report['heldout']['correct'],report
    assert report['two_atom_baseline']['correct']<report['heldout']['correct'],report
    assert report['raw_first_feature_address_baseline']['correct']<report['heldout']['correct'],report
    assert report['always_write_baseline']['correct']<report['heldout']['correct'],report
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('learned_address','program_search','heldout','two_atom_baseline',
            'raw_first_feature_address_baseline','always_write_baseline','native')},indent=2))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('artifacts/m20'))
    ap.add_argument('--train-count',type=int,default=110)
    ap.add_argument('--test-count',type=int,default=48)
    ap.add_argument('--beam',type=int,default=90)
    args=ap.parse_args();run(args.output,train_count=args.train_count,
                             test_count=args.test_count,beam=args.beam)

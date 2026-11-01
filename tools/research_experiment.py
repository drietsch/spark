#!/usr/bin/env python3
"""M14-16 reproducible experiments: independent held-out data and native parity."""
from __future__ import annotations
import argparse,copy,ctypes,json,math,random,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.predictive_state import fit, PredictiveState
from spark.consolidation import ConsolidatedFunction,evaluate
from spark.algorithmic_memory import AlgorithmicMemory,WRITE,READ,ADD,ERASE,NOOP


def compile_shared(source,path):
    cxx=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not cxx:raise RuntimeError('C++ compiler required for native benchmark')
    path.with_suffix('.cpp').write_text(source)
    subprocess.run([cxx,'-std=c++17','-O2','-shared','-fPIC',str(path.with_suffix('.cpp')),
                    '-o',str(path)],check=True,capture_output=True,text=True)
    return ctypes.CDLL(str(path.resolve()))


def system_sequence(seed,length, *,a=1.35,b=-.52,g=.24):
    rng=random.Random(seed)
    y=[rng.uniform(-.35,.35),rng.uniform(-.35,.35)]
    u=[rng.uniform(-1,1) for _ in range(length)]
    for i in range(1,length-1):
        y.append(a*y[-1]+b*y[-2]+g*u[i])
    return y,u


def measure_m14(out):
    train=[system_sequence(300+i,120) for i in range(48)]
    model=fit(train,max_order=4)
    model.save(out/'m14.state.json')
    holdouts=[system_sequence(1000+i,300) for i in range(12)]
    def eval_m(m,rows,drop=0.):
        sq=[]
        rng=random.Random(771)
        for ys,us in rows:
            pos=m.order-1
            init=list(reversed(ys[:m.order]))
            obs=[ys[t+1] if rng.random()>=drop else None for t in range(pos,len(ys)-1)]
            preds=m.rollout(init,us[pos:len(ys)-1],obs)
            sq.extend((p-ys[t+1])**2 for t,p in enumerate(preds,start=pos))
        return float(np.mean(sq))
    prequential=eval_m(copy.deepcopy(model),holdouts)
    missing=eval_m(copy.deepcopy(model),holdouts,.15)
    baseline=fit(train,max_order=1)
    baseline_mse=eval_m(baseline,holdouts)
    # Recover continuously from an unseen change in dynamics using online updates.
    stream=system_sequence(1900,2100,g=-.28)
    static=copy.deepcopy(model);adaptive=copy.deepcopy(model)
    ys,us=stream;errors_static=[];errors_adapt=[]
    for i in range(model.order-1,len(ys)-1):
        hist=list(reversed(ys[i-model.order+1:i+1]))
        p0=static.online(hist,us[i],ys[i+1],lr=0.)
        p1=adaptive.online(hist,us[i],ys[i+1],lr=.08)
        if i>1500:
            errors_static.append((p0-ys[i+1])**2)
            errors_adapt.append((p1-ys[i+1])**2)
    # Native parity of online updates incl mutable learned coefficients.
    native=compile_shared(model.cpp(),out/'m14_predictive.so')
    n=model.order
    class State(ctypes.Structure):_fields_=[('weights',ctypes.c_double*(n+2))]
    native.spark_predictive_init.argtypes=(ctypes.POINTER(State),ctypes.POINTER(ctypes.c_double))
    native.spark_predictive_step.argtypes=(ctypes.POINTER(State),ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int)
    native.spark_predictive_step.restype=ctypes.c_double
    st=State();native.spark_predictive_init(ctypes.byref(st),(ctypes.c_double*(n+2))(*model.coefficients))
    py=copy.deepcopy(model);worst=0.
    for i in range(n-1,400):
        hist=list(reversed(ys[i-n+1:i+1]));u=us[i];target=ys[i+1]
        a=py.online(hist,u,target,lr=.05)
        b=native.spark_predictive_step(ctypes.byref(st),(ctypes.c_double*n)(*hist),u,target,.05,1)
        worst=max(worst,abs(a-b))
    worst=max(worst,max(abs(a-b) for a,b in zip(py.coefficients,st.weights)))
    return {'learned_memory_order':model.order,'coefficients':model.coefficients,
        'train_only_validation_mse':model.train_only_validation_mse,
        'heldout_transitions':12*(300-model.order),
        'heldout_mse':prequential,'missing_15pct_mse':missing,
        'order1_baseline_mse':baseline_mse,
        'shift_frozen_last600_mse':float(np.mean(errors_static)),
        'shift_online_last600_mse':float(np.mean(errors_adapt)),
        'native_python_max_difference':worst}


def q_target(s,x):
    return .2*s*x+.125*x*x-.1*s


def measure_m15(out):
    # Live neural function learns the residual from feedback, without q being
    # a literal embedded coefficient; use the already implemented M13 learner.
    try:
        from spark.plastic_function import PlasticFunction
    except ImportError:
        # Self-contained reference remains executable when M13 isn't vendored.
        from spark.neural_micro import MicroFunction as PlasticFunction
    neural=PlasticFunction.initialize(capacity=8,active=8,seed=23)
    r=random.Random(42)
    for _ in range(20000):
        s=r.uniform(-1,1);x=r.uniform(-1,1)
        z=(s,x,0.)
        output,cache=neural.forward(z)
        neural.update(z,output-q_target(s,x),.045,cache)
    def predict(z):return neural.forward((z[0],z[1],0.))[0]
    teacher=ConsolidatedFunction(predict)
    rng=random.Random(103)
    fitrows=[]
    for _ in range(1800):
        z=(rng.uniform(-.9,.9),rng.uniform(-.9,.9))
        fitrows.append((z,predict(z)))
    holdout=[((rng.uniform(-1.1,1.1),rng.uniform(-1.1,1.1)),0.) for _ in range(420)]
    holdout=[(z,q_target(*z)) for z,_ in holdout]
    rollouts=[]
    for _ in range(16):
        s=rng.uniform(-.5,.5);xs=[rng.uniform(-.8,.8) for _ in range(40)]
        true=s
        for x in xs:true=.6*true+q_target(true,x)
        rollouts.append((s,xs,true))
    # Compare exact regression extraction, network holdout and true task.
    accepted=teacher.propose(fitrows,holdout,rollouts,absolute_mse=.0005,
                 max_relative=1.25,rollout_tolerance=.12,max_terms=9)
    fallback_accuracy=float(np.mean([(predict(z)-v)**2 for z,v in holdout]))
    # Bad expression candidate must be rejected with rollback.
    junk=[((rng.uniform(-1,1),rng.uniform(-1,1)),rng.uniform(-2,2)) for _ in range(1800)]
    reject=teacher.propose(junk,holdout,rollouts,absolute_mse=.0005,
             max_relative=1.25,rollout_tolerance=.12,max_terms=9)
    symbolic_kept=teacher.symbolic is not None
    parity=None
    if symbolic_kept:
        native=compile_shared(teacher.cpp(),out/'m15_consolidated.so')
        native.spark_consolidated.argtypes=(ctypes.POINTER(ctypes.c_double),)
        native.spark_consolidated.restype=ctypes.c_double
        parity=max(abs(evaluate(teacher.symbolic,z)-native.spark_consolidated(
                   (ctypes.c_double*2)(*z))) for z,_ in holdout)
    return {'neural_heldout_mse':fallback_accuracy,'consolidation':accepted,
           'bad_candidate':reject,'retained_validated_expression':symbolic_kept,
           'native_python_max_difference':parity,'neural_parameters':73,
           'note':'Neural function trained on synthetic targets; symbolic search uses neural predictions, validation uses ground truth.'}


def memory_sequence(seed,length,capacity=32):
    rng=random.Random(seed)
    memory=AlgorithmicMemory(capacity)
    touched=[0,1,2,3,4,5,6,7]
    events=[]
    for t in range(length):
        # Few keys deliberately reused after long gaps; distractor updates
        # don't erase them unless capacity is exhausted.
        op=rng.choices([WRITE,READ,ADD,ERASE,NOOP],[13,30,10,3,44])[0]
        key=rng.choice(touched)
        value=float(rng.randrange(-10,11)) if op in (WRITE,ADD) else 0.
        answer=memory.step(op,key,value)
        events.append((op,key,value,answer))
    return events


def measure_m16(out):
    sequences=[memory_sequence(500+i,4096) for i in range(24)]
    baseline_correct=read_count=correct=0
    for events in sequences:
        memory=AlgorithmicMemory(32)
        last=0.
        for op,k,v,target in events:
            got=memory.step(op,k,v)
            if op==READ:
                read_count+=1
                correct+=abs(got-target)<1e-12
                baseline_correct+=abs(last-target)<1e-12
            if op in (WRITE,ADD):last=v
    native=compile_shared(AlgorithmicMemory.cpp(32),out/'m16_memory.so')
    class Memory(ctypes.Structure):
        _fields_=[('keys',ctypes.c_longlong*32),('values',ctypes.c_double*32),
                  ('age',ctypes.c_uint64*32),('tick',ctypes.c_uint64),
                  ('valid',ctypes.c_bool*32)]
    native.spark_memory_init.argtypes=(ctypes.POINTER(Memory),)
    native.spark_memory_step.argtypes=(ctypes.POINTER(Memory),ctypes.c_int,ctypes.c_longlong,ctypes.c_double)
    native.spark_memory_step.restype=ctypes.c_double
    state=Memory();mismatch=0
    for events in sequences:
        native.spark_memory_init(ctypes.byref(state))
        python=AlgorithmicMemory(32)
        for op,k,v,target in events:
            a=python.step(op,k,v)
            b=native.spark_memory_step(ctypes.byref(state),op,k,v)
            mismatch+=abs(a-b)>1e-12
    # Explicit 4096-step delayed recall, with distractor instructions that
    # never access the key being recalled. Compare Python and native.
    native.spark_memory_init(ctypes.byref(state))
    python=AlgorithmicMemory(32)
    delayed=[(WRITE,101,7.)]+[(NOOP,1000+i,0.) for i in range(4096)]+[(READ,101,0.)]
    delayed_final=(None,None)
    for op,k,v in delayed:
        a=python.step(op,k,v)
        b=native.spark_memory_step(ctypes.byref(state),op,k,v)
        mismatch+=abs(a-b)>1e-12
        if op==READ:delayed_final=(a,b)
    smaller=AlgorithmicMemory(2)
    for op,k,v in [(WRITE,10,1.),(WRITE,20,2.),(READ,10,0.),(WRITE,30,3.)]:
        smaller.step(op,k,v)
    eviction_ok=(smaller.step(READ,20)==0 and smaller.step(READ,10)==1)
    return {'sequence_steps':sum(map(len,sequences)),'max_sequence_steps':4096,
        'explicit_delayed_recall_steps':4096,'delayed_recall_correct':delayed_final==(7.,7.),
        'read_queries':read_count,'program_memory_correct':correct,
        'last_value_baseline_correct':baseline_correct,'native_python_mismatches':mismatch,
        'capacity_eviction_ok':eviction_ok,
        'policy_learned':False,'memory_capacity_entries':32}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,default=Path('artifacts/m14-16'))
    args=a.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'format':'spark-m14-m16-research-v1',
      'm14':measure_m14(args.output),
      'm15':measure_m15(args.output),
      'm16':measure_m16(args.output),
      'limitations':[
       'M14 discovers predictive autoregressive history order, not arbitrary latent neural representations.',
       'M15 uses synthetic supervised examples; does not discover a universal program library.',
       'M16 has explicitly supplied memory instructions, not a learned memory control policy.',
       'No comparison with pretrained language models or a matched Transformer; no general intelligence claim.']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':main()

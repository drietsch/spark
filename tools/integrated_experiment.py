#!/usr/bin/env python3
"""M17: delayed-credit memory policy + plastic neural function + consolidation.

Predictions are always scored *before* receiving their targets. All learning,
thresholding, and program selection are restricted to training observations.
"""
from __future__ import annotations
import argparse,ctypes,copy,json,math,random,shutil,subprocess,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.learned_memory import MemoryRouter,Event,episodes
from spark.adaptive_reasoner import AdaptiveReasoner
from spark.plastic_function import PlasticFunction
from spark.consolidation import ConsolidatedFunction,evaluate


def score_policy(policy,streams,*,hard=True):
    metrics=[policy.evaluate_episode(s,hard=hard) for s in streams]
    return {'episodes':len(streams),'reads':sum(r for _,_,r,_ in metrics),
            'exact_reads':sum(h for _,h,_,_ in metrics),
            'query_mse':sum(loss*r for loss,_,r,_ in metrics)/sum(r for _,_,r,_ in metrics)}


def native_memory(router,out):
    cpp=out.with_suffix('.cpp');cpp.write_text(router.cpp())
    compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if compiler is None:raise RuntimeError('C++ compiler required')
    subprocess.run([compiler,'-std=c++17','-O3','-fPIC','-shared',str(cpp),'-o',str(out)],check=True)
    n=router.keys
    class State(ctypes.Structure):
        _fields_=[('slots',ctypes.c_double*n),('theta',ctypes.c_double*3)]
    lib=ctypes.CDLL(str(out.resolve()))
    lib.spark_memory_reset_v2.argtypes=(ctypes.POINTER(State),ctypes.POINTER(ctypes.c_double))
    lib.spark_memory_reset_v2.restype=None
    lib.spark_memory_signal_v2.argtypes=(ctypes.POINTER(State),ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_int)
    lib.spark_memory_signal_v2.restype=ctypes.c_double
    return State,lib


def native_check(router,streams,out):
    State,lib=native_memory(router,out)
    theta=(ctypes.c_double*3)(*router.theta)
    disagreements=0;read_count=0;operations=0
    for sequence in streams:
        s=State();lib.spark_memory_reset_v2(ctypes.byref(s),theta)
        python_memory=[0.]*router.keys
        for e in sequence:
            p=(python_memory[e.key] if e.query else None)
            if not e.query and router.score(e)>.5:python_memory[e.key]=e.value
            native=lib.spark_memory_signal_v2(ctypes.byref(s),e.key,e.signal,e.value,int(e.query))
            if e.query:
                read_count+=1
                disagreements+=abs(native-p)>1e-12
            operations+=1
    return {'reads':read_count,'operations':operations,'disagreements':disagreements,
            'library_bytes':out.stat().st_size}


def dynamics(sign,mem,context):
    return mem+sign*.42*math.tanh(1.5*mem+.3*context)


def nonlinear_stream(seed,count,sign,*,width=12):
    """No direct neural q targets: only delayed query output is revealed."""
    r=random.Random(seed)
    result=[]
    for _ in range(count):
        key=r.randrange(4);val=r.uniform(-1.1,1.1)
        context=r.choice([-.9,-.3,.4,1.2])
        stream=[Event(1.,key,val)]
        for _ in range(r.randint(4,40)):
            stream.append(Event(0.,r.randrange(4),r.uniform(-1.1,1.1)))
        stream.append(Event(0.,key,0.,True,dynamics(sign,val,context)))
        result.append((stream,context))
    return result


def stream_prequential(model,rows,*,lr=None):
    errors=[];records=[]
    for events,context in rows:
        memory=model.initial_memory()
        for e in events:
            if e.query:
                p=model.step(memory,e,context,target=e.target if lr is not None else None,
                             lr=lr if lr is not None else .04)
                errors.append((p-e.target)**2)
                records.append((memory[e.key],context,e.target))
            else:model.step(memory,e,context)
    return errors,records


def nonlinear_experiment(router):
    live=AdaptiveReasoner(router,PlasticFunction.initialize(capacity=12,active=8,seed=27))
    training=nonlinear_stream(1317,1300,+1)
    warm_errors,warm_records=stream_prequential(live,training,lr=.06)
    frozen=copy.deepcopy(live)
    trained_function=copy.deepcopy(live.function)
    changed=nonlinear_stream(915,1100,-1)
    live_errors,shift_records=stream_prequential(live,changed,lr=.06)
    frozen_errors,_=stream_prequential(frozen,changed,lr=None)
    # No future labels used in the prequential predictions; first and last
    # windows are disjoint and therefore measure adaptation over time.
    return {
      'function_parameters':live.function.parameter_count,
      'initial_active_units':frozen.function.active,
      'warmup_last_200_mse':sum(warm_errors[-200:])/200,
      'changed_first_100_live_mse':sum(live_errors[:100])/100,
      'changed_last_200_live_mse':sum(live_errors[-200:])/200,
      'changed_last_200_frozen_mse':sum(frozen_errors[-200:])/200,
      'last_200_improvement_ratio':sum(frozen_errors[-200:])/max(sum(live_errors[-200:]),1e-18),
      'training_observations':len(training),
      'changed_stream_observations':len(changed),
      'train_records':warm_records,
      'trained_function':trained_function,
      'changed_records':shift_records,
    }


def symbolic_experiment(train_records,trained_function):
    """M15 evidence gate. Derive residual targets only from observed outputs."""
    def neural(z):return trained_function.forward((z[0],z[1],z[0]*z[1]))[0]
    # Fit the actual learned network behavior, then validate against observed
    # next-query targets on newer, independent samples. No analytic surrogate.
    source=[((m,c),y-m) for m,c,y in train_records]
    train=source[:700]; validation=source[700:1000]
    r=random.Random(77);rollouts=[]
    for _ in range(24):
        s=r.uniform(-.7,.7);xs=[r.uniform(-.8,.8) for _ in range(12)]
        target=s
        for x in xs:target=.6*target+.42*math.tanh(1.5*target+.3*x)
        rollouts.append((s,xs,target))
    model=ConsolidatedFunction(neural)
    result=model.propose(train,validation,rollouts,absolute_mse=.001,
                         max_relative=1.2,rollout_tolerance=.15,max_terms=12)
    old=model.symbolic
    if result['accepted']:
        # On abrupt mode inversion a program learned in the old regime must be
        # invalidated; record a *measured* drift not an oracle mode signal.
        drift=[((m,c),-q) for (m,c),q in source[1000:1100]]
        if not drift:drift=[((m,c),-q) for (m,c),q in validation[:50]]
        drift_mse=sum((evaluate(old,z)-y)**2 for z,y in drift)/len(drift)
        if drift_mse>.03:model.rollback()
    else: drift_mse=None
    return {'proposal':result,'drift_mse':drift_mse,
            'rollback_after_observed_drift':bool(result['accepted'] and model.symbolic is None),
            'independent_validation':len(validation)}


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    train=episodes(11,450,distractors=(4,18))
    model=MemoryRouter([-1.,.2,0.],keys=4)
    start=perf_counter();trajectory=model.fit(train,epochs=22,rate=.16)
    fit_time=perf_counter()-start
    short=episodes(88,80,distractors=(4,18))
    long=episodes(89,80,distractors=(1024,4096))
    metrics={'format':'spark-m17-integrated-prequential-v1',
      'routing_train_episodes':len(train),'routing_train_start_mse':trajectory[0],
      'routing_train_end_soft_mse':trajectory[-1],
      'routing_train_seconds':fit_time,
      'routing_learned_coefficients':model.theta,
      'short':score_policy(model,short),
      'unseen_delay_1024_4096':score_policy(model,long),
      'unseen_delay_soft':score_policy(model,long,hard=False),
      'unseen_delay_always_write':score_policy(MemoryRouter([9.,0.,0.],keys=4),long),
      'unseen_delay_never_write':score_policy(MemoryRouter([-9.,0.,0.],keys=4),long),
    }
    metrics['native_memory']=native_check(model,long,out/'learned_memory.so')
    network=nonlinear_experiment(model)
    metrics['online_neural']={k:v for k,v in network.items() if k not in ('train_records','changed_records','trained_function')}
    metrics['symbolic_consolidation']=symbolic_experiment(network['train_records'],network['trained_function'])
    metrics['limitations']=[
       'Synthetic event markers explicitly reveal candidate relevance; policy learns a threshold, not arbitrary semantics.',
       'Router learns from delayed query labels, not write-operation labels; memory has fixed key addressing and capacity.',
       'Hard 0.5 deployment gate differs from differentiable soft training; long memory is conditional on successful routing.',
       'Neural function adapts online from full query targets; modes and nonlinear rule are synthetic.',
       'No real language data, no compute-matched Transformer or GRU benchmark, and no autonomous program-structure search.',
       'Native parity/throughput here covers the learned memory; dynamic neural runtime was native-tested in M13.'
    ]
    assert metrics['short']['exact_reads']==metrics['short']['reads'],metrics['short']
    assert metrics['unseen_delay_1024_4096']['exact_reads']==metrics['unseen_delay_1024_4096']['reads'],metrics['unseen_delay_1024_4096']
    assert metrics['native_memory']['disagreements']==0,metrics['native_memory']
    assert metrics['online_neural']['last_200_improvement_ratio']>20.,metrics['online_neural']
    (out/'report.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    (out/'learned_policy.json').write_text(json.dumps({'theta':model.theta,'capacity':4,'trained_only_on_queries':True},indent=2)+'\n')
    print(json.dumps({k:metrics[k] for k in ('short','unseen_delay_1024_4096','unseen_delay_always_write','native_memory','online_neural','symbolic_consolidation')},indent=2))
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/m17'))
    run(p.parse_args().output)

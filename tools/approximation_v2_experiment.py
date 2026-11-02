#!/usr/bin/env python3
"""Fresh-holdout study of calibration coverage, correlations and guarded repair."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spark.function_ir import digest
from spark.function_rewrite import read_program,readers_for,gated_function,parameter_array,prune_gated_function,calibration_program
from spark.model_compiler import write_package
from function_experiment_data import load_articles,prepare_split,hash_json
from function_simplification_experiment import worker,benchmark as shared_benchmark


def load(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def prepare(args,report):
    import numpy as np
    from spark.function_experiment import fit_candidates,correlation_prune,quality_passes
    protocol=args.protocol_data;source=read_program(args.source)
    if source['coefficients_sha256']!=protocol['source_coefficients_sha256']:raise ValueError('wrong baseline')
    work=args.output/'work';work.mkdir(exist_ok=True)
    write_package(source,readers_for(source,args.source),work/'baseline',tokenizer=args.source/'tokenizer',optimize_program=False)
    args.source=work/'baseline'
    prior=[r for split in ('train','validation','test') for r in load(args.prior/(split+'-windows.json'))]
    excluded=sorted({r['article'] for r in prior});prior_ids={r['id'] for r in prior}
    old_dataset=load(args.prior/'dataset.json')
    if old_dataset['revision']!=protocol['dataset_revision']:raise ValueError('prior study dataset revision differs')
    articles,_=load_articles(protocol,protocol['dataset_revision'],'train');all_ids={a['id'] for a in articles}
    dev_ids={identity for identity in all_ids if int(identity[:16],16)%5==0};calibration_ids=all_ids-dev_ids
    train_protocol=copy.deepcopy(protocol)
    train,train_meta=prepare_split(train_protocol,protocol['dataset_revision'],'train',args.source/'tokenizer',
             excluded_articles=set(excluded)|dev_ids,excluded_windows=prior_ids)
    development_protocol=copy.deepcopy(protocol);development_protocol['split_windows']['train']=protocol['split_windows']['validation'];development_protocol['seed']+=1
    dev,dev_meta=prepare_split(development_protocol,protocol['dataset_revision'],'train',args.source/'tokenizer',
             excluded_articles=set(excluded)|calibration_ids,excluded_windows=prior_ids|{r['id'] for r in train})
    write(args.output/'train-windows.json',train);write(args.output/'validation-windows.json',dev)
    write(args.output/'dataset.json',{'dataset':protocol['dataset'],'revision':protocol['dataset_revision'],'excluded_prior_article_ids':excluded,
          'prior_window_ids':sorted(prior_ids),'train':train_meta,'validation':dev_meta,'test_source_split':'validation'})
    name=protocol['function'];program=read_program(args.source);_,spec=gated_function(program,name)
    calibration_program(args.source,work/'calibration',name)
    worker(args,'collect-train','collect',work/'calibration',args.output/'train-windows.json',output=work/'train-features.npz')
    worker(args,'collect-validation','collect',work/'calibration',args.output/'validation-windows.json',output=work/'dev-features.npz')
    training=np.load(work/'train-features.npz');development=np.load(work/'dev-features.npz')
    h=training['features'];y=training['target'];hd=development['features'];yd=development['target']
    original_weight=parameter_array(program,args.source,spec['bindings']['output_weight']);original_bias=parameter_array(program,args.source,spec['bindings']['output_bias'])
    width=protocol['retained_width'];scale=max(float(np.sqrt(np.mean((yd-y.mean(axis=0))**2))),1e-12)
    methods={};coverage=[]
    for method in ('prefix','marginal_contribution'):
        if method=='prefix':
            keep=list(range(width));prefix_weight=np.zeros_like(original_weight);prefix_weight[:,keep]=original_weight[:,keep]
            # Zero importance on excluded units fixes this explicit control subset.
            ranked=fit_candidates(h,y,hd,yd,prefix_weight,[width],protocol['ridge_factors'])[0]
            order=[ranked['keep'].index(i) for i in keep]
            ranked['weight']=ranked['weight'][:,order]
        else:ranked=fit_candidates(h,y,hd,yd,original_weight,[width],protocol['ridge_factors'])[0];keep=ranked['keep']
        methods[method]={'keep':keep,'weight':ranked['weight'],'bias':ranked['bias'],'dev_nrmse':ranked['dev_nrmse'],'ridge_factor':ranked['ridge_factor']}
    for count,label in ((min(protocol['coverage_control_samples'],len(h)),'small'),(len(h),'full')):
        options=[]
        for ridge in protocol['ridge_factors']:
            fitted=correlation_prune(h[:count],y[:count],original_weight,width,ridge)
            residual=hd[:,fitted['keep']]@fitted['weight'].astype(float).T+fitted['bias'].astype(float)-yd
            fitted['dev_nrmse']=float(np.sqrt(np.mean(residual**2))/scale);options.append(fitted)
        best=min(options,key=lambda row:row['dev_nrmse'])
        coverage.append({'samples':count,'condition':label,'best_dev_nrmse':best['dev_nrmse'],'ridge_factor':best['ridge_factor'],
                         'trials':[{k:v for k,v in row.items() if k in ('ridge_factor','dev_nrmse')} for row in options]})
        if label=='full':methods['conditional_covariance']=best
    base_quality=worker(args,'development-baseline','quality',args.source,args.output/'validation-windows.json',logits=work/'baseline-development.npy',bootstrap=protocol['performance']['bootstrap_samples'])
    choices=[];method_reports={};template=None
    for method,fitted in methods.items():
        directory=work/method
        prune_gated_function(args.source,directory,name,fitted['keep'],native_template=template)
        template=directory;solutions=[]
        retained=original_weight[:,fitted['keep']]
        for alpha in protocol['blend_factors']:
            weight=((1-alpha)*retained.astype(float)+alpha*fitted['weight'].astype(float)).astype(np.float32)
            bias=((1-alpha)*original_bias.astype(float)+alpha*fitted['bias'].astype(float)).astype(np.float32)
            path=work/f'{method}-alpha{alpha}.npz';np.savez(path,function=np.asarray(name),weight=weight,bias=bias,keep=np.asarray(fitted['keep']))
            solutions.append({'path':str(path),'label':method,'alpha':alpha})
        solution_list=work/(method+'-solutions.json');write(solution_list,solutions)
        observed=worker(args,'development-'+method,'quality',directory,args.output/'validation-windows.json',reference=work/'baseline-development.npy',
                        solutions_json=solution_list,bootstrap=protocol['performance']['bootstrap_samples'])
        method_reports[method]={'function_dev_nrmse':fitted['dev_nrmse'],'ridge_factor':fitted['ridge_factor'],
                                'before_recovery':observed['before_recovery'],'variants':observed['variants']}
        for variant,solution in zip(observed['variants'],solutions):
            choices.append({'method':method,'alpha':variant['alpha'],'quality':variant['quality'],'solution':solution['path'],
                            'keep':fitted['keep'],'ridge_factor':fitted['ridge_factor']})
    def key(row):
        q=row['quality'];return (quality_passes(q,protocol['quality_gates']),-q['mean_teacher_kl_nats'],q['gold_accuracy'],-row['alpha'])
    selected=max(choices,key=key);shutil.copyfile(selected['solution'],args.output/'solution.npz')
    chosen=np.load(args.output/'solution.npz')
    candidate=prune_gated_function(args.source,args.output/'candidate',name,selected['keep'],readout_weight=chosen['weight'],readout_bias=chosen['bias'],native_template=work/selected['method'])
    # Keep common worker paths explicit for later measurements.
    selection={k:v for k,v in selected.items() if k!='solution'}
    selection.update(protocol_sha256=digest(args.protocol),solution_sha256=digest(args.output/'solution.npz'),
                     development_qualified=quality_passes(selected['quality'],protocol['quality_gates']))
    write(args.output/'selection.json',selection)
    write(args.output/'model-performance-inputs.json',dev[0]['inputs'])
    write(args.output/'function-performance-inputs.json',{'x':development['arguments'][:protocol['context_tokens']].tolist()})
    report.update(selected_method=selected['method'],selected_alpha=selected['alpha'],selected_width=width,
                  calibration_samples=len(h),development_samples=len(hd),coverage_analysis=coverage,method_comparison=method_reports,
                  development_quality=selected['quality'],development_qualified=selection['development_qualified'],
                  selection_sha256=digest(args.output/'selection.json'),original_function_capacity=spec['capacity'],
                  baseline_payload_bytes=program['coefficient_bytes'],candidate_payload_bytes=candidate['coefficient_bytes'],
                  payload_saved_bytes=program['coefficient_bytes']-candidate['coefficient_bytes'],test_data_accessed=False)


def locked(args):
    selected=load(args.output/'selection.json');prepared=load(args.output/'prepare-report.json')
    if digest(args.protocol)!=selected['protocol_sha256'] or digest(args.output/'solution.npz')!=selected['solution_sha256'] or digest(args.output/'selection.json')!=prepared['selection_sha256']:
        raise ValueError('selection changed after validation')
    return selected


def evaluate(args,report):
    from spark.function_experiment import quality_passes
    selected=locked(args);protocol=args.protocol_data;data=load(args.output/'dataset.json');work=args.output/'work'
    test_protocol=copy.deepcopy(protocol);test_protocol['split_windows']['validation']=protocol['split_windows']['test'];test_protocol['seed']+=2
    train=load(args.output/'train-windows.json');dev=load(args.output/'validation-windows.json')
    excluded=set(data['excluded_prior_article_ids'])|{r['article'] for r in train+dev}
    test,meta=prepare_split(test_protocol,protocol['dataset_revision'],'validation',work/'baseline/tokenizer',
                           excluded_articles=excluded,excluded_windows=set(data['prior_window_ids'])|{r['id'] for r in train+dev})
    if {r['article'] for r in test}&excluded:raise AssertionError('fresh holdout reused prior articles')
    write(args.output/'test-windows.json',test);data['test']=meta;write(args.output/'dataset.json',data)
    baseline=worker(args,'test-baseline','quality',work/'baseline',args.output/'test-windows.json',logits=work/'baseline-test.npy',bootstrap=protocol['performance']['bootstrap_samples'])
    prefix=worker(args,'test-prefix','quality',work/'prefix',args.output/'test-windows.json',reference=work/'baseline-test.npy',bootstrap=protocol['performance']['bootstrap_samples'])
    observed=worker(args,'test-recovery','quality',work/selected['method'],args.output/'test-windows.json',reference=work/'baseline-test.npy',
                    solution=args.output/'solution.npz',persisted=args.output/'candidate',bootstrap=protocol['performance']['bootstrap_samples'])
    report.update(selected_method=selected['method'],selected_alpha=selected['alpha'],baseline_quality=baseline['quality'],prefix_quality=prefix['quality'],
                  unrecovered_quality=observed['before_recovery'],recovered_quality=observed['quality'],live_recovery=observed['live_update'],
                  heldout_quality_gates_passed=quality_passes(observed['quality'],protocol['quality_gates']),
                  fresh_articles=len({r['article'] for r in test}),test_queries=sum(len(r['labels']) for r in test),
                  prior_article_overlap=0,selection_sha256=digest(args.output/'selection.json'))


def benchmark(args,report):
    # Reuse the same validated independent-process benchmark implementation.
    from spark.function_rewrite import extract_function
    from spark.function_experiment import paired_speedup
    import numpy as np
    selected=locked(args);work=args.output/'work';settings=args.protocol_data['performance'];rng=np.random.default_rng(args.protocol_data['seed'])
    extract_function(work/'baseline',work/'baseline-function',args.protocol_data['function'])
    extract_function(args.output/'candidate',work/'candidate-function',args.protocol_data['function']);report['results']={}
    for scope in ('function','model'):
        paths={'baseline':work/'baseline-function' if scope=='function' else work/'baseline',
               'candidate':work/'candidate-function' if scope=='function' else args.output/'candidate'}
        observations={k:[] for k in paths}
        for index in range(settings['paired_rounds']):
            order=list(paths);rng.shuffle(order)
            for name in order:
                result=worker(args,f'benchmark-{scope}-{index}-{name}','benchmark',paths[name],args.output/(scope+'-performance-inputs.json'),
                              repetitions=settings['function_repetitions'] if scope=='function' else settings['full_model_repetitions'],warmup=settings['warmup_repetitions'])
                observations[name].append(result)
        timings={k:[r['median_native_milliseconds'] for r in rows] for k,rows in observations.items()}
        statistics=paired_speedup(timings['baseline'],timings['candidate'],samples=settings['bootstrap_samples'])
        statistics.update(baseline_median_ms=float(np.median(timings['baseline'])),candidate_median_ms=float(np.median(timings['candidate'])),observations=observations)
        report['results'][scope]=statistics


def summary(args,report):
    prepared=load(args.output/'prepare-report.json');tested=load(args.output/'evaluate-report.json');timed=load(args.output/'benchmark-report.json')
    report.update(development=prepared,evaluation=tested,performance=timed,
                  conclusion={'development_qualified':prepared['development_qualified'],'fresh_holdout_gates_passed':tested['heldout_quality_gates_passed'],
                  'prior_article_overlap':tested['prior_article_overlap'],'local_latency_gain_demonstrated':timed['results']['function']['speedup_demonstrated'],
                  'whole_model_latency_gain_demonstrated':timed['results']['model']['speedup_demonstrated'],
                  'live_recovery_improved_teacher_kl':tested['recovered_quality']['mean_teacher_kl_nats']<tested['unrecovered_quality']['mean_teacher_kl_nats'],
                  'payload_saved_bytes':prepared['payload_saved_bytes']})
    report['conclusion']['quality_preserving_local_advantage']=prepared['development_qualified'] and tested['heldout_quality_gates_passed'] and timed['results']['function']['speedup_demonstrated']
    with (args.output/'candidate/README.txt').open('a') as stream:
        stream.write('\nEXPERIMENTAL APPROXIMATE REWRITE. See experiment.json for fresh-holdout gates and measured results.\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','evaluate','benchmark','summary'),required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'experiments/approximation_v2.json');parser.add_argument('--output',type=Path,default=ROOT/'artifacts/approximation-v2')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True);args.protocol_data=load(args.protocol)
    os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';shutil.copyfile(args.protocol,args.output/'protocol.json')
    report={'status':'running','phase':args.phase,'protocol_sha256':digest(args.protocol)};started=time.perf_counter()
    try:globals()[args.phase](args,report);report['status']='completed'
    except Exception as error:report.update(status='failed',error=re.sub(r'https?://\S+','<url>',str(error))[-4000:]);raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;write(args.output/(args.phase+'-report.json'),report)
        if args.phase=='summary' and (args.output/'candidate').exists():write(args.output/'candidate/experiment.json',report)


if __name__=='__main__':main()

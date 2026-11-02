#!/usr/bin/env python3
"""Predeclared single-function simplification, live recovery and paired benchmark."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spark.function_ir import compile_program,digest
from spark.function_rewrite import calibration_program,extract_function,gated_function,parameter_array,prune_gated_function,read_program
from function_experiment_data import hash_json,prepare_split,resolve_dataset


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(path):return json.loads(Path(path).read_text())


def worker(args,label,mode,program,inputs,*,output=None,**options):
    destination=output or args.output/(label+'.json')
    command=[sys.executable,str(ROOT/'tools/function_experiment_worker.py'),mode,'--program',str(program),
             '--inputs',str(inputs),'--output',str(destination),'--threads',str(args.protocol_data['performance']['threads'])]
    for key,value in options.items():
        if value is not None:command.extend(['--'+key.replace('_','-'),str(value)])
    logs=args.output/'logs';logs.mkdir(exist_ok=True)
    started=time.perf_counter();print('Starting '+label,flush=True)
    with (logs/(label+'.log')).open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        tail=(logs/(label+'.log')).read_text(errors='replace')[-4000:]
        raise RuntimeError('worker '+label+' failed: '+re.sub(r'https?://\S+','<url>',tail))
    print(f'Completed {label} in {time.perf_counter()-started:.2f}s',flush=True)
    return load(destination) if mode!='collect' else destination


def assert_source(args):
    program=read_program(args.source)
    if program['coefficients_sha256']!=args.protocol_data['source_coefficients_sha256']:
        raise ValueError('baseline is not the predeclared compiled checkpoint')
    if digest(args.source/'runtime/function_ir_runtime.hpp')!=digest(ROOT/'runtime/function_ir_runtime.hpp'):
        raise ValueError('baseline and transformed programs must use identical numerical kernels')
    return program


def prepare(args,report):
    import numpy as np
    from spark.function_experiment import fit_candidates,quality_passes
    protocol=args.protocol_data;program=assert_source(args);name=protocol['function']
    function,spec=gated_function(program,name)
    shutil.copyfile(args.protocol,args.output/'protocol.json')
    revision=resolve_dataset(protocol)
    data={'dataset':protocol['dataset'],'config':protocol['dataset_config'],'revision':revision,
          'protocol_sha256':digest(args.protocol),'source_coefficients_sha256':program['coefficients_sha256']}
    write(args.output/'dataset.json',data)
    report['stage']='prepare-train-and-development'
    train,train_meta=prepare_split(protocol,revision,'train',args.source/'tokenizer')
    dev,dev_meta=prepare_split(protocol,revision,'validation',args.source/'tokenizer',
                               excluded_articles=train_meta['all_article_ids'],excluded_windows=[c['id'] for c in train])
    write(args.output/'train-windows.json',train);write(args.output/'validation-windows.json',dev)
    data.update(train=train_meta,validation=dev_meta);write(args.output/'dataset.json',data)
    work=args.output/'work';work.mkdir(exist_ok=True)
    report['stage']='collect-function-calibration'
    calibration_program(args.source,work/'calibration',name)
    worker(args,'collect-train','collect',work/'calibration',args.output/'train-windows.json',output=work/'train-features.npz')
    worker(args,'collect-validation','collect',work/'calibration',args.output/'validation-windows.json',output=work/'dev-features.npz')
    training=np.load(work/'train-features.npz');development=np.load(work/'dev-features.npz')
    weight=parameter_array(program,args.source,spec['bindings']['output_weight'])
    report['stage']='fit-and-select-on-development'
    candidates=fit_candidates(training['features'],training['target'],development['features'],development['target'],
                              weight,protocol['candidate_widths'],protocol['ridge_factors'])
    eligible=[c for c in candidates if c['dev_nrmse']<=protocol['development_function_nrmse_limit']]
    selected=min(eligible,key=lambda c:c['width']) if eligible else max(candidates,key=lambda c:c['width'])
    solution=args.output/'solution.npz'
    np.savez(solution,function=np.asarray(name),weight=selected['weight'],bias=selected['bias'],keep=np.asarray(selected['keep']))
    selection={k:v for k,v in selected.items() if k not in ('weight','bias')}
    selection.update(protocol_sha256=digest(args.protocol),solution_sha256=digest(solution),
                     function_reconstruction_gate_passed=bool(eligible),
                     candidates=[{k:v for k,v in c.items() if k not in ('weight','bias','keep')} for c in candidates],
                     trained_on='official training articles only',selected_on='official validation articles only',
                     train_samples=int(len(training['features'])),validation_samples=int(len(development['features'])),
                     selected_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    # Store/hash the model decision before even downloading or parsing test data.
    write(args.output/'selection.json',selection)
    report.update(selected_width=selected['width'],development_function_nrmse=selected['dev_nrmse'])
    report['stage']='materialize-approximate-targets'
    prefix=work/'prefix';pruned=work/'pruned';recovered=args.output/'recovered-program'
    prune_gated_function(args.source,prefix,name,list(range(selected['width'])))
    prune_gated_function(args.source,pruned,name,selected['keep'],native_template=prefix)
    target=prune_gated_function(args.source,recovered,name,selected['keep'],readout_weight=selected['weight'],readout_bias=selected['bias'],native_template=pruned)
    compile_program(args.source)  # Normalize deterministic cache once before workers.
    report['stage']='development-full-model-quality'
    base=worker(args,'development-baseline','quality',args.source,args.output/'validation-windows.json',logits=work/'baseline-development.npy',bootstrap=protocol['performance']['bootstrap_samples'])
    prefix_quality=worker(args,'development-prefix','quality',prefix,args.output/'validation-windows.json',reference=work/'baseline-development.npy',bootstrap=protocol['performance']['bootstrap_samples'])
    repair=worker(args,'development-recovery','quality',pruned,args.output/'validation-windows.json',reference=work/'baseline-development.npy',
                  solution=solution,persisted=recovered,bootstrap=protocol['performance']['bootstrap_samples'])
    selection['development_quality_gate_passed']=quality_passes(repair['quality'],protocol['quality_gates'])
    selection['development_qualified']=selection['function_reconstruction_gate_passed'] and selection['development_quality_gate_passed']
    selection['development_quality']=repair['quality']
    write(args.output/'selection.json',selection)
    write(args.output/'model-performance-inputs.json',dev[0]['inputs'])
    write(args.output/'function-performance-inputs.json',{'x':development['arguments'][:protocol['context_tokens']].tolist()})
    report.update(development_qualified=selection['development_qualified'],development_quality=repair['quality'],
                  development_prefix_quality=prefix_quality['quality'],development_unrecovered_quality=repair['before_recovery'],
                  original_function_capacity=spec['capacity'],function=name,baseline_payload_bytes=program['coefficient_bytes'],
                  transformed_payload_bytes=target['coefficient_bytes'],payload_saved_bytes=program['coefficient_bytes']-target['coefficient_bytes'],
                  selected_solution_sha256=selection['solution_sha256'],selection_sha256=digest(args.output/'selection.json'),
                  test_data_accessed=False,development_live_recovery=repair['live_update'])


def locked_selection(args):
    selection=load(args.output/'selection.json')
    if digest(args.protocol)!=selection['protocol_sha256'] or digest(args.output/'solution.npz')!=selection['solution_sha256']:
        raise ValueError('predeclared protocol or selected model changed before test')
    prepared=load(args.output/'prepare-report.json')
    if prepared['status']!='completed' or digest(args.output/'selection.json')!=prepared['selection_sha256']:
        raise ValueError('selected model is not the completed development decision')
    return selection


def evaluate(args,report):
    from spark.function_experiment import quality_passes
    selection=locked_selection(args);assert_source(args);protocol=args.protocol_data;data=load(args.output/'dataset.json')
    # The workflow commits the development receipt before starting this phase.
    report.update(selection_sha256=digest(args.output/'selection.json'),solution_sha256=selection['solution_sha256'],
                  selection_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  test_first_access_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='open-sealed-test-split')
    train=load(args.output/'train-windows.json');dev=load(args.output/'validation-windows.json')
    excluded=data['train']['all_article_ids']+data['validation']['all_article_ids']
    test,test_meta=prepare_split(protocol,data['revision'],'test',args.source/'tokenizer',excluded_articles=excluded,
                                 excluded_windows=[c['id'] for c in train+dev])
    write(args.output/'test-windows.json',test);data['test']=test_meta;write(args.output/'dataset.json',data)
    split_articles=[{c['article'] for c in rows} for rows in (train,dev,test)]
    if any(split_articles[i]&split_articles[j] for i in range(3) for j in range(i+1,3)):
        raise AssertionError('article leakage across splits')
    work=args.output/'work';report['stage']='heldout-full-model-quality'
    baseline=worker(args,'test-baseline','quality',args.source,args.output/'test-windows.json',logits=work/'baseline-test.npy',bootstrap=protocol['performance']['bootstrap_samples'])
    prefix=worker(args,'test-prefix','quality',work/'prefix',args.output/'test-windows.json',reference=work/'baseline-test.npy',bootstrap=protocol['performance']['bootstrap_samples'])
    repair=worker(args,'test-recovery','quality',work/'pruned',args.output/'test-windows.json',reference=work/'baseline-test.npy',solution=args.output/'solution.npz',
                  persisted=args.output/'recovered-program',bootstrap=protocol['performance']['bootstrap_samples'])
    report.update(dataset_revision=data['revision'],test_windows=len(test),test_queries=sum(len(c['labels']) for c in test),
                  split_audit={'article_sets_disjoint':True,'selected_window_sets_disjoint':True,'selection_locked_before_test':True},
                  baseline_quality=baseline['quality'],prefix_quality=prefix['quality'],unrecovered_quality=repair['before_recovery'],
                  recovered_quality=repair['quality'],live_recovery=repair['live_update'],
                  heldout_quality_gates_passed=quality_passes(repair['quality'],protocol['quality_gates']),
                  recovery_reduced_teacher_kl=repair['quality']['mean_teacher_kl_nats']<repair['before_recovery']['mean_teacher_kl_nats'],
                  recovery_gold_accuracy_delta=repair['quality']['gold_accuracy']-repair['before_recovery']['gold_accuracy'])


def benchmark(args,report):
    import numpy as np
    from spark.function_experiment import paired_speedup
    selection=locked_selection(args);protocol=args.protocol_data;settings=protocol['performance'];work=args.output/'work'
    report['stage']='compile-isolated-function-benchmarks'
    extract_function(args.source,work/'baseline-function',protocol['function'])
    extract_function(args.output/'recovered-program',work/'recovered-function',protocol['function'])
    report.update(machine={'platform':platform.platform(),'cpu_count':os.cpu_count(),'python':sys.version.split()[0],
                           'threads':settings['threads'],'numpy':np.__version__},results={})
    rng=np.random.default_rng(protocol['seed'])
    for scope in ('function','model'):
        report['stage']='paired-'+scope+'-benchmarks'
        paths={'baseline':work/'baseline-function' if scope=='function' else args.source,
               'candidate':work/'recovered-function' if scope=='function' else args.output/'recovered-program'}
        observations={'baseline':[],'candidate':[]};rounds=[]
        inputs=args.output/(scope+'-performance-inputs.json')
        for index in range(settings['paired_rounds']):
            order=list(paths);rng.shuffle(order);rounds.append(order)
            for variant in order:
                result=worker(args,f'benchmark-{scope}-{index}-{variant}','benchmark',paths[variant],inputs,
                              repetitions=settings['function_repetitions'] if scope=='function' else settings['full_model_repetitions'],
                              warmup=settings['warmup_repetitions'])
                observations[variant].append(result)
        timings={k:[r['median_native_milliseconds'] for r in v] for k,v in observations.items()}
        stats=paired_speedup(timings['baseline'],timings['candidate'],seed=protocol['seed'],samples=settings['bootstrap_samples'])
        stats.update(baseline_median_ms=float(np.median(timings['baseline'])),candidate_median_ms=float(np.median(timings['candidate'])),
                     baseline_peak_rss_median_bytes=float(np.median([r['peak_rss_bytes'] for r in observations['baseline']])),
                     candidate_peak_rss_median_bytes=float(np.median([r['peak_rss_bytes'] for r in observations['candidate']])),
                     random_pair_orders=rounds,observations=observations,inputs_sha256=digest(inputs))
        for left,right in zip(observations['baseline'],observations['candidate']):
            if left['input_shapes']!=right['input_shapes'] or left['output_shapes']!=right['output_shapes']:
                raise AssertionError('performance comparison has mismatched work')
        if scope=='function':
            original=read_program(paths['baseline']);modified=read_program(paths['candidate'])
            stats.update(baseline_capture_bytes=original['coefficient_bytes'],candidate_capture_bytes=modified['coefficient_bytes'],
                         approximate_dense_multiply_ratio=selection['width']/load(args.output/'prepare-report.json')['original_function_capacity'])
        report['results'][scope]=stats
        write(args.output/'benchmark-partial.json',report)
    report['latency_scope']='compiled C ABI, same inputs/outputs, warmed execution; compilation/loading/tokenization/Python marshalling excluded'


def summarize(args,report):
    prepared=load(args.output/'prepare-report.json');evaluation=load(args.output/'evaluate-report.json');performance=load(args.output/'benchmark-report.json')
    if any(r['status']!='completed' for r in (prepared,evaluation,performance)):
        raise ValueError('experiment phases incomplete')
    report.update(protocol=load(args.output/'protocol.json'),dataset=load(args.output/'dataset.json'),
                  selection=load(args.output/'selection.json'),development=prepared,evaluation=evaluation,performance=performance,
                  conclusion={'development_qualified':prepared['development_qualified'],
                              'heldout_quality_gates_passed':evaluation['heldout_quality_gates_passed'],
                              'learned_recovery_reduced_teacher_kl':evaluation['recovery_reduced_teacher_kl'],
                              'function_latency_gain_demonstrated':performance['results']['function']['speedup_demonstrated'],
                              'whole_model_latency_gain_demonstrated':performance['results']['model']['speedup_demonstrated'],
                              'model_payload_reduction_bytes':prepared['payload_saved_bytes'],
                              'original_model_unchanged':digest(args.source/'coefficients.bin')==args.protocol_data['source_coefficients_sha256']},
                  caveats=['Exploratory single-function, single-model experiment on one runner.',
                           'Held out from optimization; source pretraining overlap with WikiText is unknown.',
                           'Quality gates use point estimates; article-cluster confidence intervals are also reported.',
                           'Pruning and readout reconstruction are available in conventional frameworks too.',
                           'Approximate rewrite does not change the exact compiler mode or establish logical proof validity.'])
    report['conclusion']['local_advantage_demonstrated']=bool(prepared['development_qualified'] and evaluation['heldout_quality_gates_passed'] and
              evaluation['recovery_reduced_teacher_kl'] and performance['results']['function']['speedup_demonstrated'] and prepared['payload_saved_bytes']>0)
    target=read_program(args.output/'recovered-program')
    report['target']={k:target[k] for k in ('format','coefficients_sha256','coefficient_bytes')}
    (args.output/'recovered-program/experiment.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output/'recovered-program/README.txt').open('a') as stream:
        stream.write('\nEXPERIMENTAL APPROXIMATE FUNCTION REWRITE. Exact source-model equivalence is not claimed.\nSee experiment.json for held-out quality, performance and limitations.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','evaluate','benchmark','summarize'),required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--protocol',type=Path,default=ROOT/'experiments/function_simplification_v1.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/function-simplification')
    args=parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    args.protocol_data=load(args.protocol);settings=args.protocol_data['performance']
    os.environ['OPENBLAS_NUM_THREADS']=str(settings['threads']);os.environ['OMP_NUM_THREADS']=str(settings['threads'])
    report={'status':'running','stage':'initialize','phase':args.phase,'protocol_sha256':digest(args.protocol)};started=time.perf_counter()
    try:
        globals()[args.phase](args,report);report.update(status='completed',stage='complete')
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=re.sub(r'https?://\S+','<url>',str(error))[-5000:]);raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;write(args.output/(args.phase+'-report.json'),report)
        if args.phase=='summarize' and (args.output/'recovered-program').is_dir():
            write(args.output/'recovered-program/experiment.json',report)


if __name__=='__main__':main()

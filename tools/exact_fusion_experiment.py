#!/usr/bin/env python3
"""Build, audit, validate and benchmark an exact whole-model fusion pass."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spark.function_ir import digest
from spark.function_rewrite import read_program,readers_for
from spark.model_compiler import write_package
from spark.function_fusion import compile_fused_program,verify_fusion


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def load(path):return json.loads(Path(path).read_text())


def cases(args):
    import numpy as np
    source=load(args.regressions/'test-windows.json')[:args.protocol_data['regression_windows']]
    result=[{'name':'regression-'+str(i),'inputs':case['inputs']} for i,case in enumerate(source)]
    body=[]
    for case in load(args.regressions/'validation-windows.json'):
        ids=list(case['inputs']['tokens'])
        for position,label in zip(case['inputs']['queries'],case['labels']):ids[position]=label
        body.extend(ids[1:-1])
    spec=load(args.source/'tokenizer/spark_tokenizer.json')
    rng=np.random.default_rng(args.protocol_data['seed'])
    for length in args.protocol_data['long_context_lengths']:
        ids=[spec['cls_id'],*(body*((length//len(body))+1))[:length-2],spec['sep_id']]
        queries=sorted(int(i) for i in rng.choice(range(1,length-1),size=8,replace=False))
        for position in queries:ids[position]=spec['mask_id']
        result.append({'name':'length-'+str(length),'inputs':{'tokens':ids,'mask':[1]*length,'positions':list(range(length)),'queries':queries}})
    base=copy.deepcopy(source[0]['inputs']);n=len(base['tokens'])
    edge=copy.deepcopy(base);edge['positions']=list(range(8192-n,8192));result.append({'name':'rotary-edge','inputs':edge})
    holes=copy.deepcopy(base);holes['mask']=[int(i%7!=0 or i in base['queries']) for i in range(n)];result.append({'name':'attention-holes','inputs':holes})
    for side in ('left','right'):
        pad=[spec['pad_id']]*9;left=side=='left';padded=copy.deepcopy(base)
        padded['tokens']=pad+base['tokens'] if left else base['tokens']+pad
        padded['mask']=[0]*9+[1]*n if left else [1]*n+[0]*9
        padded['positions']=list(range(n+9));padded['queries']=[q+(9 if left else 0) for q in base['queries']]
        result.append({'name':side+'-padding','inputs':padded})
    return result


def build(args,report):
    source=read_program(args.source)
    if source['coefficients_sha256']!=args.protocol_data['source_coefficients_sha256']:raise ValueError('wrong unchanged baseline')
    shutil.copyfile(args.protocol,args.output/'protocol.json')
    tokenizer=args.source/'tokenizer'
    baseline=write_package(source,readers_for(source,args.source),args.output/'baseline',tokenizer=tokenizer,optimize_program=False)
    target,audit=compile_fused_program(args.output/'baseline',args.output/'candidate')
    write(args.output/'audit.json',audit)
    checked=verify_fusion(baseline,target,audit)
    all_cases=cases(args);write(args.output/'cases.json',all_cases)
    for length in (128,*args.protocol_data['long_context_lengths']):
        case=all_cases[0] if length==128 else next(c for c in all_cases if c['name']=='length-'+str(length))
        write(args.output/f'performance-{length}.json',case['inputs'])
    count=lambda p:sum(len(f['nodes']) for f in [*p['functions'],p['main']])
    report.update(rewrites=len(audit['records']),rule_counts=audit['rule_counts'],audit=checked,
                  baseline_operations=count(baseline),candidate_operations=count(target),coefficients_unchanged=baseline['coefficients_sha256']==target['coefficients_sha256'],
                  coefficient_bytes=target['coefficient_bytes'],regression_cases=len(all_cases),
                  kernel_sha256=digest(args.output/'baseline/runtime/function_ir_runtime.hpp'))
    if digest(args.output/'candidate/runtime/function_ir_runtime.hpp')!=report['kernel_sha256']:raise AssertionError('kernels differ between conditions')


def equivalence(args,report):
    import numpy as np
    from spark.function_runtime import NativeFunctionProgram
    rows=load(args.output/'cases.json');comparisons=[];total=0;worst=0.
    with NativeFunctionProgram(args.output/'baseline') as baseline,NativeFunctionProgram(args.output/'candidate') as candidate:
        def compare(name,inputs):
            nonlocal total,worst
            a=baseline.run(inputs);b=candidate.run(inputs);equal=True;error=0.;values=0
            for key in a:
                x=np.ctypeslib.as_array(a[key]['values']);y=np.ctypeslib.as_array(b[key]['values'])
                equal=equal and a[key]['shape']==b[key]['shape'] and x.tobytes()==y.tobytes()
                error=max(error,float(np.max(np.abs(x-y))));values+=len(x)
            comparisons.append({'case':name,'sequence_length':len(inputs['tokens']),'bitwise_equal':bool(equal),'max_abs_error':error,'values':values})
            total+=values;worst=max(worst,error)
            write(args.output/'equivalence-partial.json',{'comparisons':comparisons})
            if not equal:raise AssertionError('bitwise fusion mismatch in '+name)
        for row in rows:compare(row['name'],row['inputs'])
        probe=rows[0]['inputs'];program=baseline.program;parameters={p['id']:p for p in program['parameters']}
        for name in ('function0','function11','function21'):
            for width in (1,576,1152):
                baseline.set_active_width(name,width);candidate.set_active_width(name,width);compare(name+'-width-'+str(width),probe)
        name='function11'
        for binding in ('input_weight','input_bias','output_weight','output_bias'):
            pid=baseline.functions[name][1]['live']['bindings'][binding];p=parameters[pid]
            with (args.output/'baseline/coefficients.bin').open('rb') as stream:
                stream.seek(p['offset']);original=__import__('struct').unpack('<f',stream.read(4))[0]
            baseline.set_function_coefficient(name,binding,0,original+.125);candidate.set_function_coefficient(name,binding,0,original+.125)
            compare('updated-'+binding,probe)
            baseline.set_function_coefficient(name,binding,0,original);candidate.set_function_coefficient(name,binding,0,original)
            compare('restored-'+binding,probe)
    report.update(bitwise_equivalent=True,comparisons=comparisons,values_checked=total,max_abs_error=worst,
                  original_coefficient_file_unchanged=digest(args.output/'baseline/coefficients.bin')==args.protocol_data['source_coefficients_sha256'],
                  candidate_coefficient_file_unchanged=digest(args.output/'candidate/coefficients.bin')==args.protocol_data['source_coefficients_sha256'])


def benchmark(args,report):
    import numpy as np
    from spark.function_experiment import paired_speedup
    rng=np.random.default_rng(args.protocol_data['seed']);report['results']={}
    logs=args.output/'logs';logs.mkdir(exist_ok=True)
    for length,config in args.protocol_data['performance'].items():
        observed={'baseline':[],'candidate':[]};orders=[]
        for index in range(config['paired_rounds']):
            order=list(observed);rng.shuffle(order);orders.append(order)
            for condition in order:
                output=args.output/f'timing-{length}-{index}-{condition}.json'
                command=[sys.executable,str(ROOT/'tools/function_experiment_worker.py'),'benchmark','--program',str(args.output/condition),
                         '--inputs',str(args.output/f'performance-{length}.json'),'--output',str(output),'--repetitions',str(config['repetitions']),
                         '--warmup',str(config['warmup']),'--threads',str(args.protocol_data['threads'])]
                with (logs/(output.stem+'.log')).open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
                observed[condition].append(load(output))
        times={k:[v['median_native_milliseconds'] for v in values] for k,values in observed.items()}
        statistics=paired_speedup(times['baseline'],times['candidate'],seed=args.protocol_data['seed'],samples=args.protocol_data['bootstrap_samples'])
        statistics.update(baseline_median_ms=float(np.median(times['baseline'])),candidate_median_ms=float(np.median(times['candidate'])),
                          baseline_peak_rss_median_bytes=float(np.median([r['peak_rss_bytes'] for r in observed['baseline']])),
                          candidate_peak_rss_median_bytes=float(np.median([r['peak_rss_bytes'] for r in observed['candidate']])),observations=observed,orders=orders)
        for a,b in zip(observed['baseline'],observed['candidate']):
            if a['input_shapes']!=b['input_shapes'] or a['output_shapes']!=b['output_shapes']:raise AssertionError('benchmark work differs')
        report['results'][length]=statistics;write(args.output/'benchmark-partial.json',report)


def summary(args,report):
    built=load(args.output/'build-report.json');checked=load(args.output/'equivalence-report.json');timed=load(args.output/'benchmark-report.json')
    report.update(build=built,equivalence=checked,performance=timed,
        conclusion={'bitwise_equivalent':checked['bitwise_equivalent'],'audit_verified':built['audit']['verified'],
                    'coefficients_unchanged':built['coefficients_unchanged'],
                    'whole_model_gain_demonstrated':any(v['speedup_demonstrated'] for v in timed['results'].values())})
    report['conclusion']['adoption_gate_passed']=all(report['conclusion'].values())
    write(args.output/'candidate/validation.json',report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('build','equivalence','benchmark','summary'),required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--regressions',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'experiments/exact_fusion_v1.json');parser.add_argument('--output',type=Path,default=ROOT/'artifacts/exact-fusion')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True);args.protocol_data=load(args.protocol)
    os.environ['OPENBLAS_NUM_THREADS']=str(args.protocol_data['threads']);os.environ['OMP_NUM_THREADS']=str(args.protocol_data['threads'])
    report={'status':'running','phase':args.phase,'protocol_sha256':digest(args.protocol)};started=time.perf_counter()
    try:globals()[args.phase](args,report);report['status']='completed'
    except Exception as error:report.update(status='failed',error=re.sub(r'https?://\S+','<url>',str(error))[-4000:]);raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;write(args.output/(args.phase+'-report.json'),report)
        if args.phase=='summary' and (args.output/'candidate').exists():write(args.output/'candidate/validation.json',report)


if __name__=='__main__':main()

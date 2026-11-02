#!/usr/bin/env python3
"""Isolated collection, quality and performance workers for the fixed experiment."""
from __future__ import annotations
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def matrix(result):
    import numpy as np
    return np.ctypeslib.as_array(result['values']).reshape(result['shape'])


def collect(args):
    import numpy as np
    from spark.function_runtime import NativeFunctionProgram
    windows = json.loads(args.inputs.read_text())
    arrays = {key: [] for key in ('arguments','features','target')}
    with NativeFunctionProgram(args.program) as runtime:
        for case in windows:
            values = runtime.run(case['inputs'])
            for key in arrays:
                # Boundary tokens are excluded from reconstruction fitting.
                arrays[key].append(matrix(values[key])[1:-1].copy())
    np.savez(args.output, **{key: np.concatenate(value) for key,value in arrays.items()})
    return {'status':'completed','samples':sum(len(x) for x in arrays['target'])}


def apply_recovery(runtime, solution):
    import numpy as np
    data = np.load(solution); name = str(data['function']); before = time.perf_counter(); count = 0
    for binding, key in (('output_weight','weight'),('output_bias','bias')):
        for index, value in enumerate(data[key].ravel().tolist()):
            runtime.set_function_coefficient(name, binding, index, value); count += 1
    return {'coefficient_updates':count,'elapsed_seconds':time.perf_counter()-before}


def quality(args):
    import numpy as np
    from spark.function_runtime import NativeFunctionProgram
    from spark.function_ir import digest
    from spark.function_experiment import quality_rows, summarize_quality
    windows = json.loads(args.inputs.read_text()); rows = []
    file_hash = digest(args.program / 'coefficients.bin')
    libraries = {str(p): (digest(p),p.stat().st_mtime_ns) for p in (args.program/'native').glob('*.so')}
    with NativeFunctionProgram(args.program) as runtime:
        def evaluate(reference, save_to=None):
            observed=[]; storage=None
            for i,case in enumerate(windows):
                values = matrix(runtime.run(case['inputs'])['logits'])
                if save_to is not None:
                    if storage is None:
                        storage=np.lib.format.open_memmap(save_to, mode='w+', dtype=np.float64, shape=(len(windows),*values.shape))
                    storage[i]=values
                original = values if reference is None else reference[i]
                observed.extend(quality_rows(original,values,case['labels'],[case['article']]*len(case['labels'])))
            if storage is not None: storage.flush()
            return observed
        if args.reference is None:
            rows=evaluate(None,args.logits)
            report={'quality':summarize_quality(rows,samples=args.bootstrap),'rows':rows}
        else:
            reference=np.load(args.reference,mmap_mode='r')
            rows=evaluate(reference)
            report={'quality':summarize_quality(rows,samples=args.bootstrap),'rows':rows}
            if args.solutions_json:
                report['before_recovery']=report.pop('quality');report['rows_before_recovery']=report.pop('rows');report['variants']=[]
                for item in json.loads(args.solutions_json.read_text()):
                    changed=apply_recovery(runtime,Path(item['path']));rows=evaluate(reference)
                    report['variants'].append({'label':item['label'],'alpha':item['alpha'],'quality':summarize_quality(rows,samples=args.bootstrap),
                                               'updates':changed})
            elif args.solution:
                report['before_recovery']=report.pop('quality');report['rows_before_recovery']=report.pop('rows')
                report['live_update']=apply_recovery(runtime,args.solution)
                rows=evaluate(reference)
                report.update(quality=summarize_quality(rows,samples=args.bootstrap),rows=rows)
                report['live_update']['saved_coefficients_unchanged']=digest(args.program/'coefficients.bin')==file_hash
                report['live_update']['native_library_unchanged']=all(digest(Path(p))==h and Path(p).stat().st_mtime_ns==mtime for p,(h,mtime) in libraries.items())
                if args.persisted:
                    expected=matrix(runtime.run(windows[0]['inputs'])['logits']).copy()
                    # Close the large live model before loading the persisted one.
                    runtime.close()
                    with NativeFunctionProgram(args.persisted) as fresh:
                        actual=matrix(fresh.run(windows[0]['inputs'])['logits'])
                        report['live_update']['persisted_checkpoint_exact']=bool(np.array_equal(actual,expected))
                    if not all(report['live_update'][key] for key in ('saved_coefficients_unchanged','native_library_unchanged','persisted_checkpoint_exact')):
                        raise AssertionError('live/persisted recovery consistency failed')
        report['coefficient_file_unchanged']=digest(args.program/'coefficients.bin')==file_hash
    return report


def benchmark(args):
    """Time the compiled C ABI with preallocated, already-validated buffers."""
    import numpy as np
    from spark.function_runtime import NativeFunctionProgram, _flatten
    inputs=json.loads(args.inputs.read_text())
    with NativeFunctionProgram(args.program) as runtime:
        validated=runtime.run(inputs); symbols={}; arrays=[]
        for spec in runtime.program['main']['inputs']:
            flat,shape=_flatten(inputs[spec['id']]);arrays.append((ctypes.c_double*len(flat))(*flat))
            for actual,declared in zip(shape,spec['shape']):
                if type(declared) is str:symbols[declared]=actual
        dimensions=(ctypes.c_int64*len(symbols))(*(symbols[k] for k in runtime.program['symbols']))
        output_arrays=[(ctypes.c_double*math.prod(validated[name]['shape']))() for name in runtime.program['main']['outputs']]
        pointer=ctypes.POINTER(ctypes.c_double)
        input_pointers=(pointer*len(arrays))(*(ctypes.cast(a,pointer) for a in arrays))
        output_pointers=(pointer*len(output_arrays))(*(ctypes.cast(a,pointer) for a in output_arrays))
        def invoke():
            if not runtime.lib.spark_fn_run(runtime.handle,input_pointers,dimensions,output_pointers):raise ValueError(runtime._error())
        for _ in range(args.warmup):invoke()
        elapsed=[]
        for _ in range(args.repetitions):
            start=time.perf_counter_ns();invoke();elapsed.append((time.perf_counter_ns()-start)/1e6)
        # Keep output work observable and check deterministic ABI/public parity.
        exact=all(list(buffer)==list(validated[name]['values']) for name,buffer in zip(runtime.program['main']['outputs'],output_arrays))
        if not exact:raise AssertionError('benchmark C ABI output differs from validated public API')
        return {'native_milliseconds':elapsed,'median_native_milliseconds':float(np.median(elapsed)),
                'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'input_shapes':{v['id']:_flatten(inputs[v['id']])[1] for v in runtime.program['main']['inputs']},
                'output_shapes':{k:v['shape'] for k,v in validated.items()},'abi_public_parity_exact':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('collect','quality','benchmark'))
    parser.add_argument('--program',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path)
    parser.add_argument('--logits',type=Path);parser.add_argument('--solution',type=Path);parser.add_argument('--persisted',type=Path)
    parser.add_argument('--solutions-json',type=Path)
    parser.add_argument('--repetitions',type=int,default=3);parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--bootstrap',type=int,default=2000);parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();os.environ['OPENBLAS_NUM_THREADS']=str(args.threads);os.environ['OMP_NUM_THREADS']=str(args.threads)
    result={'status':'running'};started=time.perf_counter()
    try:
        result=globals()[args.mode](args);result['status']='completed'
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        if args.mode!='collect' or result.get('status')!='completed':args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','rows_before_recovery','quality','before_recovery')},indent=2),flush=True)


if __name__=='__main__':main()

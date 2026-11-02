#!/usr/bin/env python3
"""Compile the real pinned model, then validate the deployed function program."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
MODEL = 'answerdotai/ModernBERT-base'
REVISION = '8949b909ec900327062f0ebf497f51aef5e6f0c8'
SOURCE_SHA256 = 'c323c702d9ef69b8b6e62fa74075c26a0dcad49b38492270f1897eb35439996e'


def save(args, report):
    (args.output / (args.phase + '-tests.json')).write_text(json.dumps(report, indent=2) + '\n')


def compile_and_test(args, report):
    if any(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'numpy')):
        raise AssertionError('compiler package phase requires the runtime-only virtual environment')
    from spark.function_ir import digest, validate
    from spark.function_runtime import NativeFunctionProgram
    from spark.model_compiler import compile_modernbert_export
    report['stage'] = 'verify-source'
    if digest(args.source_export / 'weights.sparkmb') != SOURCE_SHA256:
        raise AssertionError('source export checksum does not match delivered checkpoint')
    report['stage'] = 'lower-and-generate-program'
    target = compile_modernbert_export(args.source_export, args.output / 'package')
    package = args.output / 'package'
    validate(target)
    report.update(coefficient_bytes=target['coefficient_bytes'], coefficients_sha256=target['coefficients_sha256'],
                  functions=len(target['functions']), main_operations=len(target['main']['nodes']),
                  total_operations=sum(len(f['nodes']) for f in [*target['functions'], target['main']]),
                  optimizations=target['optimizations'], target_format=target['format'])
    report['stage'] = 'standalone-function-program'
    process = subprocess.run([sys.executable, '-I', str(package / 'tools/run_function_program.py'), '--program', str(package),
                    '--text', 'The capital of France is [MASK].'], cwd=package, capture_output=True, text=True, check=True)
    isolated = json.loads(process.stdout)
    if any(isolated['framework_imports'].values()): raise AssertionError('standalone program imported a framework')
    report['standalone_cli'] = isolated
    sys.path.insert(0, str(ROOT / 'tools'))
    from run_function_program import text_inputs
    inputs, _ = text_inputs(package, target, 'The capital of France is [MASK].')
    native_library = next((package / 'native').glob('*.so')); library_hash = digest(native_library)
    checks = {'runtime_only_environment': True, 'standalone_cli': True,
              'no_source_runtime_bundled': not any(package.rglob('*modernbert*')),
              'no_source_model_in_native_kernels': 'ModernBert' not in (package / 'program.cpp').read_text() and 'modernbert' not in (package / 'runtime/function_ir_runtime.hpp').read_text().lower()}
    report['stage'] = 'live-function-and-composition-checks'
    with NativeFunctionProgram(package) as runtime:
        def values(): return list(runtime.run(inputs)['logits']['values'])
        baseline = values()
        live_functions = target['metadata']['dense_functions']
        for name in (live_functions[0], live_functions[len(live_functions) // 2], live_functions[-1]):
            capacity = runtime.functions[name][1]['live']['capacity']
            for width in (1, capacity // 2):
                runtime.set_active_width(name, width)
                checks[name + f'_width{width}_changes_output'] = values() != baseline
                runtime.set_active_width(name, capacity)
                checks[name + f'_width{width}_restores_exactly'] = values() == baseline
        function = live_functions[len(live_functions) // 2]
        for binding in ('input_weight', 'input_bias', 'output_weight', 'output_bias'):
            pid = runtime.functions[function][1]['live']['bindings'][binding]
            parameter = runtime.parameters[pid][1]
            with (package / 'coefficients.bin').open('rb') as stream:
                stream.seek(parameter['offset']); original = struct.unpack('<f', stream.read(4))[0]
            runtime.set_function_coefficient(function, binding, 0, original + .25)
            checks[binding + '_changes_output'] = values() != baseline
            runtime.set_function_coefficient(function, binding, 0, original)
            checks[binding + '_restores_exactly'] = values() == baseline
        for name, changed in (
                ('invalid_tokens', {**inputs, 'tokens': [-1] * len(inputs['tokens'])}),
                ('empty_attention', {**inputs, 'mask': [0] * len(inputs['mask'])}),
                ('duplicate_queries', {**inputs, 'queries': inputs['queries'] * 2}),
                ('invalid_positions', {**inputs, 'positions': [8192] * len(inputs['positions'])})):
            try: runtime.run(changed)
            except ValueError: checks[name + '_rejected'] = True
            else: checks[name + '_rejected'] = False
    with NativeFunctionProgram(package) as fresh:
        checks['fresh_reload_exact'] = list(fresh.run(inputs)['logits']['values']) == baseline
    # Provenance and frontend names have no execution semantics in the target.
    path = package / 'program.json'; original_program = path.read_text()
    anonymous = json.loads(original_program); anonymous.pop('source'); anonymous.pop('metadata')
    try:
        path.write_text(json.dumps(anonymous))
        with NativeFunctionProgram(package) as fresh:
            checks['runs_without_source_identity'] = list(fresh.run(inputs)['logits']['values']) == baseline
    finally:
        path.write_text(original_program)
    checks['coefficient_file_unchanged'] = digest(package / 'coefficients.bin') == target['coefficients_sha256']
    checks['live_updates_did_not_recompile'] = digest(native_library) == library_hash
    report['checks'] = checks
    if not all(checks.values()): raise AssertionError('compiled program checks failed: ' + ', '.join(k for k, v in checks.items() if not v))


def parity(args, report):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    from spark.function_runtime import NativeFunctionProgram
    sys.path.insert(0, str(ROOT / 'tools'))
    from convert_modernbert_complete import precision_reference, precision_softmax
    from test_converted_modernbert import make_cases
    torch.set_num_threads(2)
    report['stage'] = 'load-pinned-reference'
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, token=False, trust_remote_code=False)
    config.reference_compile = False
    teacher = AutoModelForMaskedLM.from_pretrained(MODEL, revision=REVISION, config=config, token=False, trust_remote_code=False,
                   use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager').cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, token=False, trust_remote_code=False)
    precise = precision_reference(teacher)
    cases = make_cases(tokenizer, ROOT / 'examples/modernbert_independent_masks.jsonl')
    report.update(stage='compare-native-function-program', comparisons=[], absolute_tolerance=.002, relative_tolerance=.0001,
                  mathematical_absolute_tolerance=1e-7, mathematical_relative_tolerance=1e-7)
    def reference(model, case):
        n = len(case['ids'])
        hidden = model.model(input_ids=torch.tensor([case['ids']]), attention_mask=torch.tensor([case.get('mask', [1] * n)]),
                            position_ids=torch.tensor([case.get('positions', list(range(n)))])).last_hidden_state
        return model.decoder(model.head(hidden[:, case['queries']]))[0].numpy(), hidden[0].numpy()
    with NativeFunctionProgram(args.output / 'package') as runtime:
        report['coefficients_sha256'] = runtime.program['coefficients_sha256']
        for case in cases:
            n = len(case['ids']); start = time.perf_counter()
            with torch.inference_mode(): expected, original_hidden = reference(teacher, case)
            with torch.inference_mode(), precision_softmax(): high_logits, high_hidden = reference(precise, case)
            mask = case.get('mask', [1] * n)
            actual = runtime.run({'tokens': case['ids'], 'mask': mask, 'positions': case.get('positions', list(range(n))), 'queries': case['queries']})
            logits = np.ctypeslib.as_array(actual['logits']['values']).reshape(actual['logits']['shape'])
            hidden = np.ctypeslib.as_array(actual['encoded']['values']).reshape(actual['encoded']['shape'])
            valid = np.asarray(mask, dtype=bool)
            checks = [np.allclose(logits, expected, atol=.002, rtol=.0001),
                      np.allclose(logits, high_logits, atol=1e-7, rtol=1e-7),
                      np.allclose(hidden[valid], high_hidden[valid], atol=1e-7, rtol=1e-7)]
            row = {'case': case['name'], 'sequence_length': n, 'queries': len(case['queries']), 'logit_values_checked': int(logits.size),
                   'allclose': bool(all(checks)), 'top1_agreement': int(np.sum(logits.argmax(1) == expected.argmax(1))),
                   'top5_set_agreement': sum(set(a.argsort()[-5:]) == set(b.argsort()[-5:]) for a, b in zip(logits, expected)),
                   'max_original_logit_error': float(np.max(np.abs(logits - expected))),
                   'max_mathematical_logit_error': float(np.max(np.abs(logits - high_logits))),
                   'max_mathematical_hidden_error': float(np.max(np.abs(hidden[valid] - high_hidden[valid]))),
                   'max_original_hidden_difference': float(np.max(np.abs(hidden[valid] - original_hidden[valid]))),
                   'predictions': [tokenizer.decode([int(i)]) for i in logits.argmax(1)], 'elapsed_seconds': time.perf_counter() - start}
            report['comparisons'].append(row); save(args, report); print(json.dumps(row), flush=True)
    rows = report['comparisons']
    report.update(test_cases=len(rows), queries=sum(r['queries'] for r in rows), logit_values_checked=sum(r['logit_values_checked'] for r in rows),
                  top1_agreement=sum(r['top1_agreement'] for r in rows), top5_set_agreement=sum(r['top5_set_agreement'] for r in rows),
                  max_sequence_tested=max(r['sequence_length'] for r in rows))
    for key in ('max_original_logit_error', 'max_mathematical_logit_error', 'max_mathematical_hidden_error', 'max_original_hidden_difference'):
        report[key] = max(r[key] for r in rows)
    if not all(r['allclose'] for r in rows) or report['top1_agreement'] != report['queries']:
        raise AssertionError('compiled target failed source parity gate')
    (args.output / 'package/validation.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('compile', 'parity'), required=True)
    parser.add_argument('--source-export', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/model-compiler')
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == 'compile' and args.source_export is None: parser.error('--compile requires --source-export')
    report = {'status': 'running', 'stage': 'initialize', 'checkpoint': MODEL, 'revision': REVISION, 'source_export_sha256': SOURCE_SHA256}
    start = time.perf_counter()
    try:
        (compile_and_test if args.phase == 'compile' else parity)(args, report)
        report.update(status='completed', stage='complete')
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=re.sub(r'https?://\S+', '<url>', str(error))[:2000])
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - start; save(args, report)


if __name__ == '__main__':
    main()

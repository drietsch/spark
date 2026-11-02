#!/usr/bin/env python3
"""Retest a delivered model artifact without exporting or replacing its weights."""
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

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 'answerdotai/ModernBERT-base'
REVISION = '8949b909ec900327062f0ebf497f51aef5e6f0c8'
WEIGHTS_SHA256 = 'c323c702d9ef69b8b6e62fa74075c26a0dcad49b38492270f1897eb35439996e'
SOURCE_RUN = 41348755846


def load_package(directory):
    # Import the downloaded copy, even when the repository has newer code.
    sys.path.insert(0, str(directory))
    import spark.modernbert_model as runtime
    if Path(runtime.__file__).resolve() != directory / 'spark/modernbert_model.py':
        raise AssertionError('test imported repository runtime instead of artifact runtime')
    manifest = runtime.validate_manifest(directory)
    if manifest['source'] != {'checkpoint': CHECKPOINT, 'revision': REVISION, 'pretrained': True}:
        raise AssertionError('unexpected source checkpoint')
    if manifest['weights_sha256'] != WEIGHTS_SHA256:
        raise AssertionError('artifact is not the previously delivered conversion')
    if runtime.sha256_file(directory / 'tokenizer/tokenizer.json') != manifest['tokenizer_sha256']:
        raise AssertionError('tokenizer checksum mismatch')
    return runtime, manifest


def packaged_cli(args, *inputs):
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([
        sys.executable, str(args.model / 'tools/run_converted_modernbert.py'),
        '--model', str(args.model), *inputs,
    ], cwd=args.model, env=environment, text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    if any(output['framework_imports'].values()):
        raise AssertionError('packaged CLI imported a teacher framework')
    return output


def package_tests(args, report):
    if any(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'numpy')):
        raise AssertionError('package phase requires a fresh runtime-only virtual environment')
    runtime, manifest = load_package(args.model)
    report['stage'] = 'recompile-packaged-native-source'
    runtime.compile_runtime(args.model, force=True)
    report['cli_text'] = packaged_cli(args, '--text', 'Water freezes at [MASK] degrees Celsius.')
    spec = json.loads((args.model / 'tokenizer/spark_tokenizer.json').read_text())
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer/tokenizer.json'))
    ids = [spec['cls_id'], *tokenizer.encode('A raven is a [MASK].', add_special_tokens=False).ids, spec['sep_id']]
    queries = [ids.index(spec['mask_id'])]
    request = args.output / 'cli-input.json'
    request.write_text(json.dumps({'input_ids': ids, 'query_positions': queries}))
    report['cli_ids'] = packaged_cli(args, '--ids-json', str(request))
    checks = {'runtime_only_environment': True, 'packaged_source_recompiled': True,
              'artifact_checksum_matches_delivered_model': True, 'tokenizer_checksum_valid': True}
    report['stage'] = 'live-functions-and-input-contract'
    with runtime.NativeModernBert(args.model) as native:
        def run():
            return list(native.run(ids, query_positions=queries)['logits'])
        baseline = run()
        for layer in (0, native.layers // 2, native.layers - 1):
            for width in (1, native.intermediate // 2):
                native.set_active_width(layer, width)
                checks[f'layer{layer}_width{width}_changes_output'] = run() != baseline
                native.set_active_width(layer, native.intermediate)
                checks[f'layer{layer}_width{width}_restores_exactly'] = run() == baseline
        bindings = {binding['name']: binding for binding in manifest['parameters']}
        for tensor in ('input_weight', 'input_bias', 'output_weight', 'output_bias'):
            layer = native.layers // 2
            binding = bindings[f'layer{layer}.function.' + tensor.replace('_', '.')]
            with (args.model / 'weights.sparkmb').open('rb') as stream:
                stream.seek(binding['offset'])
                original = struct.unpack('<f', stream.read(4))[0]
            native.set_function_coefficient(layer, tensor, 0, original + .25)
            checks[tensor + '_changes_output'] = run() != baseline
            native.set_function_coefficient(layer, tensor, 0, original)
            checks[tensor + '_restores_exactly'] = run() == baseline
        invalid = [
            ([native.vocab], {}), (ids, {'attention_mask': [0] * len(ids)}),
            (ids, {'query_positions': [len(ids)]}),
            (ids, {'query_positions': queries * 2}),
            (ids, {'position_ids': [native.max_sequence] * len(ids)}),
            (ids + [spec['pad_id']], {'attention_mask': [1] * len(ids) + [0], 'query_positions': [len(ids)]}),
        ]
        rejected = 0
        for tokens, kwargs in invalid:
            try:
                native.run(tokens, **kwargs)
            except ValueError:
                rejected += 1
        checks['invalid_requests_rejected'] = rejected == len(invalid)
    with runtime.NativeModernBert(args.model) as fresh:
        checks['reload_preserves_original_output'] = list(fresh.run(ids, query_positions=queries)['logits']) == baseline
    checks['checkpoint_file_unchanged'] = runtime.sha256_file(args.model / 'weights.sparkmb') == WEIGHTS_SHA256
    checks['frameworks_still_not_imported'] = not any(name in sys.modules for name in ('torch', 'transformers', 'numpy'))
    report['checks'] = checks
    if not all(checks.values()):
        raise AssertionError('package checks failed: ' + ', '.join(k for k, v in checks.items() if not v))


def make_cases(tokenizer, inputs):
    cases = []
    for line in inputs.read_text().splitlines():
        row = json.loads(line)
        ids = tokenizer(row['text'], add_special_tokens=True)['input_ids']
        cases.append({**row, 'ids': ids, 'queries': [i for i, v in enumerate(ids) if v == tokenizer.mask_token_id]})
    base = cases[4]
    for side in ('left', 'right'):
        padding = [tokenizer.pad_token_id] * 13
        left = side == 'left'
        cases.append({'name': side + '_padding', 'ids': padding + base['ids'] if left else base['ids'] + padding,
                      'mask': [0] * 13 + [1] * len(base['ids']) if left else [1] * len(base['ids']) + [0] * 13,
                      'queries': [q + (13 if left else 0) for q in base['queries']]})
    words = tokenizer(' The researcher recorded observations about forests, rivers, and seasonal rainfall.', add_special_tokens=False)['input_ids']
    for length in (512, 1024, 2048):
        ids = [tokenizer.cls_token_id] + (words * (length // len(words) + 1))[:length - 2] + [tokenizer.sep_token_id]
        queries = [64, length // 2, length - 66]
        for query in queries:
            ids[query] = tokenizer.mask_token_id
        cases.append({'name': f'context_length_{length}', 'ids': ids, 'queries': queries})
    base = cases[-2]
    cases.append({**base, 'name': 'rotary_index_8191', 'positions': list(range(8192 - len(base['ids']), 8192))})
    holes = {**cases[-3], 'name': 'internal_attention_holes'}
    holes['mask'] = [int(i % 7 != 0 or i in holes['queries']) for i in range(len(holes['ids']))]
    cases.append(holes)
    base = cases[0]
    cases.append({**base, 'name': 'reordered_nonmask_prediction_queries', 'queries': [len(base['ids']) - 2, 1, 0]})
    cases.append({**base, 'name': 'repeated_rotary_positions', 'positions': [i // 2 for i in range(len(base['ids']))]})
    return cases


def parity_tests(args, report):
    runtime, manifest = load_package(args.model)
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    sys.path.insert(0, str(ROOT / 'tools'))
    from convert_modernbert_complete import precision_reference, precision_softmax
    torch.set_num_threads(2)
    report['stage'] = 'load-original-checkpoint'
    config = AutoConfig.from_pretrained(CHECKPOINT, revision=REVISION, token=False, trust_remote_code=False)
    config.reference_compile = False
    teacher = AutoModelForMaskedLM.from_pretrained(
        CHECKPOINT, revision=REVISION, config=config, token=False, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager',
    ).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=REVISION, token=False, trust_remote_code=False)
    precise = precision_reference(teacher)
    cases = make_cases(tokenizer, args.inputs)
    report.update(stage='compare-delivered-artifact', comparisons=[], absolute_tolerance=.002,
                  relative_tolerance=.0001, mathematical_absolute_tolerance=1e-7,
                  mathematical_relative_tolerance=1e-7,
                  reference_execution='Original encoder, followed by original MLM head/decoder only at requested positions.')

    def reference(model, case):
        ids = case['ids']
        output = model.model(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([case.get('mask', [1] * len(ids))]),
                             position_ids=torch.tensor([case.get('positions', list(range(len(ids))))]))
        hidden = output.last_hidden_state
        return model.decoder(model.head(hidden[:, case['queries']]))[0].numpy(), hidden[0].numpy()

    with runtime.NativeModernBert(args.model) as native:
        for case in cases:
            started = time.perf_counter()
            with torch.inference_mode():
                expected, source_hidden = reference(teacher, case)
            with torch.inference_mode(), precision_softmax():
                high_logits, high_hidden = reference(precise, case)
            actual = native.run(case['ids'], query_positions=case['queries'], attention_mask=case.get('mask'),
                                position_ids=case.get('positions'), return_hidden=True)
            logits = np.ctypeslib.as_array(actual['logits']).reshape(actual['shape'])
            hidden = np.ctypeslib.as_array(actual['hidden']).reshape(actual['hidden_shape'])
            valid = np.asarray(case.get('mask', [1] * len(case['ids'])), dtype=bool)
            original_close = np.allclose(logits, expected, atol=.002, rtol=.0001)
            math_close = np.allclose(logits, high_logits, atol=1e-7, rtol=1e-7)
            hidden_close = np.allclose(hidden[valid], high_hidden[valid], atol=1e-7, rtol=1e-7)
            comparison = {
                'case': case['name'], 'sequence_length': len(case['ids']), 'queries': case['queries'],
                'top1_agreement': int(np.sum(logits.argmax(1) == expected.argmax(1))),
                'top5_set_agreement': sum(set(a.argsort()[-5:]) == set(b.argsort()[-5:]) for a, b in zip(logits, expected)),
                'logit_values_checked': int(logits.size), 'original_float32_allclose': bool(original_close),
                'mathematical_logits_allclose': bool(math_close), 'mathematical_hidden_allclose': bool(hidden_close),
                'max_original_logit_error': float(np.max(np.abs(logits - expected))),
                'max_mathematical_logit_error': float(np.max(np.abs(logits - high_logits))),
                'max_mathematical_hidden_error': float(np.max(np.abs(hidden[valid] - high_hidden[valid]))),
                'max_original_hidden_difference': float(np.max(np.abs(hidden[valid] - source_hidden[valid]))),
                'predictions': [tokenizer.decode([int(i)]) for i in logits.argmax(1)],
                'elapsed_seconds': time.perf_counter() - started,
            }
            report['comparisons'].append(comparison)
            # Preserve completed cases if a later case exhausts runner resources.
            save(args, report)
            print(json.dumps(comparison), flush=True)
    comparisons = report['comparisons']
    report.update(test_cases=len(cases), queries=sum(len(c['queries']) for c in cases),
                  max_sequence_tested=max(len(c['ids']) for c in cases),
                  logit_values_checked=sum(c['logit_values_checked'] for c in comparisons),
                  top1_agreement=sum(c['top1_agreement'] for c in comparisons),
                  top5_set_agreement=sum(c['top5_set_agreement'] for c in comparisons),
                  max_original_logit_error=max(c['max_original_logit_error'] for c in comparisons),
                  max_mathematical_logit_error=max(c['max_mathematical_logit_error'] for c in comparisons),
                  max_mathematical_hidden_error=max(c['max_mathematical_hidden_error'] for c in comparisons),
                  checkpoint_unchanged=runtime.sha256_file(args.model / 'weights.sparkmb') == WEIGHTS_SHA256)
    if not all(c[k] for c in comparisons for k in ('original_float32_allclose', 'mathematical_logits_allclose', 'mathematical_hidden_allclose')):
        raise AssertionError('independent numerical parity gate failed')
    if report['top1_agreement'] != report['queries'] or not report['checkpoint_unchanged']:
        raise AssertionError('prediction or checkpoint integrity gate failed')


def save(args, report):
    (args.output / (args.phase + '-tests.json')).write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('package', 'parity'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'examples/modernbert_independent_masks.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/modernbert-independent-test')
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {'status': 'running', 'stage': 'initialize', 'source_artifact_run': SOURCE_RUN,
              'checkpoint': CHECKPOINT, 'revision': REVISION, 'weights_sha256': WEIGHTS_SHA256}
    try:
        (package_tests if args.phase == 'package' else parity_tests)(args, report)
        report.update(status='completed', stage='complete')
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=re.sub(r'https?://\S+', '<url>', str(error))[:1500])
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        save(args, report)


if __name__ == '__main__':
    main()

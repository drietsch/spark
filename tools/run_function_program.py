#!/usr/bin/env python3
"""Execute a compiled SPARK function program, without source model frameworks."""
import argparse
import heapq
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.function_runtime import NativeFunctionProgram
from spark.function_ir import digest


def text_inputs(directory, program, text):
    from tokenizers import Tokenizer
    if program.get('metadata', {}).get('task') != 'masked-language-model':
        raise ValueError('this program has no masked-language text interface')
    tokenizer_path = directory / 'tokenizer/tokenizer.json'
    if digest(tokenizer_path) != program['metadata']['tokenizer_sha256']:
        raise ValueError('tokenizer checksum mismatch')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    spec = json.loads((directory / 'tokenizer/spark_tokenizer.json').read_text())
    ids = [spec['cls_id'], *tokenizer.encode(text, add_special_tokens=False).ids, spec['sep_id']]
    queries = [i for i, value in enumerate(ids) if value == spec['mask_id']]
    if not queries: raise ValueError('input must contain a mask token')
    return {'tokens': ids, 'mask': [1] * len(ids), 'positions': list(range(len(ids))), 'queries': queries}, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--inputs-json', type=Path); inputs.add_argument('--text')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.topk <= 256 or not 1 <= args.threads <= 8:
        parser.error('topk must be 1..256 and threads 1..8')
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    with NativeFunctionProgram(args.program) as program:
        if args.text:
            values, tokenizer = text_inputs(args.program, program.program, args.text)
        else:
            values = json.loads(args.inputs_json.read_text()); tokenizer = None
        output = program.run(values)
        if tokenizer:
            logits = output['logits']; vocab = logits['shape'][1]; predictions = []
            for row, query in enumerate(values['queries']):
                scores = logits['values'][row * vocab:(row + 1) * vocab]
                maximum = max(scores); denominator = sum(math.exp(v - maximum) for v in scores)
                top = heapq.nlargest(args.topk, range(vocab), key=lambda i: scores[i])
                predictions.append({'position': query, 'topk': [{'token': i, 'text': tokenizer.decode([i]), 'logit': scores[i],
                                    'prob': math.exp(scores[i] - maximum) / denominator} for i in top]})
            output = {'predictions': predictions}
        else:
            output = {k: {'shape': v['shape'], 'values': list(v['values'])} for k, v in output.items()}
        print(json.dumps({'backend': 'compiled-spark-function-program', **output,
                         'framework_imports': {name: name in sys.modules for name in ('torch', 'transformers', 'numpy')}}, indent=2))


if __name__ == '__main__':
    main()

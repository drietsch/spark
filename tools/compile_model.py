#!/usr/bin/env python3
"""Import an existing model and compile its computation into SPARK functions."""
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.model_compiler import compile_modernbert_export, compile_plastic_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint'); source.add_argument('--from-export', type=Path)
    source.add_argument('--plastic-function', type=Path)
    parser.add_argument('--revision'); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint:
        if not args.revision or not re.fullmatch('[0-9a-f]{40}', args.revision):
            parser.error('--checkpoint requires an immutable 40-character --revision')
        import torch
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
        from spark.modernbert_model import export_model
        config = AutoConfig.from_pretrained(args.checkpoint, revision=args.revision, token=False, trust_remote_code=False)
        if config.model_type != 'modernbert': parser.error('no importer registered for source architecture ' + config.model_type)
        config.reference_compile = False
        model = AutoModelForMaskedLM.from_pretrained(args.checkpoint, revision=args.revision, config=config,
                token=False, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager').cpu().eval()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision, token=False, trust_remote_code=False)
        with tempfile.TemporaryDirectory(prefix='spark-model-frontend-') as temporary:
            directory = Path(temporary)
            export_model(model, directory, source={'checkpoint': args.checkpoint, 'revision': args.revision, 'pretrained': True})
            tokenizer.save_pretrained(directory / 'tokenizer')
            (directory / 'tokenizer/spark_tokenizer.json').write_text(json.dumps({'cls_id': tokenizer.cls_token_id, 'sep_id': tokenizer.sep_token_id,
                    'mask_id': tokenizer.mask_token_id, 'pad_id': tokenizer.pad_token_id}))
            target = compile_modernbert_export(directory, args.output)
    elif args.from_export:
        target = compile_modernbert_export(args.from_export, args.output)
    else:
        from spark.plastic_function import PlasticFunction
        values = json.loads(args.plastic_function.read_text())
        target = compile_plastic_function(PlasticFunction(**values), args.output)
    print(json.dumps({'status': 'compiled', 'format': target['format'], 'functions': len(target['functions']),
          'main_operations': len(target['main']['nodes']), 'coefficient_bytes': target['coefficient_bytes'],
          'optimizations': target['optimizations'], 'program': str(args.output / 'program.json')}, indent=2))


if __name__ == '__main__':
    main()

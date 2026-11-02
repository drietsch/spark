"""Compiler semantics, source independence, and real frontend architecture fixtures."""
import copy
import ctypes.util
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from spark.function_ir import Builder, CompileError, FORMAT, arg, digest, expr, literal, validate
from spark.function_runtime import NativeFunctionProgram
from spark.model_compiler import compile_modernbert, compile_plastic_function, typed, write_package
from spark.plastic_function import PlasticFunction

NATIVE = bool(shutil.which('c++') and ctypes.util.find_library('blas'))
TEACHER = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'transformers'))


def simple_program(builder, output, *, functions=None, parameters=None, preconditions=None):
    return {'format': FORMAT, 'symbols': {'N': [1, 128]}, 'parameters': parameters or [], 'functions': functions or [],
            'main': {'inputs': builder.inputs, 'nodes': builder.nodes, 'outputs': {'result': output}, 'preconditions': preconditions or []}}


@unittest.skipUnless(NATIVE, 'C++17 and BLAS required')
class FunctionCompilerTests(unittest.TestCase):
    def test_generated_source_is_stable_across_python_hash_seeds(self):
        builder = Builder([typed('x', ['N', 2])])
        a = builder.map(expr('add', arg(0), literal(1)), 'x')
        b = builder.map(expr('mul', arg(0), literal(2)), 'x')
        result = builder.map(expr('sub', arg(0), arg(1)), a, b)
        program = simple_program(builder, result)
        code = 'import hashlib,json,sys;from spark.function_ir import emit_cpp;print(hashlib.sha256(emit_cpp(json.load(sys.stdin)).encode()).hexdigest())'
        hashes = []
        for seed in ('1', '19', '42'):
            environment = os.environ.copy(); environment['PYTHONHASHSEED'] = seed
            process = subprocess.run([sys.executable, '-c', code], input=json.dumps(program), text=True,
                                     capture_output=True, check=True, env=environment,
                                     cwd=Path(__file__).resolve().parents[1])
            hashes.append(process.stdout.strip())
        self.assertEqual(len(set(hashes)), 1)

    def test_optimizer_and_program_semantics_control_native_execution(self):
        builder = Builder([typed('x', ['N', 2])])
        first = builder.map(expr('add', arg(0), arg(0)), 'x')
        duplicate = builder.map(expr('add', arg(0), arg(0)), 'x')
        builder.map(expr('exp', arg(0)), 'x')  # Dead operation must not execute.
        result = builder.map(expr('add', arg(0), arg(1)), first, duplicate)
        source = simple_program(builder, result)
        inputs = {'x': [[1., 2.], [-3., .5]]}
        with tempfile.TemporaryDirectory() as folder:
            program = write_package(source, {}, folder)
            self.assertEqual(program['optimizations']['common_and_dead_nodes_removed'], 2)
            with NativeFunctionProgram(folder) as runtime:
                self.assertEqual(list(runtime.run(inputs)['result']['values']), [4., 8., -12., 2.])
            # A valid target-program edit changes the compiled computation;
            # no source model or model-specific layer evaluator is involved.
            program['main']['nodes'][-1]['attrs']['expression']['op'] = 'mul'
            (Path(folder) / 'program.json').write_text(json.dumps(program))
            with NativeFunctionProgram(folder) as runtime:
                self.assertEqual(list(runtime.run(inputs)['result']['values']), [4., 16., 36., 1.])

    def test_generic_bounded_function_map_indexed_arithmetic_and_reductions(self):
        body = Builder([typed('x', ['N', 2])])
        squares = body.map(expr('mul', arg(0), arg(0)), 'x')
        total = body.node('reduce', squares, axis=1, kind='sum')
        body_function = body.function('main', total)
        main = Builder([typed('x', [3, 'N', 2])])
        mapped = main.node('map_call', 'x', function='main', axes=[0])
        index = {'op': 'index', 'index': 0}
        result = main.map(expr('add', arg(0), index), mapped)
        source = simple_program(main, result, functions=[body_function])
        rows = [[[1., 2.], [3., 4.]], [[-1., 0.], [0., 2.]], [[2., 2.], [1., 3.]]]
        with tempfile.TemporaryDirectory() as folder:
            write_package(source, {}, folder)
            with NativeFunctionProgram(folder) as runtime:
                actual = runtime.run({'x': rows})['result']
                self.assertEqual(actual['shape'], [3, 2, 1])
                self.assertEqual(list(actual['values']), [5., 25., 2., 5., 10., 12.])

    def test_unused_captures_are_removed_and_payload_is_checked(self):
        builder = Builder([typed('x', ['N', 2])])
        builder.parameter('unused')
        result = builder.map(expr('add', arg(0), literal(1)), 'x')
        parameters = [{'id': 'unused', 'shape': [1], 'storage': 'f64', 'offset': 0, 'bytes': 8, 'mutable': False}]
        with tempfile.TemporaryDirectory() as folder:
            program = write_package(simple_program(builder, result, parameters=parameters), {'unused': struct.pack('<d', 17.)}, folder)
            self.assertEqual(program['parameters'], [])
            self.assertEqual(program['coefficient_bytes'], 0)
            self.assertEqual(program['optimizations']['unused_parameters_removed'], 1)
            with (Path(folder) / 'coefficients.bin').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'size mismatch'):
                NativeFunctionProgram(folder)

    def test_m13_float64_coefficients_width_updates_and_private_reload(self):
        function = PlasticFunction.initialize(capacity=8, active=5, seed=9)
        points = [[.2, .3, -.7], [1., 2., 3.], [-4., .1, 1.7]]
        with tempfile.TemporaryDirectory() as folder:
            program = compile_plastic_function(function, folder)
            checksum = digest(Path(folder) / 'coefficients.bin')
            self.assertTrue(all(p['storage'] == 'f64' for p in program['parameters']))
            with NativeFunctionProgram(folder) as runtime:
                def values(): return list(runtime.run({'x': points})['value']['values'])
                baseline = values()
                for actual, point in zip(baseline, points): self.assertAlmostEqual(actual, function.forward(point)[0], places=14)
                for width in (1, 8):
                    runtime.set_active_width('function0', width); function.set_active(width)
                    for actual, point in zip(values(), points): self.assertAlmostEqual(actual, function.forward(point)[0], places=14)
                runtime.set_active_width('function0', 5); self.assertEqual(values(), baseline)
                runtime.set_function_coefficient('function0', 'readout_bias', 0, function.bias + .5)
                for actual, original in zip(values(), baseline): self.assertAlmostEqual(actual - original, .5, places=14)
                runtime.set_function_coefficient('function0', 'readout_bias', 0, function.bias)
                self.assertEqual(values(), baseline)
                with self.assertRaises(ValueError): runtime.set_active_width('function0', 9)
                with self.assertRaises(ValueError): runtime.set_function_coefficient('function0', 'unknown', 0, 1.)
                with self.assertRaises(ValueError): runtime.set_active_width('function0', 2**32 + 1)
                with self.assertRaises(ValueError): runtime.set_function_coefficient('function0', 'readout_bias', 2**64, 1.)
                self.assertEqual(values(), baseline)
            self.assertEqual(digest(Path(folder) / 'coefficients.bin'), checksum)
            with NativeFunctionProgram(folder) as fresh:
                self.assertEqual(list(fresh.run({'x': points})['value']['values']), baseline)

    def test_standalone_package_imports_no_source_runtime_or_framework(self):
        builder = Builder([typed('x', ['N', 2])])
        result = builder.map(expr('mul', arg(0), literal(3)), 'x')
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); write_package(simple_program(builder, result), {}, directory)
            (directory / 'input.json').write_text(json.dumps({'x': [[1., 2.]]}))
            process = subprocess.run([sys.executable, '-I', str(directory / 'tools/run_function_program.py'), '--program', folder,
                       '--inputs-json', str(directory / 'input.json')], cwd=folder, text=True, capture_output=True, check=True)
            output = json.loads(process.stdout)
            self.assertFalse(any(output['framework_imports'].values()))
            self.assertEqual(output['result']['values'], [3., 6.])
            self.assertFalse(list(directory.rglob('*modernbert*')))
            self.assertNotIn('ModernBert', (directory / 'program.cpp').read_text())

    def test_type_checker_rejects_unsupported_shapes_expressions_and_cycles(self):
        builder = Builder([typed('x', ['N', 2])]); result = builder.map(expr('add', arg(0), literal(1)), 'x')
        program = simple_program(builder, result)
        for mutation in ('source_op', 'shape', 'expression', 'forward_reference', 'identifier'):
            changed = copy.deepcopy(program)
            if mutation == 'source_op': changed['main']['nodes'][0]['op'] = 'modernbert_attention'
            elif mutation == 'shape': changed['main']['inputs'][0]['shape'] = ['undefined', 2]
            elif mutation == 'expression': changed['main']['nodes'][0]['attrs']['expression']['op'] = 'eval_python'
            elif mutation == 'identifier': changed['main']['nodes'][0]['id'] = 'x;system()'
            else: changed['main']['nodes'][0]['inputs'] = ['not_yet_defined']
            with self.subTest(mutation=mutation), self.assertRaises(CompileError): validate(changed)
        body = Builder([typed('x', ['N', 2])]); recursive = body.node('call', 'x', function='recursive')
        changed = copy.deepcopy(program); changed['functions'] = [body.function('recursive', recursive)]
        with self.assertRaisesRegex(CompileError, 'recursive'): validate(changed)

    def test_runtime_input_shape_discrete_and_selection_contracts(self):
        builder = Builder([typed('x', ['N', 2]), typed('indices', ['N'], 'index')])
        result = builder.node('gather', 'x', 'indices')
        program = simple_program(builder, result, preconditions=[{'op': 'bounds', 'input': 'indices', 'upper': 'N'}])
        with tempfile.TemporaryDirectory() as folder:
            write_package(program, {}, folder)
            with NativeFunctionProgram(folder) as runtime:
                with self.assertRaises(ValueError): runtime.run({'x': [[1., 2.]], 'indices': [1]})
                with self.assertRaises(ValueError): runtime.run({'x': [[1., 2.]], 'indices': [0.]})
                with self.assertRaises(ValueError): runtime.run({'x': [[1., 2.], [1.]], 'indices': [0, 1]})
                with self.assertRaises(ValueError): runtime.run({'x': [[1., 2.]], 'indices': [0, 0]})
                with self.assertRaises(ValueError): runtime.run({'x': [[math.nan, 2.]], 'indices': [0]})


@unittest.skipUnless(NATIVE and TEACHER, 'Optional PyTorch/Transformers and C++17/BLAS required')
class ModernBertCompilerTests(unittest.TestCase):
    def test_full_architecture_variants_and_attention_boundaries(self):
        import numpy as np
        import torch
        from transformers import ModernBertConfig, ModernBertForMaskedLM
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        from convert_modernbert_complete import precision_reference, precision_softmax
        torch.set_num_threads(1)
        ids = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 2, 0, 0]
        masks = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]; queries = [10, 1]
        positions = list(range(230, 230 + len(ids)))
        for bias, tied, activation in ((False, True, 'gelu'), (True, False, 'gelu_pytorch_tanh')):
            with self.subTest(bias=bias, tied=tied):
                torch.manual_seed(27)
                config = ModernBertConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=4, num_attention_heads=4,
                        vocab_size=48, max_position_embeddings=256, local_attention=8, global_attn_every_n_layers=2,
                        reference_compile=False, pad_token_id=0, cls_token_id=1, sep_token_id=2, attention_bias=bias,
                        mlp_bias=bias, classifier_bias=bias, norm_bias=bias, tie_word_embeddings=tied,
                        hidden_activation=activation, classifier_activation=activation)
                config._attn_implementation = 'eager'; source = ModernBertForMaskedLM(config).eval()
                if bias:
                    with torch.no_grad():
                        for name, p in source.named_parameters():
                            if name.endswith('bias'): p.uniform_(-.02, .02)
                precise = precision_reference(source)
                with torch.inference_mode(), precision_softmax():
                    reference = precise.model(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([masks]), position_ids=torch.tensor([positions])).last_hidden_state
                    expected = precise.decoder(precise.head(reference[:, queries]))[0].numpy()
                with tempfile.TemporaryDirectory() as folder:
                    program = compile_modernbert(source, folder, source={'pretrained': False})
                    self.assertEqual(len(program['functions']), 6)
                    self.assertEqual(program['metadata']['source_operators_remaining'], 0)
                    self.assertNotIn('rotary_attention_residual', json.dumps(program))
                    with NativeFunctionProgram(folder) as runtime:
                        actual = runtime.run({'tokens': ids, 'mask': masks, 'positions': positions, 'queries': queries})
                        logits = np.ctypeslib.as_array(actual['logits']['values']).reshape(actual['logits']['shape'])
                        hidden = np.ctypeslib.as_array(actual['encoded']['values']).reshape(actual['encoded']['shape'])
                        np.testing.assert_allclose(logits, expected, atol=1e-10, rtol=1e-10)
                        np.testing.assert_allclose(hidden[np.asarray(masks, dtype=bool)], reference[0].numpy()[np.asarray(masks, dtype=bool)], atol=1e-10, rtol=1e-10)
                        baseline = logits.copy()
                        runtime.set_active_width('function1', 1)
                        changed = runtime.run({'tokens': ids, 'mask': masks, 'positions': positions, 'queries': queries})
                        self.assertFalse(np.array_equal(np.ctypeslib.as_array(changed['logits']['values']).reshape(2, 48), baseline))
                        runtime.set_active_width('function1', 24)
                        restored = runtime.run({'tokens': ids, 'mask': masks, 'positions': positions, 'queries': queries})
                        np.testing.assert_array_equal(np.ctypeslib.as_array(restored['logits']['values']).reshape(2, 48), baseline)

    def test_unsupported_source_operation_fails_without_approximation(self):
        from transformers import ModernBertConfig, ModernBertForMaskedLM
        config = ModernBertConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=4,
                    vocab_size=48, reference_compile=False, hidden_activation='relu', pad_token_id=0, cls_token_id=1, sep_token_id=2)
        config._attn_implementation = 'eager'
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'activation'):
            compile_modernbert(ModernBertForMaskedLM(config).eval(), folder, source={'pretrained': False})
        config.hidden_activation = 'gelu'
        source = ModernBertForMaskedLM(config).eval()
        source.forward = lambda *args, **kwargs: None
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(CompileError, 'override'):
            compile_modernbert(source, folder, source={'pretrained': False})


if __name__ == '__main__':
    unittest.main()

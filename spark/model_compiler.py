"""Source adapters lowering existing models into SPARK function programs.

ModernBERT's existing export is a frontend interchange format only. Its layer
operators and binary header never enter the deployed target. The M13 adapter
also uses the same compiler, preserving its original float64 coefficients.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import struct
import tempfile

from spark.function_ir import Builder, FORMAT, arg, compile_program, digest, expr, literal, optimize, require, validate


def typed(name, shape, dtype='f64'):
    return {'id': name, 'dtype': dtype, 'shape': shape}


def gelu(x, activation):
    half_x = expr('mul', literal(.5), x)
    if activation == 'gelu':
        inner = expr('erf', expr('mul', x, literal(math.sqrt(.5))))
    elif activation in ('gelu_new', 'gelu_pytorch_tanh'):
        cube = expr('mul', expr('mul', x, x), x)
        inner = expr('tanh', expr('mul', literal(math.sqrt(2 / math.pi)), expr('add', x, expr('mul', literal(.044715), cube))))
    else:
        raise ValueError('unsupported source activation: ' + str(activation))
    return expr('mul', half_x, expr('add', literal(1), inner))


def write_package(program, readers, directory, *, tokenizer=None):
    """Serialize reachable target captures and bundle only source-independent code."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        require((directory / 'program.json').is_file() and
                json.loads((directory / 'program.json').read_text()).get('format') == FORMAT,
                'compiler output must be empty or an existing SPARK function package')
    directory.mkdir(parents=True, exist_ok=True)
    target = optimize(program)
    with (directory / 'coefficients.bin.tmp').open('wb') as output:
        for parameter in target['parameters']:
            data = readers[parameter['id']]
            if isinstance(data, bytes):
                require(len(data) == parameter['bytes'], 'coefficient reader size mismatch')
                output.write(data)
            else:
                path, offset, size = data
                require(size == parameter['bytes'], 'coefficient reader size mismatch')
                with Path(path).open('rb') as source:
                    source.seek(offset)
                    remaining = size
                    while remaining:
                        block = source.read(min(4 * 2**20, remaining))
                        require(bool(block), 'truncated source coefficient capture')
                        output.write(block); remaining -= len(block)
    (directory / 'coefficients.bin.tmp').replace(directory / 'coefficients.bin')
    target['coefficients_sha256'] = digest(directory / 'coefficients.bin')
    root = Path(__file__).resolve().parents[1]
    for relative in ('spark/function_ir.py', 'spark/function_runtime.py', 'runtime/function_ir_runtime.hpp', 'tools/run_function_program.py'):
        output = directory / relative; output.parent.mkdir(parents=True, exist_ok=True)
        if (root / relative).resolve() != output.resolve():
            shutil.copyfile(root / relative, output)
    (directory / 'spark/__init__.py').write_text('"""Standalone compiled SPARK function program."""\n')
    if tokenizer is not None:
        shutil.copytree(tokenizer, directory / 'tokenizer', dirs_exist_ok=True)
        target.setdefault('metadata', {})['tokenizer_sha256'] = digest(directory / 'tokenizer/tokenizer.json')
    (directory / 'program.json').write_text(json.dumps(target, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (directory / 'README.txt').write_text(
        'Compiled SPARK function program.\n'
        'Run numerical input: python tools/run_function_program.py --program . --inputs-json inputs.json\n'
        'For text-enabled packages: python tools/run_function_program.py --program . --text "The capital of France is [MASK]."\n'
        'Requires Python, a C++17 compiler and CPU BLAS (libblas.so.3); text input also needs tokenizers==0.21.4.\n'
        'The target contains generated functions and generic numerical kernels. No source model framework is used for inference.\n'
        'Coefficients and dense arithmetic remain. Numerical profile and source provenance are in program.json.\n')
    compile_program(directory)
    return target


def lower_modernbert_export(source_directory):
    """Lower a checked source export into primitive functions, without source calls."""
    from spark.modernbert_model import validate_manifest
    source_directory = Path(source_directory)
    source = validate_manifest(source_directory)
    geometry = source['geometry']; constants = source['constants']
    h = geometry['hidden']; intermediate = geometry['intermediate']; heads = geometry['heads']; dimension = h // heads
    source_parameters = {p['name']: p for p in source['parameters']}
    parameters = []; mapping = {}; readers = {}; cursor = 0

    def binding(name, mutable=False):
        nonlocal cursor
        require(name in source_parameters, 'missing source tensor: ' + name)
        if name not in mapping:
            p = source_parameters[name]; identity = f'p{len(parameters)}'
            parameters.append({'id': identity, 'shape': p['shape'], 'storage': 'f32', 'offset': cursor, 'bytes': p['bytes'], 'mutable': mutable})
            cursor += p['bytes']; mapping[name] = identity
            readers[identity] = (source_directory / 'weights.sparkmb', p['offset'], p['bytes'])
        elif mutable:
            next(p for p in parameters if p['id'] == mapping[name])['mutable'] = True
        return mapping[name]

    def param(builder, name, mutable=False):
        return builder.parameter(binding(name, mutable))

    def norm(builder, x, prefix):
        mean = builder.node('reduce', x, axis=1, kind='sum')
        centered = builder.map(expr('sub', arg(0), expr('div', arg(1), literal(h))), x, mean)
        squares = builder.map(expr('mul', arg(0), arg(0)), centered)
        variance = builder.node('reduce', squares, axis=1, kind='sum')
        weight = param(builder, prefix + '.weight'); bias = param(builder, prefix + '.bias')
        denominator = expr('sqrt', expr('add', expr('div', arg(1), literal(h)), literal(constants['norm_epsilon'])))
        return builder.map(expr('add', expr('mul', expr('div', arg(0), denominator), arg(2)), arg(3)), centered, variance, weight, bias)

    def linear(builder, x, prefix):
        w = param(builder, prefix + '.weight'); b = param(builder, prefix + '.bias')
        transposed = builder.node('transpose', w, axes=[1, 0])
        product = builder.node('contract', x, transposed)
        return builder.map(expr('add', arg(0), arg(1)), product, b)

    functions = []
    for local in (False, True):
        builder = Builder([typed('q', ['N', dimension]), typed('k', ['N', dimension]), typed('v', ['N', dimension]), typed('mask', ['N'], 'bool')])
        kt = builder.node('transpose', 'k', axes=[1, 0])
        score = builder.node('contract', 'q', kt)
        allowed = expr('gt', arg(1), literal(.5))
        if local:
            distance = expr('abs', expr('sub', {'op': 'index', 'index': 0}, {'op': 'index', 'index': 1}))
            allowed = expr('and', allowed, expr('lt', distance, literal(geometry['local_window'] // 2 + 1)))
        scaled = expr('mul', arg(0), literal(1 / math.sqrt(dimension)))
        masked = builder.map(expr('where', allowed, scaled, literal('-inf')), score, 'mask')
        maximum = builder.node('reduce', masked, axis=1, kind='max')
        exponential = builder.map(expr('where', expr('isfinite', arg(1)), expr('exp', expr('sub', arg(0), arg(1))), literal(0)), masked, maximum)
        total = builder.node('reduce', exponential, axis=1, kind='sum')
        probabilities = builder.map(expr('where', expr('gt', arg(1), literal(0)), expr('div', arg(0), arg(1)), literal(0)), exponential, total)
        output = builder.node('contract', probabilities, 'v')
        functions.append(builder.function('kernel1' if local else 'kernel0', output))

    for layer in range(geometry['layers']):
        prefix = f'layer{layer}.function'
        builder = Builder([typed('x', ['N', h])])
        captures = {key: binding(prefix + '.' + key.replace('_', '.'), True) for key in ('input_weight', 'input_bias', 'output_weight', 'output_bias')}
        wi = builder.parameter(captures['input_weight']); bi = builder.parameter(captures['input_bias'])
        branches = []
        for start in (0, intermediate):
            weight = builder.node('slice', wi, axis=0, start=start, size='W')
            bias = builder.node('slice', bi, axis=0, start=start, size='W')
            transposed = builder.node('transpose', weight, axes=[1, 0])
            projected = builder.node('contract', 'x', transposed)
            branches.append(builder.map(expr('add', arg(0), arg(1)), projected, bias))
        hidden = builder.map(expr('mul', gelu(arg(0), constants['mlp_activation']), arg(1)), *branches)
        wo = builder.parameter(captures['output_weight']); bo = builder.parameter(captures['output_bias'])
        weight = builder.node('slice', wo, axis=1, start=0, size='W')
        transposed = builder.node('transpose', weight, axes=[1, 0])
        projected = builder.node('contract', hidden, transposed)
        output = builder.map(expr('add', arg(0), arg(1)), projected, bo)
        functions.append(builder.function(f'function{layer}', output,
                         {'symbol': 'W', 'capacity': intermediate, 'initial': intermediate, 'bindings': captures}))

    builder = Builder([typed('tokens', ['N'], 'index'), typed('mask', ['N'], 'bool'),
                       typed('positions', ['N'], 'index'), typed('queries', ['Q'], 'index')])
    embedding = param(builder, 'embedding.weight')
    x = builder.node('gather', embedding, 'tokens')
    x = norm(builder, x, 'embedding.norm')
    tables = {}
    for kind in ('global', 'local'):
        tables[kind] = [builder.node('gather', param(builder, f'rotary.{kind}.{trig}'), 'positions') for trig in ('cosine', 'sine')]

    def rotary(value, kind):
        first = builder.node('slice', value, axis=2, start=0, size=dimension // 2)
        second = builder.node('slice', value, axis=2, start=dimension // 2, size=dimension // 2)
        cosine, sine = tables[kind]
        a = builder.map(expr('sub', expr('mul', arg(0), arg(2)), expr('mul', arg(1), arg(3))), first, second, cosine, sine)
        b = builder.map(expr('add', expr('mul', arg(1), arg(2)), expr('mul', arg(0), arg(3))), first, second, cosine, sine)
        return builder.node('concat', a, b, axis=2)

    for layer in range(geometry['layers']):
        prefix = f'layer{layer}'
        normalized = x if layer == 0 else norm(builder, x, prefix + '.attention.norm')
        projected = linear(builder, normalized, prefix + '.attention.qkv')
        local = layer % geometry['global_every'] != 0; kind = 'local' if local else 'global'
        qkv = []
        for group in range(3):
            value = builder.node('slice', projected, axis=1, start=group * h, size=h)
            value = builder.node('reshape', value, shape=['N', heads, dimension])
            value = builder.node('transpose', value, axes=[1, 0, 2])
            qkv.append(rotary(value, kind) if group < 2 else value)
        attended = builder.node('map_call', *qkv, 'mask', function='kernel1' if local else 'kernel0', axes=[0, 0, 0, None])
        merged = builder.node('transpose', attended, axes=[1, 0, 2])
        merged = builder.node('reshape', merged, shape=['N', h])
        output = linear(builder, merged, prefix + '.attention.output')
        x = builder.map(expr('add', arg(0), arg(1)), x, output)
        normalized = norm(builder, x, prefix + '.function.norm')
        delta = builder.node('call', normalized, function=f'function{layer}')
        x = builder.map(expr('add', arg(0), arg(1)), x, delta)
    encoded = norm(builder, x, 'final.norm')
    selected = builder.node('gather', encoded, 'queries')
    prediction = linear(builder, selected, 'prediction.dense')
    prediction = builder.map(gelu(arg(0), constants['prediction_activation']), prediction)
    prediction = norm(builder, prediction, 'prediction.norm')
    decoder = embedding if geometry['tied_decoder'] else param(builder, 'decoder.weight')
    decoder = builder.node('transpose', decoder, axes=[1, 0])
    logits = builder.node('contract', prediction, decoder)
    logits = builder.map(expr('add', arg(0), arg(1)), logits, param(builder, 'decoder.bias'))
    program = {
        'format': FORMAT, 'symbols': {'N': [1, geometry['max_sequence']], 'Q': [1, geometry['max_sequence']]},
        'parameters': parameters, 'functions': functions, 'coefficient_bytes': cursor,
        'main': {'inputs': builder.inputs, 'nodes': builder.nodes, 'outputs': {'logits': logits, 'encoded': encoded},
                 'preconditions': [
                     {'op': 'bounds', 'input': 'tokens', 'upper': geometry['vocab']},
                     {'op': 'bounds', 'input': 'positions', 'upper': geometry['max_sequence']},
                     {'op': 'bounds', 'input': 'queries', 'upper': 'N'},
                     {'op': 'unique', 'input': 'queries'}, {'op': 'any', 'input': 'mask'},
                     {'op': 'selected_true', 'input': 'mask', 'indices': 'queries'},
                 ]},
        'source': source['source'],
        'metadata': {'frontend': 'modernbert-v1', 'numerical_profile': 'float64 arithmetic with unchanged float32 coefficients and rotary tables',
                     'task': 'masked-language-model', 'dense_functions': [f'function{i}' for i in range(geometry['layers'])],
                     'source_operators_remaining': 0, 'approximation': False},
    }
    validate(program)
    return program, readers


def compile_modernbert_export(source_directory, directory):
    program, readers = lower_modernbert_export(source_directory)
    tokenizer = Path(source_directory) / 'tokenizer'
    return write_package(program, readers, directory, tokenizer=tokenizer if tokenizer.exists() else None)


def compile_modernbert(model, directory, *, source):
    """Offline frontend for an HF eval-mode ModernBERT object; no fitting/training."""
    from spark.modernbert_model import export_model
    with tempfile.TemporaryDirectory(prefix='spark-source-frontend-') as temporary:
        export_model(model, temporary, source=source)
        return compile_modernbert_export(temporary, directory)


def compile_plastic_function(function, directory):
    """Import the existing M13 three-input function into the same target language."""
    function.validate(); capacity = function.capacity
    parameters = []; readers = {}; cursor = 0
    def capture(values, shape):
        nonlocal cursor
        name = f'p{len(parameters)}'; data = struct.pack('<' + 'd' * len(values), *values)
        parameters.append({'id': name, 'shape': shape, 'storage': 'f64', 'offset': cursor, 'bytes': len(data), 'mutable': True})
        cursor += len(data); readers[name] = data
        return name
    units = function.units
    bindings = {
        'hidden_weight': capture([v for u in units for v in u[:3]], [capacity, 3]),
        'hidden_bias': capture([u[3] for u in units], [capacity]),
        'gate_weight': capture([v for u in units for v in u[4:7]], [capacity, 3]),
        'gate_bias': capture([u[7] for u in units], [capacity]),
        'readout_weight': capture([u[8] for u in units], [capacity, 1]),
        'readout_bias': capture([function.bias], [1]),
    }
    builder = Builder([typed('x', ['N', 3])]); branches = []
    for kind in ('hidden', 'gate'):
        weight = builder.node('slice', builder.parameter(bindings[kind + '_weight']), axis=0, start=0, size='W')
        bias = builder.node('slice', builder.parameter(bindings[kind + '_bias']), axis=0, start=0, size='W')
        product = builder.node('contract', 'x', builder.node('transpose', weight, axes=[1, 0]))
        branches.append(builder.map(expr('add', arg(0), arg(1)), product, bias))
    exponential = expr('exp', arg(1))
    sigmoid = expr('where', expr('gt', arg(1), literal(0)),
                   expr('div', literal(1), expr('add', literal(1), expr('exp', expr('neg', arg(1))))),
                   expr('div', exponential, expr('add', literal(1), exponential)))
    hidden = builder.map(expr('mul', expr('tanh', arg(0)), sigmoid), *branches)
    readout = builder.node('slice', builder.parameter(bindings['readout_weight']), axis=0, start=0, size='W')
    result = builder.node('contract', hidden, readout)
    result = builder.map(expr('add', arg(0), arg(1)), result, builder.parameter(bindings['readout_bias']))
    target_function = builder.function('function0', result, {'symbol': 'W', 'capacity': capacity, 'initial': function.active, 'bindings': bindings})
    main = Builder([typed('x', ['N', 3])]); result = main.node('call', 'x', function='function0')
    program = {'format': FORMAT, 'symbols': {'N': [1, 8192]}, 'parameters': parameters, 'functions': [target_function],
               'coefficient_bytes': cursor, 'source': {'model': 'M13 PlasticFunction'},
               'metadata': {'frontend': 'plastic-function-v1', 'approximation': False, 'numerical_profile': 'original float64 coefficients'},
               'main': {'inputs': main.inputs, 'nodes': main.nodes, 'outputs': {'value': result}, 'preconditions': []}}
    return write_package(program, readers, directory)

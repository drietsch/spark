"""Explicit approximate transformations of verified SPARK function programs.

These operations are separate from the exact importer/compiler. They operate
on target function bodies and captures, not on source architecture classes.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import shutil

from spark.function_ir import canonical, digest, require, validate
from spark.model_compiler import write_package


def read_program(directory):
    directory = Path(directory)
    program = json.loads((directory / 'program.json').read_text())
    validate(program)
    require(digest(directory / 'coefficients.bin') == program['coefficients_sha256'], 'source coefficient checksum mismatch')
    return program


def readers_for(program, directory):
    return {p['id']: (Path(directory) / 'coefficients.bin', p['offset'], p['bytes']) for p in program['parameters']}


def parameter_array(program, directory, identity):
    import numpy as np
    parameter = next(p for p in program['parameters'] if p['id'] == identity)
    return np.memmap(Path(directory) / 'coefficients.bin', mode='r', offset=parameter['offset'],
                     dtype='<f4' if parameter['storage'] == 'f32' else '<f8', shape=tuple(parameter['shape'])).copy()


def gated_function(program, name):
    """Recognize independently gated units with a linear readout, or reject."""
    require(name in {f['name'] for f in program['functions']}, 'unknown function')
    function = next(f for f in program['functions'] if f['name'] == name)
    live = function['live']
    required = {'input_weight', 'input_bias', 'output_weight', 'output_bias'}
    require(live is not None and set(live['bindings']) == required and live['initial'] == live['capacity'],
            'rewrite requires a fully active gated function with four declared captures')
    parameters = {p['id']: p for p in program['parameters']}; bindings = live['bindings']; capacity = live['capacity']
    wo = parameters[bindings['output_weight']]; hidden = wo['shape'][0]
    expected = {'input_weight': [2 * capacity, hidden], 'input_bias': [2 * capacity],
                'output_weight': [hidden, capacity], 'output_bias': [hidden]}
    require(all(parameters[bindings[k]]['shape'] == shape for k, shape in expected.items()), 'unsupported gated capture geometry')
    nodes = {n['id']: n for n in function['nodes']}
    sequence = ['parameter', 'parameter', 'slice', 'slice', 'transpose', 'contract', 'map',
                'slice', 'slice', 'transpose', 'contract', 'map', 'map', 'parameter', 'parameter',
                'slice', 'transpose', 'contract', 'map']
    require([n['op'] for n in function['nodes']] == sequence, 'unsupported gated function body')
    ordered = function['nodes']; ids = [n['id'] for n in ordered]; width = live['symbol']
    require(len(function['inputs']) == 1, 'gated rewrite requires one tensor argument')
    x = function['inputs'][0]['id']
    require(ordered[0]['attrs']['binding'] == bindings['input_weight'] and ordered[1]['attrs']['binding'] == bindings['input_bias'] and
            ordered[13]['attrs']['binding'] == bindings['output_weight'] and ordered[14]['attrs']['binding'] == bindings['output_bias'], 'capture/body mismatch')
    for offset, start in ((2, 0), (7, capacity)):
        for j in (0, 1):
            node = ordered[offset + j]
            require(node['inputs'] == [ids[j]] and node['attrs'] == {'axis': 0, 'start': start, 'size': width}, 'unsupported gated input slice')
        require(ordered[offset + 2]['inputs'] == [ids[offset]] and ordered[offset + 2]['attrs'] == {'axes': [1, 0]} and
                ordered[offset + 3]['inputs'] == [x, ids[offset + 2]] and
                ordered[offset + 4]['inputs'] == [ids[offset + 3], ids[offset + 1]], 'unsupported gated projection')
    require(ordered[12]['inputs'] == [ids[6], ids[11]] and
            ordered[15]['inputs'] == [ids[13]] and ordered[15]['attrs'] == {'axis': 1, 'start': 0, 'size': width} and
            ordered[16]['inputs'] == [ids[15]] and ordered[16]['attrs'] == {'axes': [1, 0]} and
            ordered[17]['inputs'] == [ids[12], ids[16]] and ordered[18]['inputs'] == [ids[17], ids[14]] and
            function['output'] == ids[18], 'unsupported gated readout')
    from spark.function_ir import arg, expr
    add = canonical(expr('add', arg(0), arg(1)))
    require(all(canonical(ordered[i]['attrs']['expression']) == add for i in (6, 11, 18)), 'unsupported affine expressions')
    def uses_index(value):
        if isinstance(value, dict): return value.get('op') == 'index' or any(uses_index(v) for v in value.values())
        if isinstance(value, list): return any(uses_index(v) for v in value)
        return False
    require(not any(uses_index(n['attrs']) for n in ordered), 'unit-index-dependent expressions cannot be permuted')
    return function, {'capacity': capacity, 'hidden': hidden, 'bindings': bindings, 'feature_value': ids[12]}


def _repack(program):
    cursor = 0
    for p in program['parameters']:
        p['offset'] = cursor; cursor += p['bytes']
    program['coefficient_bytes'] = cursor
    program.pop('coefficients_sha256', None)


def _reuse_native(destination, template, program):
    if template is None: return
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    (destination / 'program.json').write_text(json.dumps(program))
    shutil.copytree(Path(template) / 'native', destination / 'native', dirs_exist_ok=True)


def prune_gated_function(source, destination, name, keep, *, readout_weight=None, readout_bias=None, native_template=None):
    """Remove selected units physically, optionally installing a calibrated readout."""
    import numpy as np
    program = read_program(source); function, spec = gated_function(program, name)
    require(type(keep) is list and keep and all(type(i) is int and 0 <= i < spec['capacity'] for i in keep) and
            len(set(keep)) == len(keep), 'invalid retained units')
    readers = readers_for(program, source); n = len(keep); old = spec['capacity']; bindings = spec['bindings']
    captures = {key: parameter_array(program, source, pid) for key, pid in bindings.items()}
    rows = keep + [i + old for i in keep]
    captures['input_weight'] = captures['input_weight'][rows]
    captures['input_bias'] = captures['input_bias'][rows]
    captures['output_weight'] = captures['output_weight'][:, keep]
    if readout_weight is not None: captures['output_weight'] = np.asarray(readout_weight)
    if readout_bias is not None: captures['output_bias'] = np.asarray(readout_bias)
    shapes = {'input_weight': (2*n, spec['hidden']), 'input_bias': (2*n,), 'output_weight': (spec['hidden'], n), 'output_bias': (spec['hidden'],)}
    for key, data in captures.items():
        require(data.shape == shapes[key] and np.isfinite(data).all(), 'invalid replacement coefficient array')
        parameter = next(p for p in program['parameters'] if p['id'] == bindings[key])
        data = np.asarray(data, dtype='<f4' if parameter['storage'] == 'f32' else '<f8', order='C')
        require(np.isfinite(data).all(), 'replacement overflows coefficient storage')
        parameter['shape'] = list(data.shape); parameter['bytes'] = data.nbytes
        readers[parameter['id']] = data.tobytes()
    input_parameters = {bindings['input_weight'], bindings['input_bias']}
    nodes = {node['id']: node for node in function['nodes']}
    for node in function['nodes']:
        if node['op'] == 'slice' and node['attrs']['start'] == old and nodes[node['inputs'][0]]['attrs'].get('binding') in input_parameters:
            node['attrs']['start'] = n
    function['live']['capacity'] = n; function['live']['initial'] = n
    program.setdefault('metadata', {}).update(approximation=True, rewrite={'kind': 'physical-gated-unit-pruning', 'function': name,
                'original_capacity': old, 'retained_units': keep, 'readout_calibrated': readout_weight is not None})
    _repack(program); validate(program)
    _reuse_native(destination, native_template, program)
    tokenizer = Path(source) / 'tokenizer'
    return write_package(program, readers, destination, tokenizer=tokenizer if tokenizer.exists() else None)


def extract_function(source, destination, name):
    program = read_program(source); function, _ = gated_function(program, name)
    readers = readers_for(program, source)
    inputs = copy.deepcopy(function['inputs'])
    used_symbols = {d for value in inputs for d in value['shape'] if type(d) is str}
    program['symbols'] = {k: v for k, v in program['symbols'].items() if k in used_symbols}
    program['main'] = {'inputs': inputs, 'nodes': [{'id': 'result', 'op': 'call', 'inputs': [v['id'] for v in inputs], 'attrs': {'function': name}}],
                       'outputs': {'value': 'result'}, 'preconditions': []}
    program['metadata'] = {'task': 'isolated-function', 'approximation': program.get('metadata', {}).get('approximation', False)}
    return write_package(program, readers, destination)


def calibration_program(source, destination, name):
    program = read_program(source); function, spec = gated_function(program, name)
    readers = readers_for(program, source)
    feature_function = copy.deepcopy(function); feature_function['name'] = 'observed_features'; feature_function['live'] = None
    def freeze(value):
        if isinstance(value, dict): return {k: freeze(v) for k, v in value.items()}
        if isinstance(value, list): return [freeze(v) for v in value]
        return spec['capacity'] if value == function['live']['symbol'] else value
    feature_function['nodes'] = freeze(feature_function['nodes']); feature_function['output'] = spec['feature_value']
    require(feature_function['name'] not in {f['name'] for f in program['functions']}, 'calibration function name collision')
    program['functions'].append(feature_function)
    calls = [node for node in program['main']['nodes'] if node['op'] == 'call' and node['attrs']['function'] == name]
    require(len(calls) == 1, 'calibration requires a single explicit invocation')
    call = calls[0]
    identity = 'observed_feature_values'
    require(identity not in {n['id'] for n in program['main']['nodes']}, 'calibration SSA name collision')
    program['main']['nodes'].append({'id': identity, 'op': 'call', 'inputs': call['inputs'], 'attrs': {'function': feature_function['name']}})
    program['main']['outputs'] = {'arguments': call['inputs'][0], 'features': identity, 'target': call['id']}
    program['metadata'] = {'task': 'function-calibration', 'approximation': False}
    return write_package(program, readers, destination)

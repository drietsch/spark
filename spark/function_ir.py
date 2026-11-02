"""SPARK's source-independent tensor/function language and native compiler.

Programs are SSA function compositions. Shapes, indexed scalar expressions,
reductions, contractions, bounded maps, and mutable function captures have
explicit semantics. No operation denotes a source model or a neural layer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

FORMAT = 'spark-function-program-v1'
IDENTIFIER = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}\Z')
DTYPES = ('f64', 'index', 'bool')
MAX_ELEMENTS = 128 * 2**20


class CompileError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CompileError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def literal(value):
    return {'op': 'literal', 'value': value}


def arg(index):
    return {'op': 'arg', 'index': index}


def expr(op, *args):
    return {'op': op, 'args': list(args)}


def _shape(shape, symbols):
    require(type(shape) is list and 1 <= len(shape) <= 4, 'tensor rank must be 1..4')
    for dim in shape:
        require((type(dim) is int and 1 <= dim <= 131072) or
                (type(dim) is str and dim in symbols), f'invalid shape dimension {dim!r}')
    return list(shape)


def _broadcast(shapes):
    out = [1] * max(map(len, shapes))
    for shape in shapes:
        for i, dim in enumerate(reversed(shape)):
            old = out[-1-i]
            require(old == 1 or dim == 1 or old == dim, 'incompatible elementwise broadcast')
            if dim != 1:
                out[-1-i] = dim
    return out


ARITIES = {'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'gt': 2, 'lt': 2,
           'eq': 2, 'and': 2, 'neg': 1, 'abs': 1, 'exp': 1, 'sqrt': 1,
           'erf': 1, 'tanh': 1, 'isfinite': 1, 'where': 3}


def _expression(value, inputs, rank, depth=0):
    require(type(value) is dict and depth <= 32, 'invalid scalar expression/depth')
    op = value.get('op')
    if op == 'literal':
        require(set(value) == {'op', 'value'} and
                ((type(value['value']) in (int, float) and math.isfinite(value['value'])) or value['value'] == '-inf'),
                'invalid scalar literal')
    elif op in ('arg', 'index'):
        limit = inputs if op == 'arg' else rank
        require(set(value) == {'op', 'index'} and type(value['index']) is int and 0 <= value['index'] < limit,
                'invalid scalar argument/index')
    else:
        require(op in ARITIES and set(value) == {'op', 'args'} and type(value['args']) is list and
                len(value['args']) == ARITIES[op], f'unsupported scalar operation {op!r}')
        for child in value['args']:
            _expression(child, inputs, rank, depth + 1)


def validate(program):
    """Type/shape check every reachable and unreachable function before codegen."""
    require(type(program) is dict and program.get('format') == FORMAT, 'invalid function program format')
    require(set(program) <= {'format', 'symbols', 'parameters', 'functions', 'main', 'source', 'metadata', 'coefficients_sha256', 'coefficient_bytes', 'optimizations'},
            'unknown program fields')
    symbols = program.get('symbols')
    require(type(symbols) is dict and len(symbols) <= 8, 'invalid shape symbols')
    for name, bounds in symbols.items():
        require(type(name) is str and IDENTIFIER.fullmatch(name) and type(bounds) is list and len(bounds) == 2 and
                all(type(n) is int for n in bounds) and 1 <= bounds[0] <= bounds[1] <= 8192, 'invalid symbol bounds')
    params = program.get('parameters')
    require(type(params) is list and len(params) <= 8192, 'invalid parameter count')
    parameter_types = {}; cursor = 0
    for parameter in params:
        require(type(parameter) is dict and set(parameter) == {'id', 'shape', 'offset', 'bytes', 'mutable', 'storage'}, 'invalid coefficient binding')
        name = parameter['id']
        require(type(name) is str and IDENTIFIER.fullmatch(name) and name not in parameter_types, 'invalid/duplicate parameter identifier')
        shape = _shape(parameter['shape'], {})
        require(math.prod(shape) <= MAX_ELEMENTS, 'coefficient tensor allocation budget exceeded')
        require(parameter['storage'] in ('f32', 'f64'), 'invalid coefficient precision')
        require(parameter['offset'] == cursor and type(parameter['offset']) is int and
                type(parameter['bytes']) is int and parameter['bytes'] == (4 if parameter['storage'] == 'f32' else 8) * math.prod(shape) and
                type(parameter['mutable']) is bool, 'invalid contiguous coefficient layout')
        cursor += parameter['bytes']; parameter_types[name] = ('f64', shape)
    require(cursor <= 2 * 2**30 and program.get('coefficient_bytes', cursor) == cursor, 'coefficient allocation budget exceeded')
    functions = program.get('functions')
    require(type(functions) is list and len(functions) <= 256, 'invalid function table')
    names = [f.get('name') if type(f) is dict else None for f in functions]
    require(all(type(n) is str and IDENTIFIER.fullmatch(n) for n in names) and len(set(names)) == len(names), 'invalid function names')
    verified = {}; types = {}

    def check_function(function, main=False):
        fields = {'inputs', 'nodes', 'outputs', 'preconditions'} if main else {'name', 'inputs', 'nodes', 'output', 'live'}
        require(type(function) is dict and set(function) == fields, 'invalid function definition fields')
        local_symbols = dict(symbols)
        if not main and function['live'] is not None:
            live = function['live']
            require(type(live) is dict and set(live) == {'symbol', 'capacity', 'initial', 'bindings'}, 'invalid live function definition')
            require(type(live['symbol']) is str and IDENTIFIER.fullmatch(live['symbol']) and live['symbol'] not in symbols and
                    type(live['capacity']) is int and type(live['initial']) is int and 1 <= live['initial'] <= live['capacity'] <= 8192,
                    'invalid active function width')
            require(type(live['bindings']) is dict and all(type(k) is str and IDENTIFIER.fullmatch(k) and
                    v in parameter_types and params[list(parameter_types).index(v)]['mutable'] for k, v in live['bindings'].items()),
                    'live functions may only update declared mutable captures')
            local_symbols[live['symbol']] = [1, live['capacity']]
        inputs = function['inputs']; nodes = function['nodes']
        require(type(inputs) is list and 1 <= len(inputs) <= 16 and type(nodes) is list and len(nodes) <= 8192, 'invalid function size')
        values = {}
        for value in inputs:
            require(type(value) is dict and set(value) == {'id', 'dtype', 'shape'} and type(value['id']) is str and
                    IDENTIFIER.fullmatch(value['id']) and value['id'] not in values and value['dtype'] in DTYPES, 'invalid typed function input')
            values[value['id']] = (value['dtype'], _shape(value['shape'], local_symbols))
        for node in nodes:
            require(type(node) is dict and set(node) == {'id', 'op', 'inputs', 'attrs'} and type(node['id']) is str and
                    IDENTIFIER.fullmatch(node['id']) and node['id'] not in values and type(node['inputs']) is list and
                    all(type(i) is str and i in values for i in node['inputs']) and type(node['attrs']) is dict,
                    'invalid SSA node, forward reference, or duplicate definition')
            op, attrs = node['op'], node['attrs']; operands = [values[n] for n in node['inputs']]
            shapes = [v[1] for v in operands]; dtype = 'f64'
            if op == 'parameter':
                require(not operands and set(attrs) == {'binding'} and attrs['binding'] in parameter_types, 'invalid parameter reference')
                dtype, shape = parameter_types[attrs['binding']]
            elif op == 'map':
                require(operands and set(attrs) == {'expression'}, 'invalid indexed scalar map')
                shape = _broadcast(shapes); _expression(attrs['expression'], len(operands), len(shape))
            elif op == 'reduce':
                require(len(operands) == 1 and set(attrs) == {'axis', 'kind'} and type(attrs['axis']) is int and
                        0 <= attrs['axis'] < len(shapes[0]) and attrs['kind'] in ('sum', 'max'), 'invalid reduction')
                shape = list(shapes[0]); shape[attrs['axis']] = 1
            elif op == 'transpose':
                require(len(operands) == 1 and set(attrs) == {'axes'} and type(attrs['axes']) is list and
                        all(type(a) is int for a in attrs['axes']) and sorted(attrs['axes']) == list(range(len(shapes[0]))), 'invalid permutation')
                shape = [shapes[0][a] for a in attrs['axes']]; dtype = operands[0][0]
            elif op == 'reshape':
                require(len(operands) == 1 and set(attrs) == {'shape'}, 'invalid reshape')
                shape = _shape(attrs['shape'], local_symbols)
                def product_signature(dims):
                    return math.prod(d for d in dims if type(d) is int), sorted(d for d in dims if type(d) is str)
                require(product_signature(shape) == product_signature(shapes[0]), 'reshape changes element count')
                dtype = operands[0][0]
            elif op == 'slice':
                require(len(operands) == 1 and set(attrs) == {'axis', 'start', 'size'} and type(attrs['axis']) is int and
                        0 <= attrs['axis'] < len(shapes[0]) and type(attrs['start']) is int and attrs['start'] >= 0,
                        'invalid slice')
                shape = list(shapes[0]); size = _shape([attrs['size']], local_symbols)[0]; dim = shape[attrs['axis']]
                upper = local_symbols[size][1] if type(size) is str else size
                lower = local_symbols[dim][0] if type(dim) is str else dim
                require(attrs['start'] + upper <= lower, 'slice exceeds tensor bounds')
                shape[attrs['axis']] = size; dtype = operands[0][0]
            elif op == 'concat':
                require(len(operands) == 2 and set(attrs) == {'axis'} and type(attrs['axis']) is int and
                        len(shapes[0]) == len(shapes[1]) and 0 <= attrs['axis'] < len(shapes[0]), 'invalid concatenation')
                axis = attrs['axis']; shape = list(shapes[0])
                require(all(a == b for i, (a, b) in enumerate(zip(*shapes)) if i != axis) and
                        type(shape[axis]) is int and type(shapes[1][axis]) is int, 'incompatible concatenation shapes')
                shape[axis] += shapes[1][axis]
            elif op == 'gather':
                require(len(operands) == 2 and not attrs and len(shapes[0]) == 2 and len(shapes[1]) == 1 and operands[1][0] == 'index', 'invalid row gather')
                shape = [shapes[1][0], shapes[0][1]]
            elif op == 'contract':
                require(len(operands) == 2 and not attrs and all(len(s) == 2 for s in shapes) and shapes[0][1] == shapes[1][0], 'invalid sum-product contraction')
                shape = [shapes[0][0], shapes[1][1]]
            elif op in ('call', 'map_call'):
                require(set(attrs) == ({'function'} if op == 'call' else {'function', 'axes'}) and attrs['function'] in verified,
                        'unknown, recursive, or forward function call')
                target = verified[attrs['function']]
                require(len(operands) == len(target['inputs']), 'function argument count mismatch')
                axes = [None] * len(operands) if op == 'call' else attrs['axes']
                require(type(axes) is list and len(axes) == len(operands) and
                        all(a is None or type(a) is int and 0 <= a < len(s) for a, s in zip(axes, shapes)), 'invalid bounded map axes')
                mapped = [s[a] for a, s in zip(axes, shapes) if a is not None]
                require(op != 'map_call' or mapped and all(d == mapped[0] for d in mapped), 'inconsistent bounded map extents')
                for (actual_dtype, actual_shape), axis, wanted in zip(operands, axes, target['inputs']):
                    actual_shape = list(actual_shape)
                    if axis is not None: actual_shape.pop(axis)
                    require((actual_dtype, actual_shape) == (wanted['dtype'], wanted['shape']), 'function argument type mismatch')
                dtype, shape = types[attrs['function']][target['output']]
                shape = ([mapped[0]] if op == 'map_call' else []) + list(shape)
            else:
                raise CompileError(f'unsupported target operation {op!r}')
            _shape(shape, local_symbols)
            values[node['id']] = (dtype, shape)
        output_ids = function['outputs'].values() if main else [function['output']]
        if main:
            require(type(function['outputs']) is dict and 1 <= len(function['outputs']) <= 16 and
                    all(type(k) is str and IDENTIFIER.fullmatch(k) for k in function['outputs']), 'invalid named outputs')
            _preconditions(function['preconditions'], {v['id']: v for v in inputs}, symbols)
        require(all(o in values for o in output_ids), 'undefined function result')
        if not main and function['live']:
            width_symbol = function['live']['symbol']
            require(all(width_symbol not in value['shape'] for value in inputs) and width_symbol not in values[function['output']][1],
                    'live width must remain internal to a fixed function interface')
        return values

    for function in functions:
        types[function['name']] = check_function(function)
        verified[function['name']] = function
    types['main'] = check_function(program.get('main'), main=True)
    return types


def _preconditions(conditions, inputs, symbols):
    require(type(conditions) is list and len(conditions) <= 64, 'invalid input contract')
    for condition in conditions:
        require(type(condition) is dict, 'invalid precondition')
        op = condition.get('op')
        if op == 'bounds':
            require(set(condition) == {'op', 'input', 'upper'} and condition['input'] in inputs and
                    inputs[condition['input']]['dtype'] == 'index', 'invalid index bounds contract')
            _shape([condition['upper']], symbols)
        elif op in ('unique', 'any'):
            require(set(condition) == {'op', 'input'} and condition['input'] in inputs, 'invalid input predicate')
        elif op == 'selected_true':
            require(set(condition) == {'op', 'input', 'indices'} and condition['input'] in inputs and
                    condition['indices'] in inputs and inputs[condition['indices']]['dtype'] == 'index', 'invalid selection predicate')
        else:
            raise CompileError(f'unsupported input contract {op!r}')


def optimize(program):
    """Conservative CSE + dead SSA/function/capture elimination; no FP reassociation."""
    validate(program); result = copy.deepcopy(program); removed = 0
    for function in [*result['functions'], result['main']]:
        aliases = {}; seen = {}; nodes = []
        for original in function['nodes']:
            node = copy.deepcopy(original); node['inputs'] = [aliases.get(n, n) for n in node['inputs']]
            key = canonical([node['op'], node['inputs'], node['attrs']])
            if key in seen:
                aliases[node['id']] = seen[key]; removed += 1
            else:
                seen[key] = node['id']; nodes.append(node)
        if 'outputs' in function:
            function['outputs'] = {k: aliases.get(v, v) for k, v in function['outputs'].items()}
            needed = set(function['outputs'].values())
        else:
            function['output'] = aliases.get(function['output'], function['output']); needed = {function['output']}
        retained = []
        for node in reversed(nodes):
            if node['id'] in needed:
                retained.append(node); needed.update(node['inputs'])
            else:
                removed += 1
        function['nodes'] = list(reversed(retained))
    reachable = set()
    for function in reversed([*result['functions'], result['main']]):
        if function is result['main'] or function['name'] in reachable:
            reachable.update(n['attrs']['function'] for n in function['nodes'] if n['op'] in ('call', 'map_call'))
    result['functions'] = [f for f in result['functions'] if f['name'] in reachable]
    used = {n['attrs']['binding'] for f in [*result['functions'], result['main']] for n in f['nodes'] if n['op'] == 'parameter'}
    for f in result['functions']:
        if f['live']: used.update(f['live']['bindings'].values())
    result['parameters'] = [p for p in result['parameters'] if p['id'] in used]
    cursor = 0
    for p in result['parameters']:
        p['offset'] = cursor; cursor += p['bytes']
    result['coefficient_bytes'] = cursor
    result['optimizations'] = {'common_and_dead_nodes_removed': removed, 'unused_parameters_removed': len(program['parameters']) - len(result['parameters'])}
    result.pop('coefficients_sha256', None)
    validate(result)
    return result


class Builder:
    def __init__(self, inputs):
        self.inputs = inputs; self.nodes = []

    def node(self, op, *inputs, **attrs):
        name = f'v{len(self.nodes)}'
        self.nodes.append({'id': name, 'op': op, 'inputs': list(inputs), 'attrs': attrs})
        return name

    def parameter(self, name):
        return self.node('parameter', binding=name)

    def map(self, expression, *inputs):
        return self.node('map', *inputs, expression=expression)

    def function(self, name, output, live=None):
        return {'name': name, 'inputs': self.inputs, 'nodes': self.nodes, 'output': output, 'live': live}


def _cpp_expression(value):
    op = value['op']
    if op == 'literal':
        return '-std::numeric_limits<double>::infinity()' if value['value'] == '-inf' else repr(float(value['value']))
    if op == 'arg': return f'a[{value["index"]}]'
    if op == 'index': return f'double(index[{value["index"]}])'
    args = [_cpp_expression(v) for v in value['args']]
    if op in ('add', 'sub', 'mul', 'div', 'gt', 'lt', 'eq', 'and'):
        return '(' + args[0] + {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'gt': '>', 'lt': '<', 'eq': '==', 'and': '&&'}[op] + args[1] + ')'
    if op == 'neg': return '(-' + args[0] + ')'
    if op == 'where': return '(' + args[0] + '?' + args[1] + ':' + args[2] + ')'
    return 'std::' + {'abs': 'abs', 'isfinite': 'isfinite'}.get(op, op) + '(' + args[0] + ')'


def emit_cpp(program):
    types = validate(program)
    functions = program['functions']; parameter_index = {p['id']: i for i, p in enumerate(program['parameters'])}
    function_index = {f['name']: i for i, f in enumerate(functions)}
    symbols = list(program['symbols'])
    lines = ['#include "runtime/function_ir_runtime.hpp"', 'using namespace spark_function;', 'namespace {']
    def dimension(value):
        return 'shape_' + value if type(value) is str else str(value)

    def emit(function, main=False):
        name = 'main' if main else function['name']; inputs = function['inputs']
        cpp_name = 'evaluate_main' if main else 'evaluate_function_' + name
        lines.append(('std::vector<Tensor>' if main else 'Tensor') + f' {cpp_name}(Context& c,const std::vector<Tensor>& in,const int64_t* symbols) {{')
        for i, symbol in enumerate(symbols): lines.append(f'const int64_t {dimension(symbol)}=symbols[{i}];')
        if not main and function['live']:
            lines.append(f'const int64_t {dimension(function["live"]["symbol"])}=c.widths[{function_index[name]}];')
        ids = [v['id'] for v in inputs] + [n['id'] for n in function['nodes']]
        indices = {v: i for i, v in enumerate(ids)}
        lines.append(f'std::vector<Tensor> r({len(ids)});')
        for i in range(len(inputs)): lines.append(f'r[{i}]=in[{i}];')
        def ref(v): return f'r[{indices[v]}]'
        def dims(ds): return '{' + ','.join(dimension(v) for v in ds) + '}'
        final = list(function['outputs'].values()) if main else [function['output']]
        last = {v: i for i, node in enumerate(function['nodes']) for v in node['inputs']}
        for v in final: last[v] = len(function['nodes'])
        for i, node in enumerate(function['nodes']):
            op, a = node['op'], node['attrs']; operands = [ref(v) for v in node['inputs']]
            if op == 'parameter': expression = f'c.parameter({parameter_index[a["binding"]]})'
            elif op == 'map':
                expression = 'mapped({' + ','.join(operands) + '},[](const double* a,const int64_t* index)->double{return ' + _cpp_expression(a['expression']) + ';})'
            elif op == 'reduce': expression = f'reduced({operands[0]},{a["axis"]},{str(a["kind"] == "max").lower()})'
            elif op == 'transpose': expression = f'transposed({operands[0]},{dims(a["axes"])})'
            elif op == 'reshape': expression = f'reshaped({operands[0]},{dims(a["shape"])})'
            elif op == 'slice': expression = f'sliced({operands[0]},{a["axis"]},{a["start"]},{dimension(a["size"])})'
            elif op == 'concat': expression = f'concatenated({operands[0]},{operands[1]},{a["axis"]})'
            elif op == 'gather': expression = f'gathered({operands[0]},{operands[1]})'
            elif op == 'contract': expression = f'contracted({operands[0]},{operands[1]})'
            elif op == 'call': expression = f'evaluate_function_{a["function"]}(c,{{' + ','.join(operands) + '},symbols)'
            elif op == 'map_call':
                axes = dims([-1 if v is None else v for v in a['axes']])
                expression = 'mapped_call({' + ','.join(operands) + '},' + axes + ',[&](const std::vector<Tensor>& args){return evaluate_function_' + a['function'] + '(c,args,symbols);})'
            lines.append(ref(node['id']) + '=' + expression + ';')
            if node['id'] not in last: lines.append(ref(node['id']) + '=Tensor();')
            for v in set(node['inputs']):
                if last[v] == i: lines.append(ref(v) + '=Tensor();')
        lines.append('return {' + ','.join(ref(v) for v in final) + '};' if main else 'return ' + ref(final[0]) + ';')
        lines.append('}')
    for f in functions: emit(f)
    emit(program['main'], main=True)
    lines.append('}')
    specs = []
    for p in program['parameters']:
        specs.append('{' + str(p['offset']) + ',' + str(math.prod(p['shape'])) + ',{' + ','.join(map(str, p['shape'])) + '},' + str(p['mutable']).lower() + ',' + ('4' if p['storage'] == 'f32' else '8') + '}')
    widths = [f['live']['initial'] if f['live'] else 0 for f in functions]
    capacities = [f['live']['capacity'] if f['live'] else 0 for f in functions]
    lines += [
        'extern "C" const char* spark_fn_error(){return last_error.c_str();}',
        'extern "C" void* spark_fn_load(const char* path){try{last_error.clear();return new Context(path,{' + ','.join(specs) + '},{' + ','.join(map(str, widths)) + '},{' + ','.join(map(str, capacities)) + '});}catch(const std::exception& e){last_error=e.what();return nullptr;}}',
        'extern "C" void spark_fn_destroy(void* p){delete static_cast<Context*>(p);}',
        'extern "C" int spark_fn_width(void* p,int function,int width){try{ensure(p,"program closed");static_cast<Context*>(p)->width(function,width);return 1;}catch(const std::exception& e){last_error=e.what();return 0;}}',
        'extern "C" int spark_fn_coefficient(void* p,int parameter,uint64_t index,double value){try{ensure(p,"program closed");static_cast<Context*>(p)->coefficient(parameter,index,value);return 1;}catch(const std::exception& e){last_error=e.what();return 0;}}',
        'extern "C" int spark_fn_run(void* p,const double* const* inputs,const int64_t* symbols,double* const* outputs){try{ensure(p&&inputs&&symbols&&outputs,"missing program arguments");',
    ]
    for i, (symbol, bounds) in enumerate(program['symbols'].items()):
        name = dimension(symbol)
        lines.append(f'const int64_t {name}=symbols[{i}];ensure({name}>={bounds[0]}&&{name}<={bounds[1]},"shape symbol outside bounds");')
    lines.append('std::vector<Tensor> in;')
    for i, value in enumerate(program['main']['inputs']):
        ds = '{' + ','.join(dimension(d) for d in value['shape']) + '}'
        lines.append(f'in.push_back(input_tensor(inputs[{i}],{ds},{DTYPES.index(value["dtype"])}));')
    input_index = {v['id']: i for i, v in enumerate(program['main']['inputs'])}
    for condition in program['main']['preconditions']:
        v = input_index[condition['input']]; op = condition['op']
        if op == 'bounds': lines.append(f'check_bounds(in[{v}],{dimension(condition["upper"])});')
        elif op == 'unique': lines.append(f'check_unique(in[{v}]);')
        elif op == 'any': lines.append(f'check_any(in[{v}]);')
        else: lines.append(f'check_selected(in[{v}],in[{input_index[condition["indices"]]}]);')
    lines += ['auto out=evaluate_main(*static_cast<Context*>(p),in,symbols);',
              'for(size_t i=0;i<out.size();++i){ensure(outputs[i],"missing output buffer");copy_output(out[i],outputs[i]);}return 1;',
              '}catch(const std::exception& e){last_error=e.what();return 0;}}']
    return '\n'.join(lines) + '\n'


def compile_program(directory, *, force=False):
    """Generate native code from the typed program, then compile a content-keyed library."""
    directory = Path(directory)
    program = json.loads((directory / 'program.json').read_text()); validate(program)
    require(digest(directory / 'coefficients.bin') == program['coefficients_sha256'], 'coefficient checksum mismatch')
    source = emit_cpp(program)
    header = directory / 'runtime/function_ir_runtime.hpp'
    if not header.exists():
        header.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'runtime/function_ir_runtime.hpp', header)
    key = hashlib.sha256((source + digest(header)).encode()).hexdigest()
    cpp = directory / 'program.cpp'; cpp.write_text(source)
    native = directory / 'native'; native.mkdir(exist_ok=True)
    library = native / f'function_{key[:16]}.so'; receipt = library.with_suffix('.json')
    if not force and library.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved.get('compiler_input_sha256') == key and saved.get('library_sha256') == digest(library):
            return library
    compiler = shutil.which('c++') or shutil.which('g++')
    require(compiler is not None, 'C++17 compiler required')
    flags = ['-std=c++17', '-O2', '-ffp-contract=off', '-fPIC', '-shared']
    try:
        subprocess.run([compiler, *flags, str(cpp), '-o', str(library), '-l:libblas.so.3'], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise CompileError('native compilation failed: ' + error.stderr[-4000:]) from error
    receipt.write_text(json.dumps({'compiler_input_sha256': key, 'library_sha256': digest(library), 'flags': flags}, indent=2) + '\n')
    return library

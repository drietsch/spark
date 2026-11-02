"""Load and execute compiled SPARK function programs using only the stdlib."""
from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import threading

from spark.function_ir import MAX_ELEMENTS, compile_program, digest, require, validate


def _flatten(value):
    if type(value) not in (list, tuple) or not value:
        raise ValueError('inputs must be nonempty rectangular lists')
    if type(value[0]) in (list, tuple):
        rows = [_flatten(row) for row in value]
        if any(shape != rows[0][1] for _, shape in rows):
            raise ValueError('ragged tensor input')
        return [v for row, _ in rows for v in row], [len(value), *rows[0][1]]
    if any(type(v) not in (int, float, bool) or not math.isfinite(v) for v in value):
        raise ValueError('inputs contain invalid numerical values')
    return list(value), [len(value)]


class NativeFunctionProgram:
    def __init__(self, directory):
        self._lock = threading.RLock()
        self.directory = Path(directory)
        self.program = json.loads((self.directory / 'program.json').read_text())
        self.types = validate(self.program)
        require((self.directory / 'coefficients.bin').stat().st_size == self.program['coefficient_bytes'], 'coefficient payload size mismatch')
        require(digest(self.directory / 'coefficients.bin') == self.program['coefficients_sha256'], 'coefficient checksum mismatch')
        path = compile_program(self.directory)
        try:
            self.lib = ctypes.CDLL(str(path.resolve()))
        except OSError:
            self.lib = ctypes.CDLL(str(compile_program(self.directory, force=True).resolve()))
        library = self.lib; pointer = ctypes.c_void_p; doubles = ctypes.POINTER(ctypes.c_double)
        library.spark_fn_error.restype = ctypes.c_char_p
        library.spark_fn_load.argtypes = [ctypes.c_char_p]; library.spark_fn_load.restype = pointer
        library.spark_fn_destroy.argtypes = [pointer]
        library.spark_fn_run.argtypes = [pointer, ctypes.POINTER(doubles), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(doubles)]
        library.spark_fn_width.argtypes = [pointer, ctypes.c_int, ctypes.c_int]
        library.spark_fn_coefficient.argtypes = [pointer, ctypes.c_int, ctypes.c_uint64, ctypes.c_double]
        self.handle = library.spark_fn_load(bytes(self.directory / 'coefficients.bin'))
        if not self.handle:
            raise ValueError(self._error())
        self.functions = {f['name']: (i, f) for i, f in enumerate(self.program['functions'])}
        self.parameters = {p['id']: (i, p) for i, p in enumerate(self.program['parameters'])}

    def _error(self):
        return (self.lib.spark_fn_error() or b'native function execution failed').decode(errors='replace')

    def run(self, inputs):
        if type(inputs) is not dict or set(inputs) != {v['id'] for v in self.program['main']['inputs']}:
            raise ValueError('input names do not match function signature')
        symbols = {}; arrays = []
        for spec in self.program['main']['inputs']:
            values, shape = _flatten(inputs[spec['id']])
            if len(values) > MAX_ELEMENTS or len(shape) != len(spec['shape']):
                raise ValueError('input rank/allocation budget mismatch')
            for actual, declared in zip(shape, spec['shape']):
                if type(declared) is int:
                    if actual != declared: raise ValueError('input shape mismatch')
                else:
                    bounds = self.program['symbols'][declared]
                    if not bounds[0] <= actual <= bounds[1] or declared in symbols and symbols[declared] != actual:
                        raise ValueError('inconsistent or out-of-bounds shape symbol')
                    symbols[declared] = actual
            if spec['dtype'] == 'index' and any(type(v) is not int for v in values):
                raise ValueError('discrete inputs require integer indices')
            if spec['dtype'] == 'bool' and any(type(v) not in (int, bool) or v not in (0, 1) for v in values):
                raise ValueError('Boolean inputs require zero/one values')
            arrays.append((ctypes.c_double * len(values))(*values))
        if set(symbols) != set(self.program['symbols']):
            raise ValueError('shape symbols must be bound by inputs')
        dims = (ctypes.c_int64 * len(symbols))(*(symbols[s] for s in self.program['symbols']))
        outputs = []; shapes = []
        for value in self.program['main']['outputs'].values():
            shape = [symbols[d] if type(d) is str else d for d in self.types['main'][value][1]]
            size = math.prod(shape)
            if size > MAX_ELEMENTS: raise ValueError('output allocation budget exceeded')
            outputs.append((ctypes.c_double * size)()); shapes.append(shape)
        pointers = ctypes.POINTER(ctypes.c_double)
        input_ptrs = (pointers * len(arrays))(*(ctypes.cast(a, pointers) for a in arrays))
        output_ptrs = (pointers * len(outputs))(*(ctypes.cast(a, pointers) for a in outputs))
        with self._lock:
            if not self.handle: raise RuntimeError('program is closed')
            if not self.lib.spark_fn_run(self.handle, input_ptrs, dims, output_ptrs):
                raise ValueError(self._error())
        return {name: {'values': values, 'shape': shape} for name, values, shape in
                zip(self.program['main']['outputs'], outputs, shapes)}

    def set_active_width(self, function, width):
        if type(function) is not str or function not in self.functions or type(width) is not int:
            raise ValueError('unknown function/invalid width')
        live = self.functions[function][1]['live']
        if live is None or not 1 <= width <= live['capacity']:
            raise ValueError('active width outside function capacity')
        with self._lock:
            if not self.lib.spark_fn_width(self.handle, self.functions[function][0], width):
                raise ValueError(self._error())

    def set_function_coefficient(self, function, binding, index, value):
        if type(function) is not str or function not in self.functions or type(binding) is not str or type(index) is not int or index < 0 or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('invalid function coefficient update')
        live = self.functions[function][1]['live']
        if live is None or binding not in live['bindings']:
            raise ValueError('unknown mutable function binding')
        parameter, spec = self.parameters[live['bindings'][binding]]
        if index >= math.prod(spec['shape']):
            raise ValueError('coefficient index outside function capture')
        with self._lock:
            if not self.lib.spark_fn_coefficient(self.handle, parameter, index, value):
                raise ValueError(self._error())

    def close(self):
        with self._lock:
            if getattr(self, 'handle', None):
                self.lib.spark_fn_destroy(self.handle); self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

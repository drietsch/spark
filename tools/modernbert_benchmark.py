#!/usr/bin/env python3
"""M4: audit teacher trace holdout and benchmark SPARK Python and native CPU.

Requires a previously compiled ``prefix.encoder.{json,senc}`` and
``prefix.smlm``. This never loads ModernBERT: real model execution is handled
by tools/modernbert_teacher.py, which must be run separately.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.benchmark import (audit_traces, evaluate_predictions, latency_summary,
                             majority_baseline, teacher_top1)
from spark.modernbert import load_index, read_traces
from spark.semantic import SemanticEncoder, _target_code


class NativeStudent:
    """Keep both native handles alive across a timed evaluation."""
    def __init__(self, prefix: Path, build: Path):
        self.semantic = ctypes.CDLL(str((build / 'libspark_semantic_runtime.so').resolve()))
        self.probe = ctypes.CDLL(str((build / 'libspark_mlm_runtime.so').resolve()))
        self.semantic.spark_semantic_encoder_open_v1.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.semantic.spark_semantic_encoder_open_v1.restype = ctypes.c_int32
        self.semantic.spark_semantic_encoder_close_v1.argtypes = [ctypes.c_void_p]
        self.semantic.spark_semantic_predict_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                                                           ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                                           ctypes.c_uint64, ctypes.c_uint64]
        self.semantic.spark_semantic_predict_v1.restype = ctypes.c_int32
        self.probe.spark_mlm_index_open_v1.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.probe.spark_mlm_index_open_v1.restype = ctypes.c_int32
        self.probe.spark_mlm_index_close_v1.argtypes = [ctypes.c_void_p]
        self.encoder = ctypes.c_void_p()
        self.index = ctypes.c_void_p()
        enc_path = str(prefix) + '.encoder.senc'
        idx_path = str(prefix) + '.smlm'
        status = self.semantic.spark_semantic_encoder_open_v1(os.fsencode(enc_path), ctypes.byref(self.encoder))
        if status < 0:
            raise RuntimeError(f'loading native encoder failed: {status}')
        status = self.probe.spark_mlm_index_open_v1(os.fsencode(idx_path), ctypes.byref(self.index))
        if status < 0:
            self.close()
            raise RuntimeError(f'loading native index failed: {status}')

    def predict(self, ids: list[int], mask: int) -> int | None:
        tokens = (ctypes.c_uint32 * len(ids))(*ids)
        candidates = (ctypes.c_int32 * 64)()
        scores = (ctypes.c_int32 * 64)()
        status = self.semantic.spark_semantic_predict_v1(
            self.encoder.value, self.index.value, ctypes.addressof(tokens), len(ids),
            mask, 64, ctypes.addressof(candidates), ctypes.addressof(scores))
        if status < 0:
            raise RuntimeError(f'native prediction failed: status={status}')
        return candidates[0] if status else None

    def close(self):
        if self.index.value:
            self.probe.spark_mlm_index_close_v1(self.index)
            self.index = ctypes.c_void_p()
        if self.encoder.value:
            self.semantic.spark_semantic_encoder_close_v1(self.encoder)
            self.encoder = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_benchmark(traces: Path, prefix: Path, build: Path, *, repeat: int = 5,
                  allow_synthetic: bool = False, require_native_parity: bool = False) -> dict:
    if repeat < 1 or repeat > 1000:
        raise ValueError('repeat must be in 1..1000')
    rows = read_traces(traces)
    audit = audit_traces(rows, allow_synthetic=allow_synthetic)
    training = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    encoder_path = Path(str(prefix) + '.encoder.json')
    index_path = Path(str(prefix) + '.smlm')
    native_path = Path(str(prefix) + '.encoder.senc')
    encoder = SemanticEncoder.load(encoder_path)
    index = load_index(index_path, encoder=encoder)
    # Do not time loading or model training as if it were per-query inference.
    inputs = [(r['input_ids'], r['mask_index']) for r in validation]
    python_pred = [index.predict(ids, pos)['token'] for ids, pos in inputs]
    durations = []
    for _ in range(repeat):
        for ids, pos in inputs:
            start = perf_counter_ns()
            index.predict(ids, pos)
            durations.append(perf_counter_ns() - start)
    result = {
        'format': 'spark-modernbert-benchmark-v1', 'teacher_execution': 'offline traces',
        'source': 'synthetic/fixture' if allow_synthetic else 'ModernBERT-labeled traces (unverified provenance)',
        'trace_audit': audit,
        'baseline': majority_baseline(training, validation),
        'python_student': {**evaluate_predictions(validation, python_pred),
                           'latency': latency_summary(durations)},
        'disk_bytes': {'encoder_json': encoder_path.stat().st_size,
                       'encoder_native': native_path.stat().st_size,
                       'index': index_path.stat().st_size},
        'host': {'system': platform.system(), 'architecture': platform.machine(),
                 'python': platform.python_version()},
        'limitations': [
            'Teacher top-1 agreement is not ground-truth language understanding accuracy.',
            'Benchmarks measure masked-token prediction, not autoregressive decoding.',
            'Samples may share templates even when exact text does not overlap; use independent sources.',
            'The student is not a weight-preserving compilation of ModernBERT.',
        ],
    }
    if all('mask_hidden' in r for r in validation):
        result['student_signature_bit_agreement'] = sum(
            1 - ((encoder.encode(r['input_ids'], r['mask_index']) ^
                  _target_code(r['mask_hidden'], encoder.seed)).bit_count() / 256)
            for r in validation) / len(validation)
    result['mean_known_feature_fraction'] = sum(
        encoder.coverage(ids, pos) for ids, pos in inputs) / len(inputs)
    if build is not None:
        with NativeStudent(prefix, build) as native:
            native_preds = [native.predict(ids, pos) for ids, pos in inputs]
            samples = []
            for _ in range(repeat):
                for ids, pos in inputs:
                    start = perf_counter_ns()
                    native.predict(ids, pos)
                    samples.append(perf_counter_ns() - start)
            mismatched = sum(a != b for a, b in zip(python_pred, native_preds))
            result['native_student'] = {
                **evaluate_predictions(validation, native_preds),
                'latency': latency_summary(samples),
                'python_disagreement_count': mismatched,
                'python_agreement_fraction': 1 - mismatched / len(validation),
            }
            if require_native_parity and mismatched:
                raise RuntimeError(f'native and Python top-1 predictions differ on {mismatched} queries')
    else:
        result['native_student'] = None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['process_peak_rss_kib_linux'] = rss if sys.platform.startswith('linux') else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--build', type=Path, help='CMake build directory; omit for Python-only mode')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--allow-synthetic', action='store_true',
                        help='Fixture tests only: skip provenance check')
    parser.add_argument('--require-native-parity', action='store_true')
    parser.add_argument('--output', type=Path, help='write JSON report')
    args = parser.parse_args()
    if args.require_native_parity and args.build is None:
        parser.error('--require-native-parity requires --build')
    try:
        report = run_benchmark(args.traces, args.prefix, args.build, repeat=args.repeat,
                               allow_synthetic=args.allow_synthetic,
                               require_native_parity=args.require_native_parity)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

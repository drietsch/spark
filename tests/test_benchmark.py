"""M4 trace audits and Python/native benchmark (fixtures, no model download)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spark.benchmark import (audit_traces, evaluate_predictions, latency_summary,
                             majority_baseline, percentile_ms)
from spark.modernbert import compile_index
from spark.native_encoder import export_encoder
from spark.semantic import train_encoder, train_signatures
from tools.modernbert_teacher import split_id
from tools.modernbert_benchmark import run_benchmark


def sample(i, *, split='train', group=None):
    return {'input_ids': [101, 10 + i, 999, 20 + i, 102], 'mask_index': 2,
            'split': split, 'model': 'answerdotai/ModernBERT-base',
            'source_text': group if group is not None else f'Test sample {i} [MASK].',
            'topk': [{'token': 40 + i % 2, 'prob': 0.85},
                     {'token': 80 + i, 'prob': 0.05}],
            'mask_hidden': [1.0, 0.25, -0.1] if i % 2 == 0 else [-1.0, 0.2, 0.1]}


class AuditTests(unittest.TestCase):
    def test_duplicate_text_across_splits_rejected(self):
        a, b = sample(1), sample(2, split='validation')
        b['source_text'] = a['source_text']
        with self.assertRaisesRegex(ValueError, 'split leakage'):
            audit_traces([a, b])

    def test_duplicate_token_sequence_across_splits_rejected(self):
        a, b = sample(1), sample(2, split='validation')
        b['input_ids'] = list(a['input_ids'])
        with self.assertRaisesRegex(ValueError, 'split leakage'):
            audit_traces([a, b])

    def test_identical_prompts_rejected(self):
        a, b = sample(1), sample(1)
        with self.assertRaisesRegex(ValueError, 'duplicate masked context'):
            audit_traces([a, b, sample(2, split='validation')])

    def test_model_provenance_explicit(self):
        train, val = sample(1), sample(2, split='validation')
        train.pop('model')
        with self.assertRaisesRegex(ValueError, 'real ModernBERT'):
            audit_traces([train, val])
        self.assertEqual(audit_traces([train, val], allow_synthetic=True)['validation'], 1)

    def test_group_split_stability(self):
        original = 'Hello  [MASK] World.'
        a = split_id([1, 2], 1, 0.4, group_key=original)
        b = split_id([7, 8, 9], 2, 0.4, group_key='hello [MASK] world.')
        self.assertEqual(a, b)

    def test_latency_nearest_rank(self):
        self.assertEqual(percentile_ms([1_000_000, 2_000_000, 9_000_000], 95), 9.0)
        self.assertEqual(latency_summary([1_000_000, 2_000_000])['queries'], 2)
        self.assertIsNone(percentile_ms([], 50))

    def test_baseline_and_predictions(self):
        train = [sample(0), sample(2), sample(3)]
        val = [sample(5, split='validation'), sample(6, split='validation')]
        baseline = majority_baseline(train, val)
        self.assertEqual(baseline['label'], 40)
        self.assertEqual(baseline['top1_agreement_all'], 0.5)
        measured = evaluate_predictions(val, [None, 40])
        self.assertEqual(measured['coverage'], 0.5)
        self.assertEqual(measured['top1_agreement_all'], 0.5)


class EndToEndM4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('cmake') or not shutil.which('c++'):
            raise unittest.SkipTest('native C++/CMake unavailable')
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.build = cls.directory / 'build'
        subprocess.run(['cmake', '-S', str(ROOT), '-B', str(cls.build)], check=True, capture_output=True)
        subprocess.run(['cmake', '--build', str(cls.build), '-j2'], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_benchmark_native_reference_and_no_silent_real_claim(self):
        train = [sample(i) for i in range(8)]
        val = [sample(i, split='validation') for i in range(8, 11)]
        rows = train + val
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'model'
            traces = Path(tmp) / 'traces.jsonl'
            traces.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            enc = train_encoder(train)
            enc.save(Path(str(prefix) + '.encoder.json'))
            export_encoder(enc, Path(str(prefix) + '.encoder.senc'))
            compile_index(train, Path(str(prefix) + '.smlm'), encoder=enc,
                          signatures=train_signatures(train, enc))
            report = run_benchmark(traces, prefix, self.build, repeat=2)
            self.assertEqual(report['format'], 'spark-modernbert-benchmark-v1')
            self.assertEqual(report['trace_audit']['validation'], 3)
            self.assertEqual(report['python_student']['examples'], 3)
            self.assertEqual(report['native_student']['examples'], 3)
            self.assertEqual(report['native_student']['latency']['queries'], 6)
            self.assertIn('python_disagreement_count', report['native_student'])
            self.assertIn('student_signature_bit_agreement', report)
            self.assertIn('ModernBERT-labeled', report['source'])
            rows[0]['model'] = 'fake-teacher'
            traces.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'real ModernBERT'):
                run_benchmark(traces, prefix, None)


if __name__ == '__main__':
    unittest.main()

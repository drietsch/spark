"""Test offline-learning/runtime separation without neural dependencies."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.modernbert import compile_index, load_index
from spark.semantic import SemanticEncoder, _target_code, train_encoder, train_signatures

ROOT = Path(__file__).resolve().parents[1]


def trace(ids, pos, token, hidden, split='train'):
    return {'input_ids': ids, 'mask_index': pos,
            'topk': [{'token': token, 'prob': 0.95}],
            'mask_hidden': hidden, 'split': split}


class SemanticTest(unittest.TestCase):
    def setUp(self):
        self.traces = [
            trace([101, 11, 999, 21, 102], 2, 41, [1., 0., -.5, .2]),
            trace([101, 11, 999, 22, 102], 2, 41, [1., .1, -.6, .3]),
            trace([101, 12, 999, 31, 102], 2, 42, [-1., 0., .8, -.2]),
        ]

    def test_deterministic_training_and_teacher_signature(self):
        a = train_encoder(self.traces)
        b = train_encoder(self.traces)
        self.assertEqual(a.fingerprint64(), b.fingerprint64())
        self.assertEqual(a.encode(self.traces[0]['input_ids'], 2),
                         b.encode(self.traces[0]['input_ids'], 2))
        self.assertNotEqual(_target_code(self.traces[0]['mask_hidden']),
                            _target_code(self.traces[-1]['mask_hidden']))

    def test_binary_roundtrip_encoder_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc = train_encoder(self.traces)
            enc_path = Path(tmp) / 'model.encoder.json'
            path = Path(tmp) / 'model.smlm'
            enc.save(enc_path)
            same = SemanticEncoder.load(enc_path)
            compile_index(self.traces, path, encoder=enc, signatures=train_signatures(self.traces, enc))
            idx = load_index(path, encoder=same)
            r = idx.predict(self.traces[0]['input_ids'], 2, fallback=True)
            self.assertIsNotNone(r['token'])
            with self.assertRaisesRegex(ValueError, 'requires its exact encoder'):
                load_index(path)
            bad = SemanticEncoder(dict(same.weights), same.seed + 1)
            with self.assertRaisesRegex(ValueError, 'requires its exact encoder'):
                load_index(path, encoder=bad)

    def test_validation_never_trained(self):
        validation = trace([101, 777, 999, 888, 102], 2, 88, [0., 1., 0., .4], split='validation')
        with self.assertRaises(ValueError):
            train_encoder(self.traces + [validation])

    def test_teacher_targets_required_and_nonfinite_rejected(self):
        missing = dict(self.traces[0]); missing.pop('mask_hidden')
        with self.assertRaises(KeyError):
            train_encoder([missing])
        bad = dict(self.traces[0]); bad['mask_hidden'] = [float('nan')]
        with self.assertRaisesRegex(ValueError, 'invalid values'):
            train_encoder([bad])

    def test_cli_end_to_end_validation(self):
        # Fixed splits to avoid any dependence on random hash partitioning.
        val = trace([101, 11, 999, 23, 102], 2, 41, [1., 0., -.5, .2], 'validation')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'traces.jsonl'
            source.write_text(''.join(json.dumps(t)+'\n' for t in self.traces + [val]))
            prefix = Path(tmp) / 'model'
            cmd = [sys.executable, str(ROOT/'tools/modernbert_semantic.py')]
            train = subprocess.run(cmd + ['compile', str(source), str(prefix)],
                                   check=True, text=True, capture_output=True)
            self.assertEqual(json.loads(train.stdout)['prompts'], 3)
            self.assertTrue(Path(str(prefix) + '.encoder.senc').exists())
            evaluation = subprocess.run(cmd + ['evaluate', str(source), str(prefix),
                                               '--fallback-nearest'],
                                        check=True, text=True, capture_output=True)
            result = json.loads(evaluation.stdout)
            self.assertEqual(result['examples'], 1)
            self.assertIn('mean_teacher_signature_bit_agreement', result)
            self.assertEqual(result['coverage'], 1)


if __name__ == '__main__':
    unittest.main()

"""Native SPARKENC / MLM ABI and Python semantic encoder parity (no torch)."""
import ctypes
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spark.modernbert import compile_index, load_index
from spark.native_encoder import export_encoder
from spark.semantic import SemanticEncoder, train_encoder, train_signatures

ROOT = Path(__file__).resolve().parents[1]


def row(ids, mask, tok, hidden, split='train'):
    return {'input_ids': ids, 'mask_index': mask, 'split': split,
            'topk': [{'token': tok, 'prob': 0.9}], 'mask_hidden': hidden}


class NativeSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('cmake') or not shutil.which('c++'):
            raise unittest.SkipTest('C++17/CMake toolchain unavailable')
        cls.work = tempfile.TemporaryDirectory()
        cls.build = Path(cls.work.name) / 'build'
        subprocess.run(['cmake', '-S', str(ROOT), '-B', str(cls.build)], check=True, capture_output=True)
        subprocess.run(['cmake', '--build', str(cls.build), '-j2'], check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(cls.build / 'libspark_semantic_runtime.so'))
        cls.lib.spark_semantic_encoder_open_v1.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        cls.lib.spark_semantic_encoder_open_v1.restype = ctypes.c_int32
        cls.lib.spark_semantic_encoder_close_v1.argtypes = [ctypes.c_void_p]
        cls.lib.spark_semantic_encode_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32,
                                                      ctypes.c_int32, ctypes.c_uint64]
        cls.lib.spark_semantic_encode_v1.restype = ctypes.c_int32
        cls.lib.spark_semantic_predict_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                                                       ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                                       ctypes.c_uint64, ctypes.c_uint64]
        cls.lib.spark_semantic_predict_v1.restype = ctypes.c_int32
        cls.probe = ctypes.CDLL(str(cls.build / 'libspark_mlm_runtime.so'))
        cls.probe.spark_mlm_index_open_v1.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        cls.probe.spark_mlm_index_open_v1.restype = ctypes.c_int32
        cls.probe.spark_mlm_index_close_v1.argtypes = [ctypes.c_void_p]

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def setUp(self):
        self.rows = [row([101, 8, 999, 9, 102], 2, 40, [1., 0., -1.]),
                     row([101, 8, 999, 10, 102], 2, 40, [1., 0.2, -0.8]),
                     row([101, 14, 999, 20, 102], 2, 41, [-1., 0., 1.])]
        self.enc = train_encoder(self.rows)
        self.path = Path(self.work.name) / f'{self.id().split(".")[-1]}.senc'
        self.index_path = self.path.with_suffix('.smlm')
        export_encoder(self.enc, self.path)
        compile_index(self.rows, self.index_path, signatures=train_signatures(self.rows, self.enc), encoder=self.enc)
        self.handle = ctypes.c_void_p()
        self.assertEqual(self.lib.spark_semantic_encoder_open_v1(str(self.path).encode(), ctypes.byref(self.handle)), 0)
        self.addCleanup(self.lib.spark_semantic_encoder_close_v1, self.handle)

    def encode_native(self, ids, pos, handle=None):
        arr = (ctypes.c_uint32 * len(ids))(*ids)
        sig = (ctypes.c_uint8 * 32)()
        status = self.lib.spark_semantic_encode_v1((handle or self.handle).value,
                         ctypes.addressof(arr), len(ids), pos, ctypes.addressof(sig))
        return status, int.from_bytes(bytes(sig), 'little')

    def test_256_bit_exact_parity_many_contexts(self):
        rng = random.Random(2026)
        pool = [8, 9, 10, 14, 20, 101, 102, 999, 0, 2**32 - 1, 777]
        for _ in range(110):
            ids = [rng.choice(pool) for _ in range(rng.randrange(1, 160))]
            pos = rng.randrange(len(ids))
            status, actual = self.encode_native(ids, pos)
            self.assertEqual(status, 0)
            self.assertEqual(actual, self.enc.encode(ids, pos))

    def test_invalid_position(self):
        self.assertLess(self.encode_native([1, 2, 3], 3)[0], 0)
        self.assertLess(self.encode_native([1, 2, 3], -1)[0], 0)

    def test_corruption_and_truncation_rejected(self):
        original = self.path.read_bytes()
        for bad in (original[:-1], original[:60], original[:100] + bytes([original[100] ^ 4]) + original[101:]):
            badpath = Path(self.work.name) / 'corrupt.senc'
            badpath.write_bytes(bad)
            ptr = ctypes.c_void_p()
            self.assertLess(self.lib.spark_semantic_encoder_open_v1(str(badpath).encode(), ctypes.byref(ptr)), 0)
            self.assertIsNone(ptr.value)

    def test_composed_native_prediction(self):
        idx = ctypes.c_void_p()
        self.assertEqual(self.probe.spark_mlm_index_open_v1(str(self.index_path).encode(), ctypes.byref(idx)), 0)
        self.addCleanup(self.probe.spark_mlm_index_close_v1, idx)
        model = load_index(self.index_path, encoder=self.enc)
        for t in self.rows:
            tokens = (ctypes.c_uint32 * len(t['input_ids']))(*t['input_ids'])
            found = (ctypes.c_int32 * 64)()
            scores = (ctypes.c_int32 * 64)()
            count = self.lib.spark_semantic_predict_v1(
                self.handle.value, idx.value, ctypes.addressof(tokens), len(t['input_ids']),
                t['mask_index'], 64, ctypes.addressof(found), ctypes.addressof(scores))
            self.assertGreater(count, 0)
            self.assertEqual(found[0], model.predict(t['input_ids'], t['mask_index'])['token'])
        # index compiled with a different *learned* encoder must fail before probing
        other = SemanticEncoder(dict(self.enc.weights), seed=self.enc.seed + 1)
        badindex = Path(self.work.name) / 'mismatched.smlm'
        compile_index(self.rows, badindex, signatures=train_signatures(self.rows, other), encoder=other)
        idx2 = ctypes.c_void_p()
        self.assertEqual(self.probe.spark_mlm_index_open_v1(str(badindex).encode(), ctypes.byref(idx2)), 0)
        self.addCleanup(self.probe.spark_mlm_index_close_v1, idx2)
        ids = (ctypes.c_uint32 * 3)(1, 999, 2)
        found = (ctypes.c_int32 * 64)()
        scores = (ctypes.c_int32 * 64)()
        self.assertEqual(self.lib.spark_semantic_predict_v1(
            self.handle.value, idx2.value, ctypes.addressof(ids), 3, 1, 64,
            ctypes.addressof(found), ctypes.addressof(scores)), -2)

    def test_demo_and_python_cli(self):
        cmd = [str(self.build / 'spark_semantic_demo'), str(self.path), str(self.index_path),
               '2', '101', '8', '999', '9', '102']
        demo = subprocess.run(cmd, text=True, capture_output=True, check=True)
        self.assertIn('token=40', demo.stdout)


if __name__ == '__main__':
    unittest.main()

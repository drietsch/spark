"""ABI v1 native probe parity against the SPARK Python reference.

The MLIR C++ dialect is tested separately when an MLIR SDK is available.
These tests ensure its C ABI call target exists and is usable.
"""
import ctypes
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from spark.modernbert import compile_index, load_index, signature
from spark.semantic import SemanticEncoder

ROOT = Path(__file__).resolve().parents[1]


def trace(ids, pos, tokens):
    return {'input_ids': ids, 'mask_index': pos,
            'topk': [{'token': token, 'prob': p} for token, p in tokens]}


class NativeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++')
        if compiler is None:
            raise unittest.SkipTest('C++17 compiler missing')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.libpath = Path(cls.tmp.name) / 'libsparkmlm.so'
        subprocess.run([compiler, '-std=c++17', '-O2', '-fPIC', '-shared',
                        '-I', str(ROOT / 'include'),
                        str(ROOT / 'runtime/spark_mlm_runtime.cpp'),
                        '-o', str(cls.libpath)], check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(cls.libpath))
        cls.lib.spark_mlm_index_open_v1.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        cls.lib.spark_mlm_index_open_v1.restype = ctypes.c_int32
        cls.lib.spark_mlm_index_close_v1.argtypes = [ctypes.c_void_p]
        cls.lib.spark_mlm_probe_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32,
                                              ctypes.c_uint64, ctypes.c_uint64]
        cls.lib.spark_mlm_probe_v1.restype = ctypes.c_int32

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def probe(self, path, sig, limit=64):
        handle = ctypes.c_void_p()
        status = self.lib.spark_mlm_index_open_v1(str(path).encode(), ctypes.byref(handle))
        self.assertEqual(status, 0)
        self.addCleanup(self.lib.spark_mlm_index_close_v1, handle)
        input_bytes = (ctypes.c_uint8 * 32).from_buffer_copy(sig.to_bytes(32, 'little'))
        tokens = (ctypes.c_int32 * 64)()
        scores = (ctypes.c_int32 * 64)()
        count = self.lib.spark_mlm_probe_v1(handle.value, ctypes.addressof(input_bytes), limit,
                                             ctypes.addressof(tokens), ctypes.addressof(scores))
        return count, list(tokens), list(scores)

    def test_baseline_match(self):
        data = [trace([10, 20, 30], 1, [(44, .7), (45, .2)]),
                trace([11, 21, 31], 1, [(9, .8), (8, .1)]),
                trace([99, 20, 88], 1, [(44, .9)])]
        path = Path(self.tmp.name) / 'simple.smlm'
        compile_index(data, path)
        model = load_index(path)
        for row in data:
            with self.subTest(row=row):
                sig = signature(row['input_ids'], row['mask_index'])
                count, tokens, scores = self.probe(path, sig)
                self.assertGreater(count, 0)
                self.assertEqual(tokens[0], model.predict(row['input_ids'], row['mask_index'])['token'])
                self.assertTrue(all(s >= 0 for s in scores[:count]))
                self.assertTrue(all(t == -1 for t in tokens[count:]))

    def test_semantic_bandwidth_and_fingerprint_index(self):
        # The saved index binds its encoder fingerprint, but the probe accepts a
        # precomputed 256-bit code. Its caller must validate the encoder itself.
        enc = SemanticEncoder({'L1:10': [1] * 256})
        data = [trace([10, 20, 30], 1, [(91, 0.9)])]
        path = Path(self.tmp.name) / 'semantic.smlm'
        from spark.modernbert import compile_index as compile_sem
        compile_sem(data, path, signatures=[(1 << 256) - 1], encoder=enc)
        count, tokens, _ = self.probe(path, (1 << 256) - 1)
        self.assertEqual((count, tokens[0]), (1, 91))

    def test_invalid_and_corrupted_index(self):
        path = Path(self.tmp.name) / 'bad.smlm'
        path.write_bytes(b'not-an-index')
        handle = ctypes.c_void_p()
        self.assertLess(self.lib.spark_mlm_index_open_v1(str(path).encode(), ctypes.byref(handle)), 0)
        self.assertIsNone(handle.value)
        self.assertLess(self.lib.spark_mlm_probe_v1(0, 0, 64, 0, 0), 0)

    def test_limit_and_empty_result(self):
        data = [trace([1, 2, 3], 1, [(7, .8), (9, .1), (8, .05)])]
        path = Path(self.tmp.name) / 'small.smlm'
        compile_index(data, path)
        n, tokens, scores = self.probe(path, signature([1, 2, 3], 1), 2)
        self.assertEqual((n, tokens[:2]), (2, [7, 9]))
        self.assertLess(self.probe(path, signature([1, 2, 3], 1), 0)[0], 0)
        self.assertGreater(scores[0], scores[1])


if __name__ == '__main__':
    unittest.main()

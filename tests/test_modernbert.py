"""Offline tests: no network, torch, or transformers required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.modernbert import compile_index, evaluate, load_index, read_traces, signature
from tools.modernbert_teacher import split_id


def trace(ids, pos, targets, split="train"):
    return {"input_ids": ids, "mask_index": pos, "topk": [
        {"token": t, "prob": p} for t, p in targets], "split": split}


class ModernBertTrial(unittest.TestCase):
    def test_signature_is_deterministic_and_context_sensitive(self):
        a = signature([101, 17, 999, 22, 102], 2)
        self.assertEqual(a, signature([101, 17, 999, 22, 102], 2))
        self.assertNotEqual(a, signature([101, 19, 999, 22, 102], 2))

    def test_binary_roundtrip_and_top1(self):
        items = [trace([101, 17, 999, 22, 102], 2, [(42, .9), (43, .1)])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.smlm"
            meta = compile_index(items, path)
            self.assertEqual(meta["prompts"], 1)
            idx = load_index(path)
            result = idx.predict(items[0]["input_ids"], 2)
            self.assertEqual(result["token"], 42)
            self.assertFalse(result["fallback"])
            self.assertEqual(evaluate(idx, items)["top1_agreement_all"], 1)
            with path.open("ab") as f:
                f.write(b"x")
            with self.assertRaises(ValueError):
                load_index(path)

    def test_holdout_split_stable(self):
        self.assertEqual(split_id([10, 20, 30], 1, .3), split_id([10, 20, 30], 1, .3))

    def test_duplicate_rejected(self):
        item = trace([1, 2, 3], 1, [(44, .9)])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "duplicate prompt"):
                compile_index([item, item], Path(tmp) / "index.smlm")

    def test_unmatched_is_reported_not_hallucinated(self):
        item = trace([1, 2, 3], 1, [(44, .9)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.smlm"
            compile_index([item], path)
            idx = load_index(path)
            unseen = trace([777, 888, 999, 111], 1, [(55, .8)], "validation")
            stats = evaluate(idx, [unseen], fallback=True)
            self.assertEqual(stats["examples"], 1)
            self.assertEqual(stats["coverage"], 1)
            self.assertEqual(stats["fallback_queries"], 1)


if __name__ == "__main__":
    unittest.main()

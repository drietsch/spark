"""Offline checks for WikiText-2 split-preserving data generation."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("large_corpus", ROOT / "tools/prepare_large_modernbert.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class LargeCorpusTests(unittest.TestCase):
    def test_deterministic_single_mask(self):
        line = "The academic library stores information and research materials for students at the university."
        a = module.pick_mask(line, "train:0:0", 42)
        self.assertEqual(a, module.pick_mask(line, "train:0:0", 42))
        self.assertEqual(a.count("[MASK]"), 1)

    def test_train_validation_remain_separate(self):
        train = [f"An interesting university researcher investigated scientific evidence concerning remarkable chemistry materials number {i} today." for i in range(18)]
        val = [f"The talented composer created original orchestral music during a successful performance with many musicians number {i}." for i in range(12)]
        rows, manifest = module.prepare_splits({"train":train,"validation":val}, 10, 6)
        self.assertEqual(manifest["examples"], {"train":10,"validation":6})
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(r["split"]=="validation" for r in rows),6)
        self.assertTrue(all(r["group_id"].startswith("wikitext2-"+r["split"]) for r in rows))
        self.assertEqual(len(set(r["text"] for r in rows)),len(rows))

    def test_fails_if_corpus_too_small(self):
        with self.assertRaises(RuntimeError):
            module.prepare_splits({"train":["Too short"],"validation":["Too short"]},1,1)

    def test_no_existing_mask(self):
        self.assertIsNone(module.pick_mask("This paragraph has [MASK] already but plenty of otherwise interesting words for testing.","key",42))

if __name__=="__main__":
    unittest.main()

"""Regression tests for previously malformed ModernBERT teacher traces."""
import unittest
from tools.modernbert_teacher import mask_preserving_window, prepare_masked_inputs


class DummyTokenizer:
    cls_token_id = 50281
    sep_token_id = 50282
    mask_token_id = 50284

    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": self.tokens}


class TeacherBoundariesTests(unittest.TestCase):
    def test_cls_sep_are_inserted(self):
        ids, pos, cropped = prepare_masked_inputs(DummyTokenizer([12, 50284, 13]), "sample", 128)
        self.assertEqual(ids, [50281, 12, 50284, 13, 50282])
        self.assertEqual(pos, 2)
        self.assertFalse(cropped)

    def test_long_mask_near_end(self):
        original = list(range(300))
        original[290] = 50284
        ids, pos, cropped = prepare_masked_inputs(DummyTokenizer(original), "long", 128)
        self.assertTrue(cropped)
        self.assertEqual((len(ids), ids[0], ids[-1], ids[pos]), (128, 50281, 50282, 50284))

    def test_long_mask_near_start(self):
        original = list(range(300))
        original[1] = 50284
        ids, pos, cropped = prepare_masked_inputs(DummyTokenizer(original), "long", 128)
        self.assertTrue(cropped)
        self.assertEqual(pos, 2)

    def test_reject_duplicate_mask(self):
        with self.assertRaises(ValueError):
            prepare_masked_inputs(DummyTokenizer([50284, 55, 50284]), "bad", 128)

    def test_reject_missing_mask(self):
        with self.assertRaises(ValueError):
            prepare_masked_inputs(DummyTokenizer([51, 52, 53]), "bad", 128)

    def test_reject_boundary_in_payload(self):
        with self.assertRaises(ValueError):
            prepare_masked_inputs(DummyTokenizer([50281, 50284, 56]), "bad", 128)


if __name__ == "__main__":
    unittest.main()

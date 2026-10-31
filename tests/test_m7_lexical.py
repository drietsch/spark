"""Unit tests for matrix-free multi-stage lexical and supervised token scoring."""
import unittest
from spark.m7_context import LexicalBackoff, lexical_features, fused_predictions


class LexicalTests(unittest.TestCase):
    def test_bidirectional_specificity(self):
        feats = lexical_features([9,10,11,12,13,14,15],3)
        self.assertIn(("LR",11,13),feats)
        self.assertIn(("L2",10,11),feats)
        self.assertIn(("R2",13,14),feats)
        self.assertGreater(feats[("LR",11,13)],feats[("L1",11)])

    def test_training_and_generalization(self):
        model=LexicalBackoff()
        model.add([1,10,0,20,2],2,31)
        model.add([1,10,0,20,3],2,31)
        model.add([1,99,0,88,3],2,42)
        self.assertEqual(model.predict([1,10,0,20,4],2),31)
        self.assertEqual(model.seen,3)
        self.assertGreater(model.stats()["feature_types"],0)

    def test_simple_fusion(self):
        a={10:3.0,20:1.0}
        b={20:5.0,10:1.0}
        self.assertEqual(fused_predictions(a,b,lexical_weight=0.9),10)
        self.assertEqual(fused_predictions(a,b,lexical_weight=0.1),20)
        self.assertEqual(fused_predictions({}, {}, majority=99),99)

    def test_invalid_mask_and_weight(self):
        with self.assertRaises(ValueError):
            lexical_features([1,2,3],3)
        with self.assertRaises(ValueError):
            fused_predictions({1:1},{2:2},lexical_weight=1.5)

    def test_empty_memory(self):
        model=LexicalBackoff()
        self.assertIsNone(model.predict([1,2,3],1))


if __name__ == "__main__":
    unittest.main()

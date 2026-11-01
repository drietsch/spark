"""M18: independent tests of cue-free contextual policy discovery and native ABI."""
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.context_memory import (ContextRouter,Expression,Observation,
                                  adapt_prequential,episodes,fit,metrics)


class ContextMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=episodes(44,100,keys=4,distractors=(4,12))
        cls.fitted,_=fit(cls.training,keys=8)

    def test_no_oracle_relevance_feature(self):
        event=Observation(2,0.6,-0.8,0.7)
        self.assertEqual(set(vars(event)),{'key','value','a','b','query','target'})
        self.assertEqual(set(vars(self.fitted.expression)),{'feature','comparison','threshold'})
        self.assertEqual(self.fitted.expression,Expression('product','gt',0.0))

    def test_delayed_read_supervision_and_ood_keys(self):
        unseen=episodes(45,12,keys=8,distractors=(1024,1400),scale=2.2)
        report=metrics(self.fitted,unseen)
        self.assertEqual(report['reads'],36)
        self.assertEqual(report['correct'],36)
        self.assertEqual(report['query_mse'],0.0)

    def test_nonlinear_feature_needed(self):
        linear,_=fit(self.training,keys=8,linear_only=True)
        unseen=episodes(48,12,keys=8,distractors=(80,150),scale=2.2)
        self.assertLess(metrics(linear,unseen)['correct'],metrics(self.fitted,unseen)['correct'])

    def test_prequential_regime_changes_not_mode_labels(self):
        blocks=[episodes(71,24,keys=8,mode='same_sign'),
                episodes(72,72,keys=8,mode='opposite_sign'),
                episodes(73,72,keys=8,mode='same_sign')]
        result=adapt_prequential(self.fitted,blocks,window=24,refresh_every=6)
        self.assertGreater(result[1]['first_12_mse'],0.1)
        self.assertGreater(result[2]['first_12_mse'],0.1)
        self.assertGreater(result[1]['last_24_accuracy'],.95)
        self.assertGreater(result[2]['last_24_accuracy'],.95)
        self.assertEqual(result[1]['final_expression']['comparison'],'lt')
        self.assertEqual(result[2]['final_expression']['comparison'],'gt')

    def test_incorrect_or_malicious_program_rejected(self):
        with self.assertRaises(ValueError):ContextRouter(Expression('__import__','gt',0))
        with self.assertRaises(ValueError):ContextRouter(Expression('product','exec',0))
        with self.assertRaises(ValueError):ContextRouter(Expression('product','gt',math.inf))
        with self.assertRaises(ValueError):ContextRouter(Expression('product','gt',0),keys=0)
        with self.assertRaises(ValueError):self.fitted.replay([Observation(-1,0,1,1)])
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'program.json';self.fitted.save(p)
            self.assertEqual(ContextRouter.load(p).expression,self.fitted.expression)
            raw=json.loads(p.read_text())
            raw['expression']['feature']='eval'
            p.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):ContextRouter.load(p)

    def test_compiled_expression_is_from_discovery(self):
        source=self.fitted.cpp_source()
        self.assertIn('(a*b) > 0',source)
        self.assertNotIn('signal',source)
        self.assertNotIn('relevance',source.lower().replace('no relevance indicator',''))
        self.assertNotIn('std::vector',source)

    @unittest.skipUnless(shutil.which('g++') or shutil.which('clang++') or shutil.which('c++'), 'C++17 compiler needed')
    def test_native_python_parity(self):
        sys.path.insert(0,str(ROOT/'tools'))
        from context_experiment import native_verify
        unseen=episodes(91,8,keys=8,distractors=(256,512),scale=2.0)
        with tempfile.TemporaryDirectory() as t:
            r=native_verify(self.fitted,unseen,Path(t)/'model.so')
        self.assertEqual(r['python_native_mismatches'],0)
        self.assertEqual(r['correct'],r['reads'])

    def test_read_target_not_used_as_feature(self):
        a=Observation(0,1.3,0.5,0.4)
        b=Observation(0,0,0,0,True,12345.)
        err,_,reads,preds=self.fitted.replay([a,b],return_reads=True)
        self.assertEqual(preds,[1.3])
        self.assertEqual(reads,1)
        self.assertAlmostEqual(err,(1.3-12345)**2)


if __name__=='__main__':unittest.main()

"""M19: no relevance labels, learned composition, untrusted serialization, native parity."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from spark.composed_memory import ComposedRouter,Program,Expression,generate,search

class ComposedMemoryTests(unittest.TestCase):
    def test_program_validation(self):
        good=Program('and',Expression('product','gt',0.),Expression('sum','gt',.25))
        good.validate()
        with self.assertRaises(ValueError):Program('exec',good.left,good.right).validate()
        with self.assertRaises(ValueError):Program('and',good.left,good.left).validate()
        with self.assertRaises(ValueError):Program('and',good.left,Expression('a','gt',float('nan'))).validate()
        with self.assertRaises(ValueError):Program('atom',good.left,good.right).validate()

    def test_serialization_rejects_arbitrary_code(self):
        p=Program('and',Expression('product','gt',0.),Expression('sum','gt',.25))
        model=ComposedRouter(p,keys=8)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.json';model.save(path)
            self.assertEqual(ComposedRouter.load(path).program,p)
            obj=json.loads(path.read_text());obj['program']['op']='python'
            path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):ComposedRouter.load(path)
            obj['program']['op']='and';obj['program']['left']['threshold']=0.13
            path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):ComposedRouter.load(path)

    def test_discovery_with_delayed_read_targets(self):
        train=generate(1919,80,keys=4)
        model,_=search(train,keys=8)
        simple,_=search(train,keys=8,composition=False)
        new=generate(727,6,keys=8,distractors=(1024,2048),scale=2.2)
        self.assertEqual(model.program.op,'and')
        self.assertEqual(model.evaluate(new)['accuracy'],1.)
        self.assertLess(simple.evaluate(new)['accuracy'],1.)

    def test_no_query_targets_leak_into_event_arguments(self):
        rows=generate(30,8,keys=4)
        for seq in rows:
            for e in seq:
                if not e.query:self.assertEqual(e.target,0.)

    @unittest.skipUnless(shutil.which('c++') or shutil.which('g++') or shutil.which('clang++'),'requires compiler')
    def test_compiled_parity(self):
        import sys
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'tools'))
        from composition_experiment import native_parity
        model=ComposedRouter(Program('and',Expression('product','gt',0.),Expression('sum','gt',.25)),keys=8)
        with tempfile.TemporaryDirectory() as d:
            result=native_parity(model,generate(313,3,keys=8,distractors=(250,500)),Path(d)/'program.so')
        self.assertEqual(result['python_native_mismatches'],0)
        self.assertEqual(result['reads'],9)
        self.assertEqual(result['correct'],9)

if __name__=='__main__':unittest.main()

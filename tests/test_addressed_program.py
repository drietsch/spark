"""M20 regression: no oracle IDs, delayed supervision, 3-step program, C++ parity."""
from __future__ import annotations
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from spark.addressed_program import (Address,AddressedProgram,Observation,atom,
    combine,generate,learn_address,discover_policy,validate_node,replay)

class AddressedProgramTests(unittest.TestCase):
    def test_training_events_have_no_hidden_key_or_write_labels(self):
        streams=generate(30,8,keys=4)
        for seq in streams:
            for e in seq:
                self.assertEqual(set(e.__dict__),{'u','v','value','a','b','query','target'})
                if not e.query:self.assertEqual(e.target,0.)
                else:self.assertEqual(e.value,0.)

    def test_address_learned_from_delayed_read_targets(self):
        calibration=generate(15,30,keys=4,calibration=True)
        found,report=learn_address(calibration,8)
        self.assertEqual((found.cu,found.cv),(1,-1))
        self.assertEqual(report['train_reads'],90)
        self.assertEqual(report['train_correct'],90)
        for k in range(8):
            e=Observation(k+2.1,2.1,0.,0.,0.)
            self.assertEqual(found.get(e,8),k)
        self.assertEqual(Address(1,0).get(Observation(4.1,2.1,0,0,0),8),4)
        self.assertNotEqual(found.get(Observation(4.1,2.1,0,0,0),8),4)

    def test_depth_three_recovery_and_shallow_negative_control(self):
        address=Address(1,-1)
        train=generate(2020,55,keys=4,distractors=(5,18),queries=4,scale=1.3)
        deep,info=discover_policy(train,address,8,depth=3,beam=90)
        shallow,_=discover_policy(train,address,8,depth=2)
        holdout=generate(3030,10,keys=8,distractors=(120,240),scale=2.5)
        self.assertEqual(info['train_mse'],0.)
        self.assertGreaterEqual(info['complexity'],7)
        self.assertEqual(deep.evaluate(holdout)['correct'],30)
        self.assertLess(shallow.evaluate(holdout)['correct'],30)

    def test_invalid_structure_and_untrusted_json_rejected(self):
        node=combine('and',combine('and',atom(16),atom(26)),atom(23))
        validate_node(node)
        model=AddressedProgram(Address(1,-1),node,8)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'p.json';model.save(p)
            self.assertEqual(AddressedProgram.load(p),model)
            doc=json.loads(p.read_text());doc['policy']['op']='eval'
            p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):AddressedProgram.load(p)
            doc=json.loads(p.read_text());doc['policy']=node;doc['address']['cu']=999
            p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):AddressedProgram.load(p)
            doc['address']={'cu':1,'cv':-1};doc['policy']['left']['left']['index']=400
            p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):AddressedProgram.load(p)
        with self.assertRaises(ValueError):
            replay([[Observation(float('nan'),0,0,0,0)] for _ in range(8)],Address(1,-1),node,8)

    @unittest.skipUnless(shutil.which('g++') or shutil.which('c++') or shutil.which('clang++'), 'C++17 compiler unavailable')
    def test_compiled_address_and_nested_program_parity(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from addressed_experiment import native_check
        node=combine('and',combine('and',atom(16),atom(26)),atom(23))
        model=AddressedProgram(Address(1,-1),node,8)
        with tempfile.TemporaryDirectory() as temp:
            result=native_check(model,generate(45,4,keys=8,distractors=(128,256),scale=2.5),Path(temp)/'m20.so')
        self.assertEqual(result['python_native_disagreements'],0)
        self.assertEqual((result['correct'],result['reads']),(12,12))
        self.assertGreater(result['operations'],500)

if __name__=='__main__':unittest.main()

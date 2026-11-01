"""M21: delayed-only labels, joint address/program learning, native live parity."""
from __future__ import annotations
import copy
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.joint_plastic_memory import JointMemory,generate,joint_search,replay_joint
from spark.plastic_function import PlasticFunction
from spark.addressed_program import Address,AddressedProgram,atom,combine,Observation


class JointPlasticTests(unittest.TestCase):
    def sample_model(self):
        # This policy is a fixed TEST FIXTURE, not passed to training.
        tree=combine('and',combine('and',atom(16),atom(23)),atom(2))
        return JointMemory(Address(1,-1),tree,8,
                           PlasticFunction.initialize(capacity=6,active=4,seed=29))

    def test_no_oracle_address_or_write_labels_in_observations(self):
        seq=generate(104,5,keys=4,distractors=(10,20))
        self.assertEqual(len(seq),5)
        for episode in seq:
            for e in episode:
                self.assertEqual(set(e.__dict__),{'u','v','value','a','b','query','target'})
                if not e.query:self.assertEqual(e.target,0.)
                else:self.assertEqual(e.value,0.)

    def test_joint_search_without_calibration(self):
        rows=generate(2021,58,keys=4,distractors=(6,16),queries=4,scale=1.3)
        model,diagnostics=joint_search(rows,capacity=8,coarse_count=24,beam=72)
        self.assertEqual(diagnostics['calibration_episodes'],0)
        self.assertEqual(diagnostics['coarse_candidates'],8)
        self.assertEqual(model.address,Address(1,-1))
        self.assertGreater(diagnostics['full_programs_searched'],1000)
        test=generate(404,7,keys=8,distractors=(128,256),queries=3,scale=2.0)
        result=replay_joint(test,model.address,model.policy,8)
        self.assertLess(result['mse'],0.1,result)

    def test_online_prediction_is_scored_before_feedback(self):
        model=self.sample_model()
        rows=generate(99,4,keys=4,queries=3)
        before=copy.deepcopy(model)
        a=before.evaluate(rows,learn=False)
        b=model.evaluate(rows,learn=True,lr=.06)
        self.assertEqual(a['predictions'][0],b['predictions'][0])
        self.assertGreater(model.steps,0)
        self.assertNotEqual(model.function.flatten(),before.function.flatten())
        # A query label is *not* read when prediction-only mode is used.
        e=next(e for e in rows[0] if e.query)
        changed=[[Observation(ev.u,ev.v,ev.value,ev.a,ev.b,ev.query,
                             999. if ev is e else ev.target) for ev in rows[0]]]
        # Check early predictions independent of the target even if it is changed.
        self.assertAlmostEqual(before.process(rows[0])[0][0],before.process(changed[0])[0][0])

    def test_serialization_rejects_invalid_program_and_coefficients(self):
        model=self.sample_model()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'m.json';model.save(path)
            loaded=JointMemory.load(path)
            self.assertEqual(model.address,loaded.address)
            self.assertEqual(model.function.flatten(),loaded.function.flatten())
            bad=json.loads(path.read_text())
            bad['policy']['op']='python_exec'
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):JointMemory.load(path)
            bad=json.loads(model_save(model,path))
            bad['units'][0][0]=float('nan')
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):JointMemory.load(path)

    def test_rejects_write_label_leakage_and_nan_inputs(self):
        model=self.sample_model()
        with self.assertRaises(ValueError):
            model.evaluate([[Observation(1,1,0.3,0,0,False,123.)] for _ in range(4)])
        with self.assertRaises(ValueError):
            model.process([Observation(float('nan'),1,1,0,0)])

    @unittest.skipUnless(shutil.which('c++') or shutil.which('g++') or shutil.which('clang++'),
                         'requires C++17 compiler')
    def test_native_prediction_and_online_parameter_update_parity(self):
        sys.path.insert(0,str(ROOT/'tools'))
        from joint_plastic_experiment import native_evaluate
        model=self.sample_model()
        streams=generate(131,7,keys=8,distractors=(32,48),queries=3)
        with tempfile.TemporaryDirectory() as tmp:
            no=native_evaluate(model,streams,Path(tmp)/'m21.so',learn=False)
            yes=native_evaluate(model,streams,Path(tmp)/'m21.so',learn=True,lr=.04)
        self.assertEqual(no['reads'],21)
        self.assertLess(no['max_python_native_prediction_delta'],1e-12)
        self.assertLess(yes['max_python_native_prediction_delta'],1e-12)
        self.assertLess(yes['max_python_native_updated_parameter_delta'],1e-12)


def model_save(model,path):
    model.save(path)
    return path.read_text()

if __name__=='__main__':unittest.main()

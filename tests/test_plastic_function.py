"""Dynamic neural mathematical function: live training, resizing and parity."""
import math
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from spark.plastic_function import PlasticFunction,PlasticProgram,cpp_source,default_program,validate_program


class PlasticTests(unittest.TestCase):
    def test_live_parameters_change_output(self):
        m=PlasticProgram(PlasticFunction.initialize(capacity=8,active=4,seed=7))
        s=(.4,-.2,.1,.3)
        p0=m.step(s,.3,-.4)
        for _ in range(120):
            target=(.3,s[0],.6*s[2]+.8,.5*s[3]+.2*s[2]+.8)
            m.observe(s,.3,-.4,target,lr=.08)
        p1=m.step(s,.3,-.4)
        self.assertGreater(abs(p1[2]-p0[2]),.1)
        self.assertLess(abs(p1[2]-target[2]),.02)
        m.function.set_active(8)
        self.assertNotAlmostEqual(m.step(s,.3,-.4)[2],p1[2],places=7)

    def test_serialization_restores_mutable_state(self):
        m=PlasticProgram(PlasticFunction.initialize(seed=3))
        s=(.2,.1,.4,.6)
        m.observe(s,.2,-.5,(.2,.2,.4,.3))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.json';m.save(p)
            loaded=PlasticProgram.load(p)
            self.assertEqual(m.step(s,.3,.5),loaded.step(s,.3,.5))
            self.assertEqual(loaded.steps,1)

    def test_rejects_malformed_program_and_nonfinite_updates(self):
        p=default_program()
        p['updates'][0]={'op':'system','command':'not permitted'}
        with self.assertRaises(ValueError):validate_program(p)
        m=PlasticProgram(PlasticFunction.initialize())
        with self.assertRaises(ValueError):m.observe((1,2,3,4),.1,.2,(1,2,3,float('nan')))
        with self.assertRaises(ValueError):m.function.set_active(999)
        with self.assertRaises(ValueError):cpp_source(default_program(),capacity=1000)

    @unittest.skipUnless(shutil.which('c++') or shutil.which('g++') or shutil.which('clang++'),'C++ compiler required')
    def test_native_mutability_and_online_parity(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from plastic_experiment import Native, truth, sample
        m=PlasticProgram(PlasticFunction.initialize(capacity=10,active=6,seed=12))
        with tempfile.TemporaryDirectory() as d:
            native=Native(m,Path(d)/'live.so')
            probes=list(sample(81,20))
            for sign in (1,-1):
                for s,x,c in sample(sign+95,60):
                    y=truth(s,x,c,sign)
                    m.observe(s,x,c,y,lr=.08)
                    native.observe(s,x,c,y,.08)
            for s,x,c in probes:
                self.assertLess(max(abs(a-b) for a,b in zip(m.step(s,x,c),native.step(s,x,c))),1e-11)
            self.assertLess(max(abs(a-b) for a,b in zip(m.function.flatten(),native.read_params())),1e-11)
            m.function.set_active(10);native.set_active(10)
            for s,x,c in probes:
                self.assertLess(max(abs(a-b) for a,b in zip(m.step(s,x,c),native.step(s,x,c))),1e-11)
            native.close()


if __name__=='__main__':unittest.main()

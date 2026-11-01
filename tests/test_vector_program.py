"""M10 multi-register symbolic discovery, safe load and compiled native parity."""
import ctypes
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from spark.vector_program import basis,cpp_source,load,save,synthesize,validate


def coupled(state,x,reset=False):
    a,b,c,d=state
    if reset and x==-2:return (0.,0.,0.,0.)
    q=(a-b)*x
    return (x,a,c+q,d+c+q)


def observations(reset):
    rng=random.Random(332)
    return [(s,x,coupled(s,x,reset)) for s,x in
            ((tuple(rng.uniform(-1,1) for _ in range(4)),
              float(rng.choice([-2,-1,0,1,2,3] if reset else [-1,0,1,2])))
             for _ in range(600))]


class VectorProgramTests(unittest.TestCase):
    def test_rejects_malicious_and_mutated_ast(self):
        p={'format':'spark-vector-symbolic-v1','registers':1,'degree':1,
           'powers':[list(x) for x in basis(1,1)],'guard':None,'then':[[]],'else':[[]]}
        validate(p)
        for key,value in [('format','eval-me'),('guard',float('nan')),
                          ('registers',999),('else',[[[[1,0],1,0]]])]:
            copy=__import__('copy').deepcopy(p);copy[key]=value
            with self.assertRaises(ValueError):validate(copy)

    def test_discovers_coupled_recurrence(self):
        model=synthesize(observations(False))
        self.assertIsNone(model.payload['guard'])
        for s in [(0.,1.,2.,3.),(-2.,3.,1.,0.),(2.,-1.,5.,-5.)]:
            self.assertEqual(model.step(s,2.),coupled(s,2.))
        cpp=cpp_source(model.payload)
        self.assertIn('spark_vector_rollout_v1',cpp)
        self.assertIn('const double t',cpp)
        self.assertNotIn('std::vector',cpp)

    def test_discovers_guard_not_supplied(self):
        model=synthesize(observations(True))
        self.assertEqual(model.payload['guard'],-2)
        for x in (-4.,-2.,-1.,0.,1.,3.):
            s=(2.,-1.,3.,-2.)
            self.assertEqual(model.step(s,x),coupled(s,x,True))

    def test_json_roundtrip_without_training_dependency(self):
        model=synthesize(observations(True))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json'
            save(path,model)
            loaded=load(path)
            self.assertEqual(loaded.payload,model.payload)
            self.assertEqual(loaded.step((1.,2.,3.,4.),2.),model.step((1.,2.,3.,4.),2.))

    @unittest.skipUnless(shutil.which('c++') or shutil.which('g++'),'no C++ compiler')
    def test_native_parity_on_new_vectors_and_long_sequences(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from vector_experiment import native_compile,native_step,native_rollout
        model=synthesize(observations(True))
        with tempfile.TemporaryDirectory() as tmp:
            lib=native_compile(model,Path(tmp)/'test.so')
            for x in (-5.,-2.,-1.,0.,1.,4.):
                s=(0.8,-0.4,1.2,-1.8)
                actual=native_step(lib,s,x)
                for a,b in zip(actual,model.step(s,x)):
                    self.assertAlmostEqual(a,b,places=10)
            rng=random.Random(2)
            seq=[float(rng.choice([-2,-1,0,1,2])) for _ in range(4096)]
            self.assertEqual(native_rollout(lib,(1.,0.,-1.,2.),seq),
                             model.rollout((1.,0.,-1.,2.),seq))


if __name__=='__main__':unittest.main()

"""M11 structural evolution, reproducibility and native C++ parity tests."""
import json
import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from spark.evolved_program import (expand_topologies,evolve,predict,rollout,
                                   cpp_source,load,save,validate,np)


def rule(state,x,w):
    a,b,c,d=state
    q=w*x*(a-b)
    return (x,a,c+q,d+c+q)


def samples(n,seed,gains,reset=False):
    rng=random.Random(seed)
    out=[]
    for i in range(n):
        state=tuple(rng.uniform(-1,1) for _ in range(4))
        x=float([-2,-1,0,1,2][i%5] if reset else [-1,0,1,2][i%4])
        w=gains[(i//(5 if reset else 4))%len(gains)]
        y=((0.,)*4 if reset and x==-2 else rule(state,x,w))
        out.append((state,x,w,y))
    return out


class EvolvedProgramTests(unittest.TestCase):
    def test_graph_mutations_expand_and_fitness_prune(self):
        nodes,gens=expand_topologies(4,depth=3,seed=5)
        self.assertEqual(gens,[6,21,56])
        self.assertEqual(len(nodes),84)
        self.assertIn((1,0,0,0,1,1),nodes)
        self.assertIn((0,1,0,0,1,1),nodes)
        if np is not None:
            nodes,gens=expand_topologies(4,depth=3,fitness_rows=samples(180,2,[.5,1,1.5]))
            self.assertEqual(gens,[6,21,48])
            self.assertEqual(len(nodes),76)

    def test_reject_untrusted_program_and_nonfinite_constants(self):
        base={'format':'spark-evolved-program-v1','registers':1,'generations':1,
              'guard':None,'outputs':[[[[1,0,0],1,1]]]}
        validate(base)
        for variant in (dict(base,guard=float('nan')),
                        dict(base,outputs=[[[[1,0,0],0,1]]]),
                        dict(base,outputs=[[[[1,0,0],1,0]]]),
                        dict(base,outputs=[[[[1,0,0],1,1],[[1,0,0],2,1]]]),
                        dict(base,outputs=[[[[1,0,0],1,1],[[],1,1]]]),
                        dict(base,format='unsafe'),dict(base,registers=99)):
            with self.assertRaises(ValueError):validate(variant)
        self.assertEqual(predict(base,(4.,),1.,.5),(4.,))
        with self.assertRaises(ValueError):predict(base,(float('nan'),),1.,.5)
        self.assertNotIn('std::vector',cpp_source(base))

    @unittest.skipIf(np is None,'training requires NumPy')
    def test_learns_shared_gain_with_unseen_gain_values(self):
        train=samples(360,123,[.5,1.0,1.5])
        candidate=evolve(train,max_generations=3,max_terms=8)
        self.assertIsNone(candidate.payload['guard'])
        self.assertEqual(candidate.payload['generations'],3)
        for state,x,w,y in samples(90,212,[-.75,.25,2.25]):
            for a,b in zip(candidate.step(state,x,w),y):self.assertAlmostEqual(a,b,places=9)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'equation.json';save(path,candidate)
            self.assertEqual(load(path).payload,candidate.payload)

    @unittest.skipIf(np is None,'training requires NumPy')
    def test_rejects_insufficient_expressive_depth(self):
        train=samples(360,312,[.5,1.,1.5])
        weak=evolve(train,max_generations=2,max_terms=8)
        heldout=samples(90,401,[-.75,.25,2.25])
        mse=sum((a-b)**2 for s,x,w,y in heldout for a,b in zip(weak.step(s,x,w),y))/360
        self.assertGreater(mse,0.01)

    @unittest.skipIf(np is None,'training requires NumPy')
    def test_train_only_guard_detection(self):
        candidate=evolve(samples(500,32,[.5,1.,1.5],reset=True),max_generations=3,max_terms=8)
        self.assertEqual(candidate.payload['guard'],-2.0)
        for state,x,w,y in samples(110,108,[-.75,.25,2.25],reset=True):
            for a,b in zip(candidate.step(state,x,w),y):self.assertAlmostEqual(a,b,places=9)

    @unittest.skipUnless(shutil.which('c++') or shutil.which('g++') or shutil.which('clang++'),'C++ compiler absent')
    def test_cpp_native_structural_parity(self):
        from tools.evolution_experiment import build_native,native_step,native_rollout
        model={'format':'spark-evolved-program-v1','registers':2,'generations':3,
               'guard':-2.0,'outputs':[
                 [[[0,0,1,0],1,1]],
                 [[[1,0,0,0],1,1],[[0,0,1,1],1,1],[[0,1,1,1],-1,1]]]}
        validate(model)
        class Wrapper:payload=model
        with tempfile.TemporaryDirectory() as folder:
            lib=build_native(Wrapper(),Path(folder)/'test.so')
            for w in (.25,-.5,1.75):
                for s,x in (((.2,.5),1.),((-.3,.7),-2.),((1.,0.),2.)):
                    expected=predict(model,s,x,w)
                    actual=native_step(lib,s,x,w)
                    for a,b in zip(actual,expected):self.assertAlmostEqual(a,b,places=10)
                xs=[-2.,0.,1.,0.,-1.,0.,0.]*40
                actual=native_rollout(lib,(.2,.5),xs,w)
                expected=rollout(model,(.2,.5),xs,w)
                for a,b in zip(actual,expected):self.assertAlmostEqual(a,b,places=10)


if __name__=='__main__':unittest.main()

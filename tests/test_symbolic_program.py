"""M8 source, search, serialization and native-code parity verification."""
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from spark.symbolic_program import (const, cpp_source, deserialize, evaluate,
                                    polynomial_program, serialize, synthesize, validate)
from fractions import Fraction

class SymbolicProgramTests(unittest.TestCase):
    def test_rejects_untrusted_operations(self):
        for bad in ({'op':'system','command':'rm -rf /'},
                    {'op':'const','num':1,'den':0},
                    {'op':'state','extra':1}):
            with self.assertRaises(ValueError):validate(bad)

    def test_expression_and_roundtrip(self):
        expr=polynomial_program([(1,0,Fraction(1)),(0,1,Fraction(1,2)),(2,1,Fraction(-1,2))])
        self.assertAlmostEqual(evaluate(expr,0.2,0.7),0.2+0.35*(1-0.04))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'expr.json'
            serialize(path,expr)
            self.assertEqual(deserialize(path),expr)
            data=json.loads(path.read_text());data['expression']={'op':'unknown'}
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):deserialize(path)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "offline synthesis needs numpy")
    def test_discovers_nonlinear_recurrence(self):
        import random
        rng=random.Random(303)
        examples=[(s,x,s+0.5*x*(1-s*s)) for s,x in
                  ((rng.uniform(-0.8,0.8),rng.uniform(-0.7,0.7)) for _ in range(150))]
        program=synthesize(examples,max_degree=3)
        self.assertLess(program.train_mse,1e-20)
        self.assertEqual(len(program.terms),3)
        for s,x in ((-1.1,0.9),(0.93,-0.7),(0.1,0.3)):
            self.assertAlmostEqual(program.step(s,x),s+0.5*x*(1-s*s),places=12)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "offline synthesis needs numpy")
    def test_discovers_parity_recursion(self):
        data=[(s,x,s^x) for _ in range(20) for s in range(2) for x in range(2)]
        program=synthesize(data,max_degree=2,square_free=True)
        self.assertEqual(len(program.terms),3)
        value=0
        for i in range(2000):
            value=program.step(value,i%3==0)
            self.assertIn(value,(0,1))

    @unittest.skipUnless(shutil.which('c++') or shutil.which('g++') or shutil.which('clang++'), 'C++ compiler absent')
    def test_cpp_native_parity(self):
        spec=importlib.util.spec_from_file_location('symbolic_exp',ROOT/'tools/symbolic_experiment.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        expr=polynomial_program([(1,0,Fraction(1)),(0,1,Fraction(1,2)),(2,1,Fraction(-1,2))])
        with tempfile.TemporaryDirectory() as tmp:
            lib=mod.cpp_compile(cpp_source(expr),Path(tmp)/'model.so')
            for s,x in ((0.3,0.2),(-0.7,0.9),(1.1,-1.0)):
                self.assertAlmostEqual(lib.spark_expr_step_v1(s,x),evaluate(expr,s,x),places=12)

if __name__=='__main__':unittest.main()

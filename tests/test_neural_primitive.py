"""M12 strict AST, learned primitive, safe native code and serialisation tests."""
import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from spark.neural_primitive import (HybridProgram,validate,fit_neural,build_hybrid,
                                  cpp_source,save,load,neural_scalar)


class NeuralPrimitiveTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.z=np.asarray([(float(i%9)/5-1.,float(i%7)/5-.7,float(i%5)/5-.4)
                           for i in range(100)])
        self.y=np.tanh(self.z[:,0] + .5*self.z[:,1])

    def test_learned_function_and_shared_symbolic_calls(self):
        net=fit_neural(self.z,self.y,width=6,epochs=120,seed=3)
        model=build_hybrid(net)
        self.assertEqual(model.parameter_count,55)
        s=(.2,-.1,.4,.6);x=.5;c=.9
        y=model.step(s,x,c)
        q=neural_scalar(net,(s[0]-s[1],x,c))
        self.assertAlmostEqual(y[2]-.6*s[2],q)
        self.assertAlmostEqual(y[3]-(.5*s[3]+.2*s[2]),q)

    def test_invalid_ast_rejected(self):
        import copy
        p=build_hybrid(fit_neural(self.z,self.y,width=4,epochs=10)).payload
        for bad in ({'op':'import'},
                    {'op':'state','index':200},
                    {'op':'const','num':1,'den':0},
                    {'op':'neural','module':'shell'},
                    {'op':'add','left':{'op':'neural'},'right':{'op':'input'}}):
            q=copy.deepcopy(p);q['args'][0]=bad
            with self.assertRaises(ValueError):validate(q)
        q=copy.deepcopy(p);q['neural']['w_hidden'][0][0]=float('nan')
        with self.assertRaises(ValueError):validate(q)

    def test_serialization_and_codegen(self):
        net=fit_neural(self.z,self.y,width=5,epochs=20)
        model=build_hybrid(net)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'p.json'
            save(path,model)
            self.assertEqual(load(path).step((1.,2.,3.,4.),.3,-.5),model.step((1.,2.,3.,4.),.3,-.5))
            cpp=cpp_source(model.payload)
            self.assertIn('spark_hybrid_step_v1',cpp)
            self.assertIn('const double q = spark_neural',cpp)
            self.assertEqual(cpp.count('const double q = spark_neural'),1)
            self.assertNotIn('torch',cpp)

    @unittest.skipUnless(shutil.which('g++') or shutil.which('c++') or shutil.which('clang++'), 'C++17 compiler not found')
    def test_python_native_parity(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from neural_function_experiment import native_compile,native_step,native_rollout
        model=build_hybrid(fit_neural(self.z,self.y,width=4,epochs=50))
        with tempfile.TemporaryDirectory() as folder:
            lib=native_compile(model,Path(folder)/'p.so')
            rng=random.Random(8)
            state=tuple(rng.uniform(-1,1) for _ in range(4))
            for _ in range(30):
                x,c=rng.uniform(-1,1),rng.uniform(-1,1)
                p=model.step(state,x,c)
                n=native_step(lib,state,x,c)
                self.assertLess(max(abs(a-b) for a,b in zip(p,n)),1e-12)
            xs=[rng.uniform(-1,1) for _ in range(200)]
            cs=[rng.uniform(-1,1) for _ in range(200)]
            predicted=model.rollout(state,xs,cs)
            native=native_rollout(lib,state,xs,cs)
            self.assertLess(max(abs(a-b) for a,b in zip(predicted,native)),1e-10)


if __name__=='__main__':unittest.main()

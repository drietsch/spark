"""Tests for learned conditional programs, safe serialization, and C++ generation."""
import random
import tempfile
import unittest
from pathlib import Path
from spark.symbolic_control import discover,save,load,validate_program,compiled_source
from spark.symbolic_program import np

class ConditionalTests(unittest.TestCase):
    def test_rejects_invalid_guards_and_unsupported_operations(self):
        for bad in (
            {"op":"if","guard":{"variable":"input","comparison":"system","value":0},
             "then":{},"else":{}},
            {"op":"import","code":"os.system"},
            {"op":"leaf","expr":{"op":"unknown"}},
            {"op":"if","guard":{"variable":"input","comparison":"lt","value":float("nan")},
             "then":{},"else":{}},
        ):
            with self.assertRaises(ValueError):validate_program(bad)

    def test_serialization_checks_before_load(self):
        expr={"op":"leaf","expr":{"op":"state"}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"control.json"
            save(path,expr)
            self.assertEqual(load(path),expr)
            path.write_text('{"format":"spark-conditional-program-v1","tree":{"op":"import"},"metadata":{}}')
            with self.assertRaises(ValueError):load(path)

    @unittest.skipIf(np is None,"training requires numpy")
    def test_reset_program_generalizes_outside_training_inputs(self):
        rng=random.Random(11)
        examples=[]
        for _ in range(400):
            s=float(rng.randrange(-7,8))
            x=float(rng.choice([-4,-3,-2,-1,0,1,2]))
            next_s=0.0 if x==-2 else s if x==-1 else s+x
            examples.append((s,x,next_s))
        program=discover(examples,max_degree=1,max_depth=2)
        self.assertLess(program.train_mse,1e-18)
        self.assertGreaterEqual(program.branches,2)
        for s in range(-8,9):
            for x in [-6,-5,-4,-3,-2,-1,0,1,2,3,4]:
                expected=0 if x==-2 else s if x==-1 else s+x
                self.assertAlmostEqual(program.step(s,x),expected)

    def test_cpp_is_conditional_math_not_lookup_or_generated_python(self):
        node={"op":"if",
          "guard":{"variable":"input","comparison":"eq","value":-2},
          "then":{"op":"leaf","expr":{"op":"const","num":0,"den":1}},
          "else":{"op":"leaf","expr":{"op":"state"}}}
        cpp=compiled_source(node)
        self.assertIn("?",cpp)
        self.assertIn("spark_control_rollout_v1",cpp)
        self.assertNotIn("std::vector",cpp)

if __name__=="__main__":unittest.main()

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class RuntimeTest(unittest.TestCase):
    def test_compilation_and_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model.idx"
            subprocess.run(["python3", str(ROOT / "tools/train_from_traces.py"),
                            str(ROOT / "examples/teacher_traces.jsonl"), str(model)], check=True)
            binary = Path(td) / "demo"
            subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(ROOT / "include"),
                            str(ROOT / "runtime/spark_runtime.cpp"),
                            str(ROOT / "tools/spark_demo.cpp"), "-o", str(binary)], check=True)
            stdout = subprocess.check_output([str(binary), str(model), "10", "20", "30"], text=True)
            self.assertIn("next=40 order=3", stdout)
            # Backoff from unseen order-3 context to known order-2 prefix.
            stdout = subprocess.check_output([str(binary), str(model), "99", "20", "30"], text=True)
            self.assertIn("order=2", stdout)

if __name__ == "__main__": unittest.main()

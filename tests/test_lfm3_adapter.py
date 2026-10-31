"""Mock server proves the HTTP adapter and SPARK integration, NOT actual LFM3 accuracy."""
import importlib.util
import json
import math
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

adapter = load("collect_llama_teacher", ROOT / "tools/collect_llama_teacher.py")

class MockLlama(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            self.reply({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            words = body["content"].split()
            result = [1] + [100 + sum(w.encode()) for w in words]
            self.reply({"tokens": result})
        elif self.path == "/completion":
            prompt = body["prompt"]
            candidate = (prompt[-1] + 7) % 2000
            self.reply({"probs": [{"id": candidate, "logprob": math.log(.7),
                      "top_logprobs": [{"id": candidate, "logprob": math.log(.7)},
                                       {"id": candidate + 1, "logprob": math.log(.2)}]}],
                        "tokens": [candidate], "content": "hi"})
        else:
            self.send_error(404)

    def reply(self, data):
        encoded = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class IntegrationTest(unittest.TestCase):
    def test_probability_parser(self):
        self.assertEqual(
            adapter.top_from_response({"completion_probabilities": [
                {"top_logprobs": [{"id": 42, "logprob": math.log(.5)}]}]}, 4, 1000),
            [{"token": 42, "weight": 500}])
        with self.assertRaises(ValueError):
            adapter.top_from_response({"probs": [{"top_probs": []}]}, 4, 1000)

    def test_end_to_end_mock(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockLlama)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prompts = root / "prompts.txt"
                prompts.write_text("alpha beta gamma delta epsilon zeta\n"
                                   "theta iota kappa lambda mu nu xi\n"
                                   "one two three four five six seven\n"
                                   "spring summer autumn winter warm cold dry\n", encoding="utf-8")
                train, holdout, idx = root / "train.jsonl", root / "holdout.jsonl", root / "model.idx"
                base = ["python3", str(ROOT / "tools/collect_llama_teacher.py"),
                        "--url", f"http://127.0.0.1:{server.server_port}",
                        "--prompts", str(prompts), "--train-out", str(train),
                        "--eval-out", str(holdout), "--min-prefix", "3",
                        "--max-per-prompt", "2"]
                capture = subprocess.run(base, capture_output=True, text=True, check=True)
                report = json.loads(capture.stdout)
                self.assertGreater(report["train_traces"], 0)
                self.assertGreater(report["eval_traces"], 0)
                subprocess.run(["python3", str(ROOT / "tools/train_from_traces.py"),
                                str(train), str(idx)], check=True, capture_output=True)
                eval_result = subprocess.run(["python3", str(ROOT / "tools/evaluate_teacher_traces.py"),
                                              str(idx), str(holdout)], check=True,
                                             capture_output=True, text=True)
                self.assertEqual(json.loads(eval_result.stdout)["held_out_traces"], 2)
                demo = root / "demo"
                subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(ROOT / "include"),
                                str(ROOT / "runtime/spark_runtime.cpp"),
                                str(ROOT / "tools/spark_demo.cpp"), "-o", str(demo)], check=True,
                               capture_output=True)
                example = json.loads(train.read_text().splitlines()[0])
                output = subprocess.check_output([str(demo), str(idx),
                                                  *[str(t) for t in example["context"]]], text=True)
                self.assertIn(f"next={example['targets'][0]['token']}", output)
                self.assertIn("order=", output)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__": unittest.main()

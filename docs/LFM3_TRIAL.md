# LFM3-1.7B GGUF -> SPARK v0.1 trial

**Status:** Adapter and offline mock-server integration tests work. **No real LFM3
teacher execution has been performed here** because this runtime cannot reach
Hugging Face and does not have llama.cpp or the GGUF weights available.

## 1. Start a real LFM3 teacher

Use `llama.cpp` **v0.8.0 / b11960 or later** (LFM3 support). Obtain the
server binary from an official current build of llama.cpp. Example, from a
source checkout:

```sh
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-server -j
./llama.cpp/build/bin/llama-server \
  -hf LiquidAI/LFM3-1.7B-GGUF:Q4_K_M \
  --host 127.0.0.1 --port 8080 -c 4096 -ngl 0
```

The GGUF file is about 1.1 GB and is downloaded on first launch. The example
runs on CPU (`-ngl 0`). Make sure you understand the model's **LFM Open
License v1.0**, especially if distributing SPARK outputs derived from it.

## 2. Capture teacher behavior and compile a SPARK index

From the SPARK project root, in a different terminal:

```sh
python3 tools/collect_llama_teacher.py \
  --url http://127.0.0.1:8080 \
  --prompts examples/lfm3_prompts.txt \
  --train-out outputs/train.jsonl \
  --eval-out outputs/eval.jsonl \
  --top-k 8 --max-per-prompt 6

python3 tools/train_from_traces.py outputs/train.jsonl outputs/lfm3.idx
cmake -S . -B build && cmake --build build -j
python3 tools/evaluate_teacher_traces.py outputs/lfm3.idx outputs/eval.jsonl

# Predict a token ID from a prefix captured in train.jsonl:
python3 - <<'PY'
import json, subprocess
trace = json.loads(open('outputs/train.jsonl').readline())
cmd = ['./build/spark_demo', 'outputs/lfm3.idx',
       *map(str, trace['context'])]
print('command:', ' '.join(cmd))
subprocess.run(cmd, check=True)
print('teacher best token:', trace['targets'][0]['token'])
PY
```

The capture uses real `llama.cpp` `/tokenize` token IDs and `/completion`
**pre-sampling** top-k log-probabilities. Numeric weights in the SPARK index
are rounded scores proportional to `exp(logprob)`. They are **not** full
probabilities. Teacher traces are split by **prompt**, not by prefix.

### Important limitations

- The implementation is a **four-token exact context lookup**, not a semantic
  distillation system yet. It cannot generalize reliably to unseen contexts.
- LFM3 teacher uses the **entire supplied prefix**; SPARK only retains the
  last four token IDs. This loss of information is intrinsic to the baseline.
- The demonstration input is **plain text completion**, not an LFM3 templated
  chat session. To capture chat behavior, add `/apply-template` integration
  with the model's chat template and handle any reasoning/control tokens.
- The current SPARK runtime provides **no automatic EOS handling, fallback
  token, or calibrated perplexity**. On unseen contexts, it returns ENOMATCH.
- Reports are **teacher top-1 agreement**, not proof of intelligence or
  instruction-following ability. A full model evaluation requires separate
  held-out corpora and generation-level tests.
- Training from weights without executing the teacher is **not implemented**;
  SPARK LLVM/MLIR dialect/codegen is still a proposed interface.
- Only use your own or properly licensed training text. Bind the server to
  loopback and do not expose the inference or trace endpoints publicly.

## 3. Suggested follow-on experiments

After the baseline is running with real traces, measure `coverage` and
`top1_agreement_all` on disjoint corpora. Compare index size to an order-1
unigram baseline, then implement an approximate memory index that can return
matches for unseen token contexts. Save tokenization revision and GGUF hash
in the artifact manifest before doing larger experiments.

Or, after starting the teacher, run the same pipeline with a single command:

```sh
bash scripts/run_lfm3_trial.sh
```

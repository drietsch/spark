#!/usr/bin/env bash
# Run this after starting an actual LFM3 llama.cpp server on localhost:8080.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_URL="${SPARK_TEACHER_URL:-http://127.0.0.1:8080}"
mkdir -p outputs
python3 tools/collect_llama_teacher.py \
    --url "$BASE_URL" \
    --prompts examples/lfm3_prompts.txt \
    --train-out outputs/lfm3_train.jsonl \
    --eval-out outputs/lfm3_holdout.jsonl \
    --top-k 8 --max-per-prompt 6
python3 tools/train_from_traces.py outputs/lfm3_train.jsonl outputs/lfm3.idx
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 tools/evaluate_teacher_traces.py outputs/lfm3.idx outputs/lfm3_holdout.jsonl
python3 - <<'PY'
import json
import subprocess
from pathlib import Path
trace = json.loads(Path('outputs/lfm3_train.jsonl').read_text().splitlines()[0])
cmd = ['./build/spark_demo', 'outputs/lfm3.idx', *map(str, trace['context'])]
print('SPARK example:', ' '.join(cmd))
subprocess.run(cmd, check=True)
print('Teacher top-1:', trace['targets'][0]['token'])
PY

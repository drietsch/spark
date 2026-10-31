#!/usr/bin/env bash
# On an internet-connected local machine with transformers + torch installed.
set -euo pipefail
cd "$(dirname "$0")/.."
INPUT="${1:-examples/modernbert_masks.jsonl}"
OUTPUT="${2:-artifacts/m4}"
DEVICE="${SPARK_TEACHER_DEVICE:-cpu}"
mkdir -p "$OUTPUT"
python3 tools/modernbert_teacher.py "$INPUT" "$OUTPUT/teacher.jsonl" \
    --device "$DEVICE" --emit-hidden --topk 16
python3 tools/modernbert_semantic.py compile "$OUTPUT/teacher.jsonl" "$OUTPUT/student"
cmake -S . -B build
cmake --build build -j2
python3 tools/modernbert_benchmark.py "$OUTPUT/teacher.jsonl" "$OUTPUT/student" \
    --build build --repeat 10 --output "$OUTPUT/report.json"
echo "M4 benchmark report: $OUTPUT/report.json"

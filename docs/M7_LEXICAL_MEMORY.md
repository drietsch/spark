# M7: larger lexical associative memory and train-only fusion

## Measured result (real ModernBERT, 31 October 2026)

This is a **masked-token agreement** benchmark against
`answerdotai/ModernBERT-base`, not ground-truth word reconstruction,
Transformer-weight conversion, or autoregressive inference.

GitHub Actions: https://github.com/drietsch/spark/actions/runs/41243701560

The corrected M6 teacher traces contain **5,000 WikiText-2 train masks** and
**500 WikiText-2 validation masks**. Their `[CLS]`, `[MASK]`, and `[SEP]`
positions are validated. M7 reuses those immutable teacher traces.

| Student | Top-1 agreement | Correct / 500 |
|---|---:|---:|
| M6 256-bit signatures | 0.2% | 1 |
| M6 teacher-supervised sparse | 4.6% | 23 |
| M7 lexical n-grams, standalone | 15.0% | 75 |
| **M7 lexical + supervised sparse fusion** | **15.2%** | **76** |
| Train-majority label | 2.4% | 12 |

M7 fused 95% Wilson interval: **12.32%–18.61%**. Improvement over M6
sparse: **+10.6 percentage points**, about **3.30x** the agreement rate.
The gain is primarily from additional lexical training data; fusion itself
contributes only **one correct prediction** beyond lexical-only.

## Architecture

1. **Offline**: tokenize WikiText-2 `train` and collect up to **550,000**
   alphabetic token positions. This includes some BPE word pieces, not only
   whole words. Build nine kinds of bidirectional context-to-token postings,
   including unigram left/right, bigrams, trigrams and cross-mask pairs.
2. **Offline**: fit the M6 teacher-supervised sparse token head on the 5,000
   teacher train traces.
3. **Tuning**: use a deterministic inner fold (**4,585 train / 415 tuning**)
   from teacher TRAIN to choose smoothing, feature specificity and fusion
   weight. The lexical memory was trained on full WikiText train, which also
   contains the inner-fold source sentences. This is *not* an independent
   inner test; the final official WikiText-2 validation split is independent.
4. **Inference**: look up at most nine lexical feature posting lists, combine
   conditional token-count votes with supervised sparse votes, and rank token
   IDs. **No dense matrix multiplications or ModernBERT execution at student
   inference time.**

The winning lexical fusion weight was 0.75, smoothing 2.0 and specificity
1.3, selected **without looking at the official 500 validation answers**.
The lexical memory has 2,356,188 feature types, 3,721,756 label postings,
and 26,784 observed candidate tokens. Building it took 17.165 seconds on
the GitHub runner, **excluding downloads, initial model tokenizer loading,
and report evaluation**.

### Reproduce using the frozen corrected M6 artifact

```bash
python -m pip install 'transformers>=4.48,<5' 'datasets>=3,<5'
# Download M6 real-model artifact from:
# https://github.com/drietsch/spark/actions/runs/41234988861/artifacts/13056190645
# Unpack teacher.jsonl as artifacts/modernbert-m6/teacher.jsonl
python tools/m7_lexical_benchmark.py artifacts/modernbert-m6/teacher.jsonl \
  --max-lexical-tokens 550000 \
  --output artifacts/modernbert-m7/report.json
```

See `.github/workflows/modernbert-m7-lexical.yml` for a GitHub Actions
recipe, including an artifact downloader that requires the source artifact
to remain available (7-day retention). To regenerate the teacher traces
after expiration, use the M6 corrected benchmark workflow.

## Remaining limitations

- **15.2% is still low** relative to a real masked-language model. This
  architecture has not demonstrated model equivalence or language reasoning.
- M7 measures Python reference inference, **not** native C++ latency. Native
  compilation of its large postings is a separate task.
- The two M7 methods share predictions and validation examples; their
  differences should be evaluated with paired tests before making
  statistical claims about the 0.2-point fusion increment.
- The lexical-memory model is built in RAM and not yet exported as a
  versioned, validated native binary artifact.
- The corpus contains real text; keep all validation text out of lexical
  training and all teacher validation labels out of tuning.

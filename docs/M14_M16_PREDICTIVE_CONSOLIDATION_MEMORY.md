# M14–M16: Predictive state, neural-to-symbolic consolidation and program memory

## Objective

Move SPARK from provided-state equations and a continuously trained neural
function toward (a) discovering **predictive state from observations**, (b)
selectively converting live neural functions into validated mathematical
expressions, and (c) executing program-controlled memory over long sequences.
These modules are incremental research prototypes, **not** a general-purpose
language model or evidence of outperforming Transformers.

## M14 — learn predictive state without hidden-state labels

`fit()` receives only independently generated sequences of `(y_t, u_t)`;
there are no physical hidden-state annotations or direct state-transition
labels. It searches orders 1–4 on a training-only group holdout, then fits a
regularized autoregressive controlled predictor. The chosen state consists of
learned-length observable history. The emitted C++ computes the recurrence
with live mutable coefficients, and supports prequential online error updates.

The experiment uses a *linear, fully identifiable AR(2)* synthetic process.
It is **not** arbitrary latent-state discovery, nonlinearity discovery, or a
partially observed POMDP solver. The separate missing-observation test uses
model predictions to impute absent readings. A distribution shift changes the
input-response coefficient; reported online/frozen errors are scored
**before** updating on that observation.

## M15 — evidence-gated neural-to-symbolic consolidation

Starting from M13's live dense neural function, M15 collects behavior samples
and searches rational polynomial expressions with bounded terms and degree.
Candidate expression coefficients and structure are selected from training
samples. Promotion requires:

1. Low error against **independent observed targets**, not merely matching the
   source neural function.
2. Error no worse than the original neural function by more than a bounded ratio.
3. Adequate multi-step recurrent-rollout behavior.

The `ConsolidationManager` automatically attempts promotion when a minimum
observation count and evaluation interval are reached, and rolls back to the
underlying live neural function after sustained prediction drift. The proposed
expressions are compiled to native C++ when accepted. A deliberately
incorrect candidate is rejected in the benchmark.

Caveat: the nonlinear function is trained from a synthetic known law, and the
outer recurrence is supplied. The current system does **not** evolve a global
program library or discover new categories of functions from raw observations.

## M16 — bounded algorithmic memory

A C++17 and Python memory state-machine implements explicit `WRITE`, `READ`,
`ADD`, `ERASE`, `NOOP` operations with LRU eviction. A compiled symbolic
program can use these operations to maintain exact values across many
intervening steps. The benchmark includes independent random sequences,
29,356 exact read queries, 4,096 intervening no-op events before a recall,
Python/native execution parity, and an evict/reuse test.

**Critical distinction**: the instruction policy is **provided**, not learned.
Thus 100% correctness here demonstrates a bounded programmable memory
substrate, not self-taught algorithmic reasoning, instruction interpretation,
or generalization to unseen language tasks. Capacity is 32 key/value records;
information beyond that limit can be evicted.

## Reproduce

```bash
python -m pip install numpy
python -m unittest discover -s tests -p test_research_m14_m16.py -v
OPENBLAS_NUM_THREADS=1 python tools/research_experiment.py --output artifacts/m14-16
```

The C++17 compiler (`c++`, `g++` or `clang++`) is required. The benchmark
writes `report.json`, learned predictive coefficients, generated C++ source,
and native shared libraries for every milestone.

## Experimental interpretation

- M14 order-2 vs order-1 is a controlled linear-memory comparison, not a GRU,
  Mamba/SSM, Transformer or attention benchmark.
- M15 validates symbolic replacement of one small neural function and catches
  an intentionally bad candidate. It does not guarantee robust consolidation
  under arbitrary drift or catastrophic forgetting.
- M16 gives a memory capability that a stateless last-value baseline cannot
  match, but the read/write policy is manually specified. Memory exactness
  does not measure learned cognitive ability.
- No throughput or intelligence superiority claims are made for M14–M16.

## Next research gates

Train partially observed nonlinear state models with delayed gradients; learn
which observations warrant memory writes, compare matched-budget GRUs/SSMs;
measure latency and memory including online training; test symbolic
consolidation against distribution shifts and privacy-safe lifelong tasks.

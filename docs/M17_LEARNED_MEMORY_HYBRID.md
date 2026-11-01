# M17 — Learn when to remember, then adapt a neural mathematical function

**Status:** constrained synthetic research baseline, not a trained language model.
M17 starts integrating M14's prediction-before-observation protocol, M15's
validated symbolic replacement, M16's bounded native memory operations, and
M13's live-plastic neural mathematical function.

## The central change

M16 memory took explicit `WRITE`/`READ` instructions. M17's write/retain
routing policy is **trained from delayed read targets**. There are no separate
training labels for write decisions, and updates use analytic gradients
propagated through the memory state. The input includes an obvious binary cue
signal correlated perfectly with relevance. Hence the current challenge is
learning a simple gating rule, **not** autonomous semantic relevance detection.

For memory key `k`, observation `e=(signal,key,value)`:

```
g = sigmoid(theta0 + theta1 * signal + theta2 * value)
m[k] <- (1-g) * m[k] + g * value  # differentiable training
m[k] <- value if g > 0.5 else m[k]  # compiled inference, hard gate
```

At read time, a query supplies the key and a supervised target for the
remembered value. All predictions precede receipt of their target.
Memory is reset between episodes; training occurs only on short episodes.
The state has four bounded memory slots; the key and read-request semantics
are still given explicitly.

The memory output is then passed into a **live neural mathematical function**:

```
q = N_theta(memory_value, context, memory_value * context)
prediction = memory_value + q
```

The plastic coefficients of `N` update from the full prediction error after
feedback. The target function reverses halfway through the stream, without
supplying a mode label. A separate frozen copy receives the same observations.
This probes immediate adaptation, not language comprehension.

M15 is exercised as a third gate: a polynomial candidate is fitted only from
older observations, compared on newer holdout samples, and checked on long
rollouts. The current candidate is deliberately **rejected** because its
long-horizon error exceeds the threshold, despite good one-step MSE.
The report preserves that rejection as a negative result.

## Reproduction

```
python -m unittest discover -s tests -p 'test_integrated_research.py' -v
OPENBLAS_NUM_THREADS=1 python tools/integrated_experiment.py --output artifacts/m17
# Optional larger neural recurrent baseline; requires PyTorch
python tools/gru_memory_baseline.py --output artifacts/m17/gru.json --epochs 25
```

C++17 is required for the compiled memory ABI; NumPy for consolidation; PyTorch
only for the optional GRU baseline, not the SPARK runtime.

## Evaluation

* Router trained on 450 short episodes with 4 keys, 3 delayed reads per episode.
* Router tested on 80 separate long-delay episodes (1,024–4,096 distractors
  per interval) and 80 separate short episodes; no test data in fitting.
* Native C++ and Python execute the same learned hard-gate policy with mutable
  theta coefficients and bounded memory.
* Nonlinear neural function trained online on 1,300 query observations, then
  evaluated prequentially on 1,100 query observations under reversed dynamics.
  Last-200 MSE is compared with a copy whose weights are frozen after phase 1.
* An always-write and a never-write control run on exactly the same queries.
* Optional GRU baseline (769 parameters) receives the same event features and
  query labels as SPARK. It has a different computational and training budget,
  so its accuracy **cannot establish superiority**; 25 epochs should be
  treated as a sanity check, not an optimized competitive benchmark.

## Critical limitations

1. Binary `signal` directly encodes which event should be retained. Learning
   from query labels instead of routing labels is an advance over M16's explicit
   memory instructions, but the latent write rule is trivial.
2. Memory is fixed-capacity and key-addressed. Read routing and key discovery
   are not learned. Hard gating at deployment is more accurate over long delays
   than the differentiable training equation; results distinguish both.
3. Online plasticity has supervised query targets, a synthetic smooth function
   and no catastrophic-forgetting evaluation beyond the one distribution shift.
4. M15's candidate is rejected; successful automated neural-to-symbolic
   consolidation has not been established on this integrated task.
5. M17 does not include realistic natural language, task-defined reasoning,
   autonomous memory policy synthesis or a compute-matched Transformer study.
   The GRU baseline is explicitly non-compute-matched.
6. Native parity here concerns the learned memory function. The neural
   mathematical primitive was verified with native online learning in M13;
   M17 orchestrates memory and plasticity together in Python.

**Next falsifiable research target:** remove the oracle-like binary event cue,
learn information value from context and delayed rewards, support unknown
keys and variable capacity, then compare against well-tuned GRU/SSM/attention
models using identical streaming data, compute budgets and repeated seeds.

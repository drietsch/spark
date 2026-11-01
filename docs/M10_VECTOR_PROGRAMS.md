# SPARK M10: learned multi-register mathematical programs

**Purpose:** replace scalar recurrent equations with learned coupled, reusable
program expressions. This is a new, **constrained** symbolic state-machine
prototype—not a language model, Transformer replacement, or model-weight
conversion.

## Architecture

The learned function maps a vector of internal registers and a scalar input to a
new vector:

\[\mathbf{s}_{t+1}=F_P(\mathbf{s}_t,x_t)\]

`F_P` consists of rational-coefficient polynomial expression trees (degree at
most 2 in this experiment), plus a globally shared optional input-equality
guard. Its AST is serialized as JSON; the compiler emits fixed C++17 arithmetic,
conditionals and a roll-forward loop. No input-dependent table lookups or learned
weight matrices are required. The compiler gives recurring monomials unique
local variables and reuses them across register updates.

**Learning:** train-only greedy sparse feature selection fits each target
register; rational constants are selected from bounded denominators. A separate
train-only internal holdout selects whether to include a guard, then final
expressions are refitted on the entire train set. All polynomial candidate
terms, branch types, maximum degree and register count are constrained by the
experimenter. This does not learn arbitrary programs or discover hidden state
from unlabelled sequences.

## Discovered rule examples

The experiments sample from a deliberately unknown-to-synthesizer rule. The
learning code receives `(state-vector, input, next-state-vector)` examples,
not the generator source. In both tasks the resulting learned behavior is
algebraically equivalent (up to associativity) to:

```
q = (s0 - s1) * x
s0' = x
s1' = s0
s2' = s2 + q
s3' = s3 + s2 + q
```

The second task adds an unknown reset event:

```
if x == -2: s' = (0, 0, 0, 0)
else: same four equations as above
```

The two outputs `s2'` and `s3'` reuse `s0*x` and `s1*x`. Shared intermediate
expressions appear once in generated machine-code input functions; this is
**common-subexpression reuse**, not general hierarchical planning.

## Verified evaluation gates

* 950 train transitions per task (synthetic, independently sampled states).
* 350 wider-range held-out transitions per task, never used in training/tuning.
* 48 separate unseen rollouts per task (lengths 32, 128, 1,024, 4,096).
* Python-vs-C++ step and sequence parity; no neural runtime.
* Full-vector affine least-squares baseline, trained on identical data.
* Exactly validated AST and safe JSON deserialization; no arbitrary code ops.

Run:

```bash
python -m pip install numpy
python -m unittest discover -s tests -p 'test_vector_program.py' -v
python tools/vector_experiment.py --output artifacts/symbolic-m10
```

Requires C++17 for native compilation. CI performs these checks on Linux.
A report and generated `.program.json`, `.cpp` and `.so` files are produced.

## Limitations

The task creators intentionally lie inside the synthesizer's grammar, the
transitions are noiseless, and all state variables are directly observed.
Comparing to an affine regressor is **not** a compute-matched Transformer
benchmark. This experiment establishes vector symbolic synthesis and compiled
function reuse only. Further steps require partial observability, noisy
sequences, hierarchical program calls, learned state discovery, benchmarks
against recurrent/SSM/attention models, and natural-language tasks.

# M9 — Conditional mathematical programs (architectural research)

SPARK's long-term goal is to discover compiled mathematical computation
rather than approximate a pretrained Transformer using token-index memories.
M8 learned rational-polynomial scalar recurrences. M9 adds learned branching
structure and reusable, natively compiled recurrent programs.

## Model

A program maps current state and input to next state:

    state_next = P(state, input)

The typed, validated AST contains conditional nodes (input == event, or
input/state/(state+input) less than a threshold) and leaves containing M8's
rational polynomial expressions. Training discovers conditions and equations
from transition examples. A bounded lookahead repairs greedy splits when
rare discrete events need special handling; structural cost favors smaller
exact programs. This is a restricted grammar search, not open-ended invention.

Native inference translates the AST into a C++17 function with arithmetic,
comparisons and branches. There are no neural matrix products, trained weight
arrays, nearest-neighbor lookups or teacher-model calls. The same compiled
function is reused at every step in a sequence.

## Reproduce

    python -m pip install numpy
    python -m unittest discover -s tests -p 'test_symbolic*.py' -v
    python tools/control_experiment.py --output artifacts/symbolic-m9

Requires a C++17 compiler. Outputs learned JSON program trees, generated
C++ source, shared libraries and a JSON report.

## Experiments

Synthetic hidden generators create labeled examples only; the solver gets
(current_state, input, next_state) tuples, not the generating rule.

* Resettable accumulator: special input events reset or hold state; all other
  discrete events add to state. 700 training transitions.
* Saturating counter: sum is clamped at unknown boundaries, requiring two
  learned comparisons. 1,300 training transitions.
* Conditional nonlinear recurrence: one input event resets state; other
  inputs follow a cubic recurrence. 650 training transitions.

Each task evaluates on 300 independent wider-range transitions and 48
unseen sequences of length 32, 128, 1,024 or 4,096, evaluated with both
Python and native C++. Least-squares affine and unbranched cubic baseline
models are fitted on the same training rows. Held-out outcomes are never
used to choose program structure.

## Interpretation

These are noiseless synthetic tasks whose rules lie within the supplied
search grammar. Exact recovery demonstrates learning conditional recurrence
and length extrapolation on these tasks, not general reasoning or superiority
to Transformers. The baselines are regression models, NOT Transformers.
Native performance on scalar updates is not comparable to Transformer token
throughput. Numpy is used offline for fitting rational coefficients.

Next research steps: noisy and partially observed learning, interacting
registers, hierarchical subprogram reuse, dynamic memory, and matched
computational comparisons against RNNs, state-space models and Transformers.

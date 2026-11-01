# SPARK M21 — Joint address/write program discovery with live neural value functions

M21 removes the separate all-write address-calibration dataset in M20. Every
candidate address projection is paired with a discovered write/retain expression
and scored on **the same delayed-query targets**. A bounded coarse-to-fine
search retains the best address/program combinations, then refines candidate
expression structure. After selection, a compact **runtime-plastic** gated-dense
neural mathematical function learns to interpret stored observations based on
**the resulting query error**, not on teacher-provided neural-function targets.

This is *alternating, coordinated learning*, **not** a general fully
end-to-end differentiable structure search. The address and Boolean policy are
selected jointly; neural values are learned next and may adapt while running.

## Mathematical computation

For a write observation `e=(u,v,value,a,b)`:

    address = round(c_u*u + c_v*v)
    if BooleanProgram(a,b):
        memory[address] = (value,a,b)

For a query with raw identity `(u,v)`:

    z = memory[address]  # stored observation, not a precalculated network output
    prediction = z.value + N_theta(z.value,z.a,z.b)
    # only after emitting/scoring prediction:
    theta <- theta - learning_rate * gradient((prediction - target)^2)

Here `N_theta` is a small, context-gated dense mathematical function from M13:

    h_j = tanh(dot(w_hj,z) + b_hj)
    g_j = sigmoid(dot(w_gj,z) + b_gj)
    N_theta(z) = b_out + sum_j w_out_j * h_j * g_j

Unlike M12, weights are **mutable runtime data** and online gradients run
inside the compiled C++ program. The learned address and Boolean expression
are validated ASTs and compiled arithmetic; neither requires token lookup
memories or a neural inference framework. The memory retains the raw
observation, so updates to the live function can reinterpret *older* memories.

## What is learned from supervision

Synthetic training episodes expose `(u,v,value,a,b)` for writes and explicitly
signaled read requests with delayed targets. They **do not expose** hidden
integer addresses, relevant-write labels, or individual neural function values.
The only selection objective is delayed query error. The learner uses 8
candidate address projections and a bounded 1–3 predicate Boolean grammar;
its search may use 2-atom programs as a quick surrogate before full refinement.

A positive initial hidden nonlinear value function and its negative after an
unannounced change are used **only inside the data generator**. That change is
not given to the learner. The experiment compares prequential online updates
against an otherwise identical frozen copy.

## Reproduction

```bash
python -m unittest discover -s tests -p 'test_joint_plastic_memory.py' -v
python tools/joint_plastic_experiment.py --output artifacts/m21
```

Requires C++17; Python stdlib only. The benchmark saves the learned program
and parameters, compiled C++, shared library, native parity metrics and JSON
report. GitHub CI runs both the regression suite and full benchmark.

## Critical limits

1. The hidden environment is deterministic, synthetic, and intentionally
   inside the address and Boolean program grammars; the nonlinear component
   is a simple tanh residual.
2. Joint structural search uses a raw-value proxy objective before online
   neural fitting. It does **not** optimize address/program topology and all
   neural weights in one globally differentiable optimization.
3. Query target values and read-request timing are explicitly supervised;
   there is no natural-language understanding or autonomous discovery of
   which future questions matter.
4. Memory capacity and neural maximum width are preallocated. Hidden-unit
   resizing is supported in the M13 component but not autonomously evolved
   by M21.
5. Longer unseen sequences test policy reuse, not arbitrary compositional
   reasoning. No compute-matched GRU, SSM or Transformer comparison is made.
6. Native execution parity checks numerical semantics, not real-time
   deadlines; the runtime is not a hard real-time system.

A meaningful next test would involve nonstationary, partially observable
memory keys, multiple nonlinear value modes, endogenous querying, and fair
resource-matched recurrent baselines.

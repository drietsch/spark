# M12 — A trainable dynamic dense function as a first-class SPARK expression

This prototype implements the proposed architecture, not Transformer distillation.
A tiny gated dense neural function `q = N(s0-s1, x, context)` is embedded as
one **shared mathematical subexpression** within a typed four-register symbolic
state-update program. The `q` function runs as direct compiled arithmetic
(`tanh`, sigmoid, additions and multiplications) with all learned numerical
constants embedded in the C++ machine-code source. No PyTorch, inference
framework, dynamic weight matrix, teacher model or nearest-neighbor memory
is used in the C++ runtime.

## Architecture

`h = tanh(W_h z + b_h)`

`g = sigmoid(W_g z + b_g)`

`q = b_o + sum_j(W_o[j] * h[j] * g[j])`

The gate responds dynamically to context, making this a small adaptive dense
function. Width 12 gives **109 learned scalars** (3×12×2 connections,
12×2 hidden/gate biases, 12 readout coefficients, 1 readout bias).
Those **are learned weights**, albeit tiny; compilation does not eliminate
learned numerical information. The experiment trains with NumPy and manual
backpropagation/Adam. No neural training package is used at inference.

The engine represents an AST containing an explicit `neural` operation whose
output is reused by multiple symbolic register expressions. In this experiment
symbolic recurrence is supplied, not automatically discovered:

```
q   = N(s0 - s1, x, context)
s0' = x
s1' = s0
s2' = 0.6*s2 + q
s3' = 0.5*s3 + 0.2*s2 + q
```

The synthetic teacher subfunction is a combination of nonlinear expressions;
q labels are provided to the neural learner during training. This is a
function-composition proof of concept **not** end-to-end autonomous program
synthesis, hidden-state discovery or superiority to neural sequence models.

## Reproduce

```
python -m pip install numpy
python -m unittest discover -s tests -p 'test_neural_primitive.py' -v
python tools/neural_function_experiment.py --output artifacts/m12
```

A C++17 compiler is required for the native benchmark. The runner trains from
3,000 synthetic transition/function examples, then evaluates on 600 entirely
separate transitions using context values absent from training. It checks
native/python step parity, 32 unseen rollouts up to 4,096 steps, throughput,
and three baselines: a quadratic symbolic approximation of the same `q` under
an identical recurrence, a standalone 180-parameter neural transition model,
and a fitted affine full-vector transition model.
The report records exact benchmark metrics. Native throughput is recurrent
state updates, **not tokens/s**.

## Limitations / next steps

- Algebraic structure and neural function arguments are given in advance;
  neither is discovered by M12.
- The neural subfunction receives direct synthetic function-target supervision.
- The dataset comes from a known smooth expression; it is not a complex
  partially observed language or reasoning task.
- Gates produce input-conditional computation but all hidden units are still
  evaluated; this is *dynamic output gating*, not weight synthesis or dynamic
  sparse execution.
- The next task is to learn subfunction placement, its input signature,
  and surrounding symbolic recurrence jointly; then compare against
  parameter-matched recurrent and attention baselines on real tasks.

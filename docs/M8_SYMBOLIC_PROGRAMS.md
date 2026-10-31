# M8: SPARK symbolic computational core (architectural pivot)

## Mission

SPARK should discover **compiled mathematical programs** which define a model's
state transitions, rather than distill a Transformer into sparse token memories.
An equation and its control flow are the model; numerical constants may be
learned and embedded in the compiled program. Weight matrices and pretrained
Transformer operations are not a requirement. Old M1–M7 associative lookup
experiments remain independent baselines, **not** the SPARK architecture target.

## First executable proof

A typed expression AST supports `state`, `input`, rational `const`, `add`,
`mul`, and `neg` operations. Program discovery searches constrained polynomial
structures of degree 0–3, fits candidate constants from examples, snaps them
to compact rational values, penalizes structure size, and serializes a
versioned expression tree. The native compiler emits safe C++17 directly from
validated nodes, without `eval`, arbitrary code injection, tensors, or a
runtime vocabulary/memory lookup.

Minimal example:

```sh
python -m pip install numpy
python tools/symbolic_experiment.py --output artifacts/symbolic-m8
python -m unittest discover -s tests -p 'test_symbolic_program.py' -v
```

The experiment must run on a machine with a C++17 compiler.
Output includes `physics.expr.json`, `parity.expr.json`, generated native C++
files, shared libraries, and `report.json`.

### What it learns

1. A nonlinear scalar transition generated from
   `state_next = state + input/2 - state²*input/2`:
   
   * A **rational polynomial expression** fitted to 600 training samples.
   * IID test: 300 independent examples.
   * Extrapolation: 300 additional examples outside training ranges.
   * Comparison: unconstrained least-squares affine transition trained on
     exactly the same samples.
   * Recursive rollouts to 1024 steps: correctness is tested against the
     generator, including Python/C++ agreement.

2. A symbolic parity state machine generated from
   `state_next = state XOR input` (binary state and input):
   
   * Discovers `state + input - 2*state*input` from 200 training samples.
   * Tests 32 unseen sequences up to length 4096 with no retraining.

The data-generating functions are used **only to create samples and evaluate**
results; the discovery algorithm is not given the true equations.

## Intellectual honesty / limitations

- This is **grammar-constrained polynomial discovery**, not open-ended
  mathematical invention, general language learning, or a ready replacement
  for attention. Its `numpy.linalg.lstsq` solver uses matrix operations
  during **training**, while compiled inference is a direct mathematical
  program. The coefficients are still learned parameters, though not a
  conventional neural weight tensor.
- The benchmark's true rules are within the preselected search grammar;
  exact recovery is therefore expected with enough noiseless observations.
- Demonstrating extrapolation on such tasks is only a **necessary prototype
  capability**, not evidence of superior general intelligence or Transformer
  replacement. It has not been evaluated on real language, perception,
  reasoning, or world modeling.
- CPU timings for compiled native loops are **not** comparable to GPU
  Transformer throughput, since work per step is dramatically different.
- Unconstrained high-degree search can overfit. A future synthesis engine
  needs discrete program operators, control flow, memory, and stronger
  compression/generalization regularization.

## Concrete research gates

Before claiming competitive architecture quality:

1. Discover multi-state, branching and compositional program structures from
   hidden rules; test length and distribution shifts, without selecting
   grammar or constants from the held-out set.
2. Learn from noisy data and support gradients/continuous optimization of
   embedded constants while the program graph remains compact.
3. Implement vector state and hierarchical recurrence with bounded complexity
   and optional dynamically selected expression branches.
4. Compare at matched data, compute budget and hardware against recurrent
   networks, state-space models and Transformers; include accuracy, memory,
   energy, training cost, and learning curves.
5. Attempt real language modeling only after success on mechanistic tasks.

Program size, validation performance, native compilation and generalization
must all be separately checked. Transformer superiority is an *ambition*, not
an experimental conclusion.

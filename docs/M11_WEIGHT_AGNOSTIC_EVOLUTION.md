# SPARK M11 — topology evolution with shared-gain robustness

This prototype is a first **WANN-inspired structural search**, adapted to
mathematical state-update programs rather than neuron connectivity. It is not
an implementation of the original Weight Agnostic Neural Networks algorithm,
and does **not** demonstrate Transformer-level intelligence.

## Learned executable model

A program updates four registers with a single **shared scalar** `gain`:

    next_state = P(state, input, gain)

Training sees examples from three gain regimes (`0.5, 1.0, 1.5`). It never sees
holdout gain regimes (`-0.75, 0.25, 2.25`). The synthesized program's shape is
shared across all gains: it cannot store separate weights for each regime.
Runtime has rational constants embedded in arithmetic instructions, but no
learned parameter matrices or nearest-neighbor memories. The scalar gain is
supplied externally at execution; this is **gain-conditioned** generalization,
not full weight independence or unsupervised parameter discovery.

## Discovery algorithm

1. Begin with constant and primitive input/state/gain nodes.
2. Apply structural *multiply-by-variable* mutations for up to three
   generations. At generation three, prune the population based on correlation
   with training residuals after lower-order expressions are fitted. This is a
   restricted form of fitness-guided structural evolution, not unrestricted
   genetic programming.
3. Fit a sparse rational head per output register using training-only greedy
   selection. Search for an optional equality-event reset branch.
4. Use stratified train-only internal validation and a description-length
   penalty to select depth, conditional structure, and term count. Refit the
   selected program on training examples only.
5. Validate the typed program then generate C++17 arithmetic, shared
   intermediate terms, branches, and native recurrent rollout code.

The program's topology is shared across output registers; shared monomials are
computed once in C++. All operations are bounded by a whitelist and
serialized using a fixed-version JSON format.

## Generalization tests

Two hidden, synthetic teacher-free rule families are used to generate training
transitions. The synthesizer sees tuples `(state, x, gain, next_state)` and
**never receives the rule source**. Both tasks have 1,050 independent training
transitions, 390 wider-range held-out transitions with **unseen gains**, and 32
native sequence tests of length 32, 128, 1,024, or 4,096.

One task updates four interacting registers with a shared nonlinear term; the
other also has an unknown reset event. The evaluation uses:

* An independently trained full-vector affine regression baseline that receives
  the same state, input and gain.
* A shallower search-depth negative control (tested in regression tests).
* Native-vs-Python parity with exact holdout and long stable-sequence checks.
* Source-level validation and compilation tests for malformed programs.

Each reported MSE averages *all four register outputs*. The long-sequence tests
use stable inputs and do not establish extrapolation to chaotic trajectories.

## Run

    python -m pip install numpy
    OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_evolved_program.py -v
    OPENBLAS_NUM_THREADS=1 python tools/evolution_experiment.py --output artifacts/symbolic-m11

Native compilation requires a C++17 compiler, not MLIR; an LLVM/MLIR dialect
integration is a separate milestone.

## Major limitations

* Fully observed synthetic deterministic transitions, explicitly inside a
  constrained polynomial/equality-reset grammar. The problem is far easier
  than natural language and does not establish compositional reasoning.
* Structural evolution uses deterministic enumerated graph mutations with
  fitness-guided pruning, not WANN's stochastic network-topology evolution.
* Offline fitting uses NumPy's least-squares solver. Native inference uses
  only compiled scalar arithmetic and control flow.
* Single gain conditioning isn't evidence of learned weight-free intelligence.
* Affine regression is not a Transformer, SSM or comparable neural baseline.

**Next research gate:** partially observed states and *learned* reusable
subprograms, adversarially held-out task compositions, noise robustness,
and matched-capacity/compute comparisons against RNNs and small Transformers.

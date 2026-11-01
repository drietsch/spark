# M13 — A genuinely live neural mathematical function in SPARK

**Correction to M12:** M12 generated C++ literals for trained neural parameters,
which froze its numerical behavior after compilation. M13 deliberately avoids
that. It compiles **the mathematical operator and its online gradient rule**,
while coefficients and active hidden-unit count remain mutable at runtime.

## Mathematical model

The neural function is a stateful primitive in an otherwise typed SPARK
mathematical program. For a three-element argument vector `z`:

```
h_j = tanh(bh_j + sum_i wh[j,i] * z_i)
g_j = sigmoid(bg_j + sum_i wg[j,i] * z_i)
q_t = bo + sum_(j < active_t) wo_j * h_j * g_j
s_(t+1) = P(s_t, x_t, context_t, q_t)
```

The primitive has mutable `wh`, `bh`, `wg`, `bg`, `wo`, and `bo`. Its context
changes gates immediately; feedback updates the parameters after each observed
transition. The number of active neurons is also adjustable within the
preallocated capacity **without any recompilation**.

The symbolic update expressions are compiled into C++17, including an exact
symbolic derivative with respect to `q`. This permits end-to-end one-step
online updates from a *next-state observation*, **without needing the target
value of `q`**:

```
prediction = P(s, x, context, q)
grad_q = sum_i((prediction_i-target_i) * dP_i/dq) / sum_i((dP_i/dq)^2)
θ_(t+1) = θ_t - η * clipped_gradient(grad_q, θ_t)
```

The generated C++ has a mutable per-instance state struct and C ABI functions
`create`, `destroy`, `set/get_parameters`, `resize`, `step`, `observe`, and
batched `native_stream`. Its generated source **never contains learned weight
literals**. Native runtime has no Python, PyTorch, or BLAS dependencies.

### Scope and meaning of “dynamic”

* **Yes:** contextual gating; online updates to all dense weights and biases;
  runtime adjustment of active width; checkpoint/reload; multiple independent
  instances with different numerical functions; immediate behavioral change.
* **Not yet:** unlimited runtime growth or arbitrary topology mutation; automatic
  decision of when to allocate a new neuron; unsupervised adaptation without
  feedback; discovery of the surrounding symbolic equations or input signature;
  guaranteed hard real-time deadlines.
* The operator implementation is compiled; it is **not a fixed function** in
  the sense of frozen numerical parameters. As with any CPU instruction, its
  underlying arithmetic is necessarily defined. Program architecture can still
  change through the separate SPARK synthesis and compilation pipeline.

## Reproduce

```bash
python -m unittest discover -s tests -p 'test_plastic_function.py' -v
python tools/plastic_experiment.py --output artifacts/m13
```

Only a C++17 compiler is required. No NumPy or other ML package is needed.

The benchmark uses 1,100 online observations from environment A, then flips the
sign of a nonlinear target relation and provides 850 fresh feedback
transitions. It evaluates 220 fixed, unseen probe inputs without using those
probes for training or tuning. The frozen baseline is a snapshot of the exact
same model taken before drift. Both Python and native C++ update every
coefficient and are checked against each other afterward.

The GitHub Actions report records actual MSE values and native throughput.
Only synthetic fully observed transitions are tested; teacher/ground-truth
neural subfunction values are **not** supplied to learning.

## Limitations and research roadmap

This is an **online-learning numerical operator**, not a full self-modifying
symbolic architecture or general-purpose language model. The current
experiment externally increases active width on drift; automatic structural
allocation, feedback-free state discovery, noisy observations, delayed rewards,
and benchmarks against online RNNs/SSMs remain open.

In particular, **throughput is not a deterministic real-time latency bound**.
No result demonstrates superiority to Transformers on language or reasoning.

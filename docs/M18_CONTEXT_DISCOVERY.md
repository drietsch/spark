# SPARK M18 — Cue-free context-dependent memory routing

## Scientific question

Can SPARK discover **which observations should be stored** by choosing an
executable mathematical program from downstream, delayed query feedback,
**without receiving an explicit write/relevance indicator**?

M17's `Event.signal` was a binary relevance cue. M18 removes that field from
its training inputs and deployment ABI. Every observation has real-valued
context coordinates `(a,b)`, an explicit memory key, and a numeric value.
The eventual read query supplies a key and feedback target. No per-event write
label is given to the learner.

The environment is simulated with a deliberately simple, hidden nonlinear
rule based on the *combination* of the two contexts. The policy search is
restricted to a small, predefined typed grammar:

- `a`, `b`, `a+b`, `a-b`, `a*b`, `a*a-b*b`;
- comparisons `<` or `>` and thresholds `-0.25`, `0`, `0.25`.

SPARK evaluates 36 possible compiled mathematical policies against the actual
**delayed read losses**, adds a description-length penalty, and selects the
best using the training episodes only. The selection is a restricted symbolic
program-search procedure—not open-ended program invention. The target rule is
inside the grammar by construction.

## Memory and training

For each nonquery observation `(key,value,a,b)`, the learned mathematical gate
`P(a,b)` determines whether to replace `memory[key]` with `value`. At each read
`(key,target)`, the current value is predicted *before* the target is consulted.
Query losses from many episodes select the gate; stored values are not trained
as separate lookup labels.

Training uses four keys, short distractor intervals, and continuous contexts.
The held-out test uses eight keys, different random seeds, larger contexts and
1,024–4,096 distractors per read. No training or selection occurs on that
held-out test.

A second experiment evaluates **online structural adaptation**. The previously
learned equation is evaluated before feedback on each new episode. Every six
episodes, a 24-episode rolling window of *past* query targets selects a new
program. The hidden environment changes its context rule and later returns to
the earlier rule; no explicit mode-change or relevance labels reach the model.

## Compiled model

The native compiler emits a C++17 function computing the *discovered* context
expression directly, maintaining a fixed-capacity array of key-addressable
numeric memory. The expression is validated before code generation; supported
operations are whitelisted, with no arbitrary code execution or neural matrix
product. Runtime complexity is O(1) per event for bounded key capacity. A
separate native parity run replays the complete held-out long stream and checks
all memory operations against Python.

## Run

```sh
python -m unittest discover -s tests -p 'test_context_memory.py' -v
python tools/context_experiment.py --output artifacts/m18
```

Requires Python 3.11+ and a C++17 compiler. No neural framework is needed.
`report.json` records the results, program, all controls and limits. The same
program is serialized in `learned_context_program.json`, and its generated
C++ and compiled shared library are included as experiment outputs.

## Controls

The same held-out episodes are evaluated with:

1. The complete discovered context equation.
2. A restricted linear-only feature grammar with otherwise identical fitting.
3. An always-write policy.
4. A never-write policy.

The adaptation benchmark is prequential, so a mode switch should hurt initial
predictions and later recover if the policy can be relearned.

## Important limitations

- Synthetic context relevance is *deliberately representable* by the search
  grammar. This does not establish broad semantic relevance learning.
- Input keys and read requests are still explicit. Unknown key discovery and
  autonomously generated read/write instructions are not implemented.
- The model selects among 36 policies; it does not yet evolve nested neural
  functions or jointly learn partially observed recurrent state.
- Delayed query targets provide supervised feedback, but no direct write labels.
- Program reselection has real compute overhead and is run outside the
  inference-critical path. No strict real-time latency guarantee is claimed.
- Native parity proves the compiled policy matches the interpreter, not
  advantage over a GRU, a Transformer or a selective state-space model.

The next milestone should evolve a contextual gate using the **live adaptive
neural mathematical primitive** where pure symbolic candidate grammars fail,
then use validated neural-to-symbolic consolidation where appropriate. Compare
with strongly tuned recurrent and state-space controls at matched compute.

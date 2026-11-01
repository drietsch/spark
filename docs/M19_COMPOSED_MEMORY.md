# SPARK M19 — Compose learned mathematical memory rules

M18 selected a single predicate from 36 predefined expressions. M19 tests a
strictly larger, transparent grammar: **a single predicate or an AND/OR of two
predicates**. Training receives raw two-dimensional context `(a,b)`, key/value
observations, and future read targets. It receives **no write instruction labels,
relevance bit, or direct description of the generator's rule**.

M19 exhaustively searches 36 atomic predicates plus 1,260 unordered pairs with
AND and OR, selecting the expression minimizing delayed query MSE with a small
program-length penalty. It serializes a validated AST and compiles the discovered
Boolean mathematical program to C++17, operating the same fixed-capacity memory
bank. The synthetic target uses the conjunction of two separate comparisons:

    (a*b > 0) and (a+b > 0.25)

The learner is not given this conjunction as one primitive, though its two
constituent predicates and AND operation exist in the allowed grammar.

## Reproduce

    python -m unittest discover -s tests -p 'test_composed_memory.py' -v
    python tools/composition_experiment.py --output artifacts/m19

The runner uses 160 train episodes of 4 memory keys and short distractions;
42 held-out episodes have 8 keys and 1,024–4,096 distractions per interval,
independent random seeds, and a wider/mixed context distribution. It reports
query correctness, the best atomic-only policy, and native/Python parity.

## Caveats

This is an intentionally synthetic grammar-contained problem. Finding the
correct expression demonstrates **structural rule composition from delayed
supervision**, not general symbolic invention or a learned natural-language
relevance concept. Memory keys/read operations remain supplied; there is no
variable-size learned address mechanism. No Transformer/GRU/SSM matched
comparison is made. Extrapolation applies to **delay and number of keys**,
not to unknown mathematics outside the grammar. Report runtime cost of
candidate search separately from the tiny final native program.

Next: program search with variable depth, hold-out rule families beyond the
candidate grammar, noisy observations, and learning whether to write/query
rather than supplying read operations explicitly.

# SPARK M20 — learned mathematical memory addressing and three-step programs

**Status:** synthetic constrained research demonstration; not a general learner or
compute-matched Transformer benchmark.

M19 learned a 2-condition memory-write policy, but addresses were supplied as
integer `key` fields. M20 introduces a second learned **mathematical function**:

```
address(u,v) -> bounded memory slot
keep(a,b)    -> write / retain
```

Both are selected from **delayed read-target errors**. The observation event
contains continuous features `(u,v,a,b,value)` and a query flag, but **no key,
write label, or relevance bit**. A query provides an observable target only after
an opportunity to predict it. The generator uses private identities to generate
targets; those identities are not passed in the event API.

## Search and representation

A calibration stream has all updates relevant. The learner evaluates eight
small linear address projections, round-to-integer slot assignment, and selects
the projection that minimizes delayed query error. Address calibration supplies
no integer address labels, but it **does tell the learner that all calibration
writes count**. This is not autonomous key or memory-allocation discovery.

The learned address is:

```
address(u,v) = round(u - v)
```

Training identities are 0–3, held-out identities 0–7, and nuisance offsets in
`u` and `v` change independently between writes and reads. Thus a raw-feature
or memorized address table cannot extrapolate to held-out identities reliably.

Write-program search then evaluates the M18 36-atom predicate library and
nested `and` / `or` combinations up to three predicates, via a bounded
structure beam. It trains exclusively on memory read targets, not write labels.
The recovered program is algebraically equivalent to:

```
keep(a,b) = (a+b > 1/4) AND (a*b > 0) AND (a-b < 1/4)
```

This is **three-level composition inside a supplied grammar**, not unbounded
program invention. The compiler generates reusable arithmetic subroutines for
both `address` and `keep`; the same learned address program is executed for
both reads and writes, and compiled C++ handles memory updates. No neural
weight matrices or Python runtime are used for native inference.

## Reproduce

```
python -m unittest discover -s tests -p 'test_addressed_program.py' -v
python tools/addressed_experiment.py --output artifacts/m20
```

The benchmark fits address projection on 55 calibration episodes, then fits
write structure on 110 separate short episodes (four identities and four reads
per episode). Evaluates on 48 independent episodes, with eight identities,
1,024–4,096 intervening distractors per read, and wider context values.
Includes a depth-two program search, raw-feature addressing, and always-write
controls. The generated `.cpp` and `.so` are tested against the Python runtime
on every held-out operation.

## Limitations / next experimental priorities

- The hidden target lies inside the explicitly finite primitive grammar.
- Calibration already provides an all-write phase; write relevance and
  addressing are learned in **separate**, not joint, stages.
- Queries are explicit, capacity is allocated, and there is no learnt
  allocation, eviction, or associative retrieval from arbitrary observations.
- No natural-language, Transformer, GRU, or state-space compute-matched
  benchmark, and no proof of general intelligence.
- M21 should learn addressing and write operations *jointly*, remove the
  calibration protocol, and test latent/novel identities under noise and
  changing memory capacity. The live neural mathematical function should be
  evaluated jointly rather than remaining an independent module.

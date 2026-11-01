# SPARK M23 — Simple logical question answering

## Milestone

M23 supplies the first **usable logical question answering interface** for SPARK.
It parses a *controlled subset* of English into a typed Horn-clause program,
executes bounded proof search, and replies **yes / no / unknown / inconsistent**
with an actual chain of premises. The same finite logical rules also compile
into a C++17 inference engine. This is a new first-class symbolic computational
primitive designed to coexist with SPARK's dynamic neural functions; the M23
benchmark does **not** evaluate any neural language understanding.

## Try it

No third-party dependencies are needed; C++17 is required for native parity.

```sh
python tools/logic_qa.py --demo
python tools/logic_qa.py --demo --json
python tools/logic_qa.py --interactive
python tools/logic_qa.py --file examples/logic_story.txt --question 'Is Milo an animal?'
python -m unittest discover -s tests -p 'test_logic_program.py' -v
python tools/logic_experiment.py --output artifacts/m23
```

Example program:

```
All cats are mammals.
All mammals are animals.
No cats are reptiles.
Milo is a cat.
If X is a parent of Y then X is an ancestor of Y.
If X is a parent of Y and Y is an ancestor of Z then X is an ancestor of Z.
Alice is a parent of Bob.
Bob is a parent of Cara.
```

Questions:

```
Is Milo an animal?             YES (two inference steps)
Is Milo a reptile?             NO (explicit negative rule)
Is Alice an ancestor of Cara?  YES (composed rules)
Is Milo a bird?                UNKNOWN (open world)
```

Unlike a closed-world database, **unknown is not false**. Adding contradictory
facts results in `inconsistent` for the disputed claim without allowing arbitrary
other conclusions. Rules use positive antecedents only, but their heads may
produce explicit negative conclusions.

## Syntax and learning

* Facts: `Alice is a cat`, `Alice is happy`, `Alice is not happy`, `Alice likes Bob`,
  `Alice is taller than Bob`, `Alice is a parent of Bob`.
* Taxonomy: `All cats are mammals`, `Every mammal is an animal`.
* Exclusions: `No cats are reptiles`.
* Bounded implications: `If X is a parent of Y and Y is a parent of Z then X is a grandparent of Z`.
* Questions: `Is Alice a mammal?`, `Is Alice not happy?`, `Does Alice like Bob?`.
* Extra **restricted structural learning**: `program.learn_transitivity('is taller than',
  [('Is Alice taller than Cara?',True), ...])` selects between existing rules
  and an additional general transitivity rule using training labels. The learned
  rule generalizes to *new* entities but is **not** open-ended logic induction.

No arbitrary Python evaluation, shell commands, or executable expressions are
accepted from user text. Facts and clauses have strict arity, finite resources,
and safe serialization.

## Technical architecture

`LogicProgram` parses controlled input into `Atom(pred,args,negative)` and
`Rule(head,body,source)`. A grounded, forward-chaining fixed point yields a
consequence relation with an indexed proof tree. `NativeLogic.compile` emits
C++17 source containing **asserted facts and rule structures only**, plus an
interpreter that performs the inference natively. The compiled library answers
integer-symbol queries; Python holds the boundary vocabulary. These vocabulary
IDs are for logic symbol identity, **not** a memorized answer table.

The M23 runtime is rule-based, and no live dynamic dense network is required for
exact propositional inference. Later SPARK work can use its neural function to
propose interpretations or rules and use symbolic inference as a verifier, but
this integration is not claimed or implemented in M23.

## Evaluation

600 synthetic questions across independent contexts of taxonomic chains,
negative facts, ancestry inference, and contradictory facts. M23 answers
600/600 in the controlled grammar, compared with 330/600 for a direct-facts-only
baseline. On a separate compiled multi-relation world, C++ and Python agree
for 100/100 questions. All questions are generated from tasks supported by
M23's parsing and reasoning rules.

**Limitations:** No unrestricted natural language, commonsense/world knowledge,
complex quantification, arbitrary disjunction, external retrieval, probability,
long-term conversational state or quality parity against a language model.
The rules are largely *provided*; the optional rule learner searches only one
hypothesis family. The high synthetic accuracy is **not** a valid generalization
metric for broader human logical language.

Next: online semantic parsing with an evolving tiny neural function, learning
new relational rules from longer examples, and uncertain abductive reasoning
without weakening proof correctness.

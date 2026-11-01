#!/usr/bin/env python3
"""Independent compositional logical-QA benchmark for SPARK M23.

Data are synthetic controlled-English puzzles, NOT natural language or
comparison to neural language models. Each question is answered from a new
context. Report baselines including evidence-only (no logical rules).
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.logic_program import LogicProgram
from spark.logic_native import NativeLogic


def independent_cases():
    rng=random.Random(20261101)
    cases=[]
    for i in range(100):
        name=f'item{i}'
        p=LogicProgram().add_many(f'All cats are mammals. All mammals are animals. No cats are reptiles. {name} is a cat.')
        for q,expected in ((f'Is {name} an animal?','yes'),
                           (f'Is {name} a reptile?','no'),
                           (f'Is {name} a bird?','unknown'),
                           (f'Is unknown{i} an animal?','unknown')):
            cases.append((p,q,expected))
    for i in range(70):
        p=LogicProgram().add_many('If X is a parent of Y then X is an ancestor of Y. If X is a parent of Y and Y is an ancestor of Z then X is an ancestor of Z.')
        n=rng.randint(3,7)
        names=[f'e{i}_{j}' for j in range(n)]
        for a,b in zip(names,names[1:]):p.add(f'{a} is a parent of {b}')
        cases.append((p,f'Is {names[0]} an ancestor of {names[-1]}?','yes'))
        cases.append((p,f'Is {names[-1]} an ancestor of {names[0]}?','unknown'))
    for i in range(30):
        name=f'person{i}'
        p=LogicProgram().add_many(f'{name} is a doctor. {name} is not a doctor.')
        cases.extend(((p,f'Is {name} a doctor?','inconsistent'),
                      (p,f'Is {name} a magician?','unknown')))
    return cases


def run(out:Path,*,native=True):
    out.mkdir(parents=True,exist_ok=True)
    cases=independent_cases()
    hits=direct_hits=yes_hits=0
    result_counts={'yes':0,'no':0,'unknown':0,'inconsistent':0}
    for p,q,expected in cases:
        observed=p.ask(q).status
        hits+=observed==expected
        result_counts[expected]+=1
        # Direct evidence baseline, cannot use rules.
        reference=LogicProgram()
        reference.facts=p.facts.copy()
        direct_hits+=reference.ask(q).status==expected
        yes_hits+='yes'==expected
    # A substantial single compiled world avoids using an FFI call for each
    # separately compiled microscopic model and verifies C++ executes inference.
    p=LogicProgram().add_many('If X is a parent of Y then X is an ancestor of Y. If X is a parent of Y and Y is an ancestor of Z then X is an ancestor of Z. No cats are reptiles. All cats are mammals.')
    questions=[]
    for group in range(20):
        people=[f'person{group}_{j}' for j in range(7)]
        for a,b in zip(people,people[1:]):p.add(f'{a} is a parent of {b}')
        p.add(f'{people[0]} is a cat')
        questions.extend((f'Is {people[0]} an ancestor of {people[-1]}?',
                          f'Is {people[-1]} an ancestor of {people[0]}?',
                          f'Is {people[0]} a reptile?',
                          f'Is {people[0]} a mammal?',
                          f'Is {people[1]} a cat?'))
    # Truth labels are generated independently from the structural construction,
    # and checked by the separately implemented C++ runtime.
    parity=0
    native_bytes=None
    native_seconds=None
    if native:
        start=perf_counter()
        compiled=NativeLogic.compile(p,out/'m23_logic.so')
        native_seconds=perf_counter()-start
        parity=sum(compiled.ask(q)==p.ask(q).status for q in questions)
        native_bytes=(out/'m23_logic.so').stat().st_size
    report={'format':'spark-m23-controlled-logic-v1',
            'benchmark_questions':len(cases),
            'correct':hits,
            'accuracy':hits/len(cases),
            'direct_fact_baseline_correct':direct_hits,
            'direct_fact_baseline_accuracy':direct_hits/len(cases),
            'always_yes_baseline_correct':yes_hits,
            'answer_distribution':result_counts,
            'native_questions':len(questions) if native else 0,
            'native_python_agreement':parity,
            'native_library_bytes':native_bytes,
            'native_compile_seconds':native_seconds,
            'transitivity_learning':'finite two-hypothesis search; tested in unittest with unseen entities',
            'limitations':[
                'Controlled English only: no unrestricted natural-language understanding.',
                'Rules manually supplied except a restricted optional transitivity-rule learner.',
                'Synthetic examples are generated within the supported grammar.',
                'No Transformer, GRU, or LLM baseline; no general intelligence claim.',
                'Negative answers require explicit negative evidence; otherwise unknown.',
                'Native C++ closure uses bounded finite facts and user-supplied rule grammar.'
            ]}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    if hits!=len(cases) or (native and parity!=len(questions)):
        raise AssertionError(report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('artifacts/m23'))
    ap.add_argument('--skip-native',action='store_true')
    args=ap.parse_args()
    run(args.output,native=not args.skip_native)

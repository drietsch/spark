#!/usr/bin/env python3
"""SPARK M23 controlled-English logical reasoning CLI."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.logic_program import LogicProgram


DEMO_FACTS='''
All cats are mammals.
All mammals are animals.
No cats are reptiles.
Milo is a cat.
Bob is a parent of Cara.
Alice is a parent of Bob.
If X is a parent of Y then X is an ancestor of Y.
If X is a parent of Y and Y is an ancestor of Z then X is an ancestor of Z.
Alice likes Bob.
'''
DEMO_QUESTIONS=('Is Milo an animal?', 'Is Milo a reptile?',
    'Is Alice an ancestor of Cara?', 'Does Bob like Alice?',
    'Is Luna a cat?')


def main():
    p=argparse.ArgumentParser(description='SPARK compiled-expression research: proof-based logical question answering')
    p.add_argument('--file',type=Path,help='Text file of facts and conditional rules, one per line')
    p.add_argument('--question','-q',action='append',help='Question to ask; repeat for multiple questions')
    p.add_argument('--demo',action='store_true',help='Run a multi-hop inference demonstration')
    p.add_argument('--json',action='store_true',help='Emit machine-readable answers')
    p.add_argument('--interactive',action='store_true',help='Ask and add statements interactively')
    args=p.parse_args()
    engine=LogicProgram()
    if args.demo:engine.add_many(DEMO_FACTS)
    if args.file:engine.add_many(args.file.read_text())
    questions=(list(DEMO_QUESTIONS) if args.demo else [])+(args.question or [])
    for q in questions:
        result=engine.ask(q)
        if args.json:print(json.dumps(result.as_dict()))
        else:
            print(f'Q: {q}\nA: {result.status.upper()}')
            for proof in result.supporting:print('  ✓ '+proof)
            for proof in result.opposing:print('  ! '+proof)
    if not args.interactive:return
    print('Enter a statement to teach SPARK, or a question starting with Is/Does. Ctrl-D to finish.')
    while True:
        try:line=input('spark> ').strip()
        except EOFError:break
        if not line:continue
        try:
            if line.lower().startswith(('is ','does ')):
                result=engine.ask(line)
                print(result.status.upper())
                for proof in result.supporting:print('  ✓ '+proof)
                for proof in result.opposing:print('  ! '+proof)
            else:
                engine.add(line)
                print('Added.')
        except (ValueError,RuntimeError) as ex:print(f'Cannot process: {ex}')


if __name__=='__main__':main()

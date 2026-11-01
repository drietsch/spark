"""M23 logic capability, compositional generalization, safety, proof and CLI."""
from __future__ import annotations
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spark.logic_program import LogicProgram,Rule,Atom,parse_atom,parse_question

ROOT=Path(__file__).resolve().parents[1]

class LogicTests(unittest.TestCase):
    def test_taxonomy_chain_open_world_and_explicit_negative(self):
        p=LogicProgram().add_many('All cats are mammals. All mammals are animals. No cats are reptiles. Milo is a cat.')
        yes=p.ask('Is Milo an animal?')
        self.assertEqual(yes.status,'yes');self.assertTrue(any('All mammals' in s for s in yes.supporting))
        self.assertEqual(p.ask('Is Milo a reptile?').status,'no')
        self.assertEqual(p.ask('Is Milo not a reptile?').status,'yes')
        self.assertEqual(p.ask('Is Milo a dog?').status,'unknown')
        self.assertEqual(p.ask('Is Luna an animal?').status,'unknown')

    def test_two_step_relation_and_variable_unification(self):
        p=LogicProgram().add_many('''
            If X is a parent of Y then X is an ancestor of Y.
            If X is a parent of Y and Y is an ancestor of Z then X is an ancestor of Z.
            Alice is a parent of Bob.
            Bob is a parent of Cara.
            Cara is a parent of Dave.
            ''')
        self.assertEqual(p.ask('Is Alice an ancestor of Dave?').status,'yes')
        self.assertEqual(p.ask('Is Dave an ancestor of Alice?').status,'unknown')
        self.assertGreater(len(p.ask('Is Alice an ancestor of Dave?').supporting),3)

    def test_three_premise_rule_and_no_invented_consequence(self):
        p=LogicProgram().add_many('''
            If X is a cat and X is happy and X is hungry then X is playful.
            Ivy is a cat. Ivy is happy. Ivy is hungry.
            No dogs are playful. Ben is a dog.
        ''')
        self.assertEqual(p.ask('Is Ivy playful?').status,'yes')
        self.assertEqual(p.ask('Is Ben playful?').status,'no')
        self.assertEqual(p.ask('Is Ben happy?').status,'unknown')

    def test_contradictions_do_not_collapse_or_explode(self):
        p=LogicProgram().add_many('Milo is a cat. Milo is not a cat.')
        self.assertEqual(p.ask('Is Milo a cat?').status,'inconsistent')
        self.assertEqual(p.ask('Is Milo a giraffe?').status,'unknown')
        self.assertEqual(p.ask('Is Milo not a cat?').status,'inconsistent')

    def test_binary_predicate_and_transitivity_training(self):
        p=LogicProgram().add_many('''Alice is taller than Bob.
            Bob is taller than Cara. Cara is taller than Dave.
            Alice likes Bob.''')
        self.assertEqual(p.ask('Is Alice taller than Cara?').status,'unknown')
        labels=[('Is Alice taller than Cara?',True),
                ('Is Bob taller than Dave?',True),
                ('Is Dave taller than Alice?',False),
                ('Is Alice taller than Dave?',True)]
        fit=p.learn_transitivity('is taller than',labels)
        self.assertTrue(fit['selected'],fit)
        self.assertEqual(fit['candidate_errors'],0)
        self.assertEqual(p.ask('Is Alice taller than Dave?').status,'yes')
        self.assertEqual(p.ask('Does Alice like Bob?').status,'yes')
        self.assertEqual(p.ask('Does Bob like Alice?').status,'unknown')
        # New entities and many-hop patterns are not in induction labels.
        p.add_many('Eve is taller than Finn. Finn is taller than Gia. Gia is taller than Hank.')
        self.assertEqual(p.ask('Is Eve taller than Hank?').status,'yes')

    def test_learned_rules_survive_saved_reload(self):
        p=LogicProgram().add_many('Alice is taller than Bob. Bob is taller than Cara.')
        labels=[('Is Alice taller than Cara?',True),('Is Cara taller than Alice?',False),
                ('Is Bob taller than Alice?',False)]
        self.assertTrue(p.learn_transitivity('is taller than',labels)['selected'])
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'learnt.json';p.save(f)
            loaded=LogicProgram.load(f)
            self.assertEqual(loaded.ask('Is Alice taller than Cara?').status,'yes')
            loaded.add_many('Dave is taller than Eve. Eve is taller than Finn.')
            self.assertEqual(loaded.ask('Is Dave taller than Finn?').status,'yes')

    def test_unsafe_rule_is_rejected(self):
        for bad in (
            'If X is happy then Y is happy',
            'If X is happy and X is sad and X is quiet and X is tired then X is human',
            'If X is not happy then X is sad',
            'Milo is happy and Alice is sad',
            'If X is happy then X is a cat and Y is a dog',
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):LogicProgram().add(bad)
        with self.assertRaises(ValueError):LogicProgram().ask('Is X happy?')

    def test_nontransitive_relation_not_induced_from_negative_examples(self):
        p=LogicProgram().add_many('Alice likes Bob. Bob likes Cara. Cara likes Dave.')
        labels=[('Does Alice like Cara?',False),
                ('Does Bob like Dave?',False),
                ('Does Cara like Alice?',False)]
        result=p.learn_transitivity('likes',labels)
        self.assertFalse(result['selected'])
        self.assertEqual(p.ask('Does Alice like Cara?').status,'unknown')

    def test_roundtrip_and_corrupted_model(self):
        p=LogicProgram().add_many('No cats are reptiles. Milo is a cat.')
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'model.json';p.save(f)
            self.assertEqual(LogicProgram.load(f).ask('Is Milo a reptile?').status,'no')
            r=json.loads(f.read_text());r['rules']=['If X is happy then Y is happy']
            f.write_text(json.dumps(r))
            with self.assertRaises(ValueError):LogicProgram.load(f)
            r['format']='not-spark';f.write_text(json.dumps(r))
            with self.assertRaises(ValueError):LogicProgram.load(f)

    def test_randomized_unseen_entity_and_hop_generalization(self):
        rng=random.Random(203)
        for trial in range(20):
            p=LogicProgram().add_many('If X is a parent of Y then X is an ancestor of Y. If X is a parent of Y and Y is an ancestor of Z then X is an ancestor of Z.')
            size=rng.randrange(4,10)
            names=[f'node{trial}_{i}' for i in range(size)]
            for a,b in zip(names,names[1:]):p.add(f'{a} is a parent of {b}')
            self.assertEqual(p.ask(f'Is {names[0]} an ancestor of {names[-1]}?').status,'yes')
            self.assertEqual(p.ask(f'Is {names[-1]} an ancestor of {names[0]}?').status,'unknown')

    def test_compiled_native_logic_parity(self):
        import shutil
        if not (shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')):
            self.skipTest('C++17 compiler absent')
        from spark.logic_native import NativeLogic
        p=LogicProgram().add_many("""All cats are mammals. All mammals are animals.
            No cats are reptiles. Milo is a cat. Milo is not a cat.
            Alice is a parent of Bob. Bob is a parent of Cara.
            If X is a parent of Y then X is an ancestor of Y.
            If X is a parent of Y and Y is an ancestor of Z then X is an ancestor of Z.""")
        queries=('Is Milo an animal?', 'Is Milo a reptile?', 'Is Milo a cat?',
                 'Is Alice an ancestor of Cara?', 'Is Cara an ancestor of Alice?',
                 'Is Luna a cat?', 'Is Milo not a cat?')
        with tempfile.TemporaryDirectory() as folder:
            native=NativeLogic.compile(p,Path(folder)/'logic.so')
            self.assertEqual([native.ask(q) for q in queries],[p.ask(q).status for q in queries])

    def test_native_deduction_on_learned_rule_and_new_entities(self):
        import shutil
        if not shutil.which('c++'):self.skipTest('compiler missing')
        from spark.logic_native import NativeLogic
        p=LogicProgram().add_many('Alice is taller than Bob. Bob is taller than Cara.')
        labels=[('Is Alice taller than Cara?',True),('Is Cara taller than Alice?',False),
                ('Is Bob taller than Alice?',False)]
        self.assertTrue(p.learn_transitivity('is taller than',labels)['selected'])
        p.add_many('Eve is taller than Finn. Finn is taller than Gia. Gia is taller than Hank.')
        with tempfile.TemporaryDirectory() as d:
            native=NativeLogic.compile(p,Path(d)/'logic.so')
            self.assertEqual(native.ask('Is Eve taller than Hank?'),'yes')
            self.assertEqual(native.ask('Is Hank taller than Eve?'),'unknown')

    def test_cli_inference_and_json(self):
        result=subprocess.run([sys.executable,str(ROOT/'tools/logic_qa.py'),'--demo','--json'],
                              cwd=ROOT,capture_output=True,text=True,check=True)
        lines=[json.loads(s) for s in result.stdout.splitlines()]
        self.assertEqual([s['answer'] for s in lines],['yes','no','yes','unknown','unknown'])

if __name__=='__main__':unittest.main()

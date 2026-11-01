"""M17 regressions: no oracle write labels, long-memory, live updates, native parity."""
import copy
import math
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.learned_memory import MemoryRouter,Event,episodes
from spark.adaptive_reasoner import AdaptiveReasoner
from spark.plastic_function import PlasticFunction


class LearnedMemoryTests(unittest.TestCase):
    def test_train_on_query_targets_not_write_instructions(self):
        model=MemoryRouter([-1.,.2,0.])
        train=episodes(11,180,distractors=(4,12))
        initial=sum(model.evaluate_episode(s)[0] for s in train)
        result=model.fit(train,epochs=15,rate=.17)
        new=sum(model.evaluate_episode(s)[0] for s in train)
        self.assertLess(new,initial*.01)
        self.assertLess(model.score(Event(0,0,.9)),.5)
        self.assertGreater(model.score(Event(1,0,-.9)),.5)
        self.assertLess(result[-1],result[0])

    def test_multiple_keys_delayed_reads_and_reset_between_episodes(self):
        policy=MemoryRouter([-8.,16.,0.])
        streams=episodes(88,8,keys=4,distractors=(2048,4096))
        hits=reads=0
        for seq in streams:
            _,h,n,_=policy.evaluate_episode(seq)
            hits+=h;reads+=n
        self.assertEqual((hits,reads),(24,24))
        bad=MemoryRouter([9.,0.,0.])
        self.assertLess(sum(bad.evaluate_episode(s)[1] for s in streams),reads)

    def test_no_query_label_before_prediction(self):
        policy=MemoryRouter([-8.,16.,0.])
        reasoner=AdaptiveReasoner(policy,PlasticFunction.initialize(capacity=6,active=4,seed=7))
        mem=reasoner.initial_memory()
        reasoner.step(mem,Event(1.,0,.4),.8)
        e=Event(0.,0,0.,True,999999.)
        # event.target is never used by step unless the caller explicitly
        # provides target as feedback AFTER the prediction is calculated.
        before=reasoner.step(mem,e,.8)
        self.assertTrue(math.isfinite(before))
        self.assertLess(abs(before),5.)

    def test_immediate_online_change_and_frozen_control(self):
        r=random.Random(32)
        policy=MemoryRouter([-8.,16.,0.])
        model=AdaptiveReasoner(policy,PlasticFunction.initialize(capacity=10,active=8,seed=27))
        def phase(sign,count,update=True):
            errors=[]
            for _ in range(count):
                val=r.uniform(-1.1,1.1);c=r.choice([-.9,-.3,.4,1.2])
                mem=model.initial_memory()
                model.step(mem,Event(1.,0,val),c)
                target=val+sign*.42*math.tanh(1.5*val+.3*c)
                p=model.step(mem,Event(0.,0,0.,True),c,target=target if update else None,lr=.06)
                errors.append((p-target)**2)
            return errors
        phase(1,650)
        frozen=copy.deepcopy(model.function)
        changed=phase(-1,700)
        baseline=AdaptiveReasoner(policy,frozen)
        test=[]
        for j in range(100):
            val=r.uniform(-1.1,1.1);c=r.uniform(-1,1)
            mem=baseline.initial_memory();baseline.step(mem,Event(1.,0,val),c)
            p=baseline.step(mem,Event(0.,0,0.,True),c)
            target=val-.42*math.tanh(1.5*val+.3*c)
            test.append((p-target)**2)
        self.assertLess(sum(changed[-100:])/100,sum(test)/len(test)*.15)

    @unittest.skipUnless(shutil.which('c++') or shutil.which('g++') or shutil.which('clang++'), 'requires C++ compiler')
    def test_native_learned_router_parity(self):
        sys.path.insert(0,str(ROOT/'tools'))
        from integrated_experiment import native_check
        policy=MemoryRouter([-8.,16.,0.])
        with tempfile.TemporaryDirectory() as t:
            result=native_check(policy,episodes(91,10,distractors=(1024,2048)),Path(t)/'memory.so')
        self.assertEqual(result['disagreements'],0)
        self.assertEqual(result['reads'],30)


if __name__=='__main__':unittest.main()

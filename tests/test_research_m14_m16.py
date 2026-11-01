import math,random,shutil,tempfile,unittest
from pathlib import Path
import numpy as np
from spark.predictive_state import PredictiveState,fit
from spark.consolidation import ConsolidatedFunction,ConsolidationManager,evaluate,synthesize
from spark.algorithmic_memory import AlgorithmicMemory,WRITE,READ,ADD,ERASE,NOOP

class M14Tests(unittest.TestCase):
 def test_predictive_model_training_from_observations_only(self):
  import sys
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
  from research_experiment import system_sequence
  rows=[system_sequence(k,80) for k in range(20)]
  m=fit(rows,max_order=3)
  self.assertEqual(m.order,2)
  s,u=system_sequence(70,150)
  err=[]
  for t in range(1,149):
   err.append((m.predict([s[t],s[t-1]],u[t])-s[t+1])**2)
  self.assertLess(np.mean(err),1e-12)
 def test_prequential_online_updates_and_safety(self):
  m=PredictiveState(1,[1.,0.,0.])
  self.assertEqual(m.online([2.],.5,2.,lr=.1),2.)
  with self.assertRaises(ValueError):m.online([1.],.2,3.,lr=20.)

class M15Tests(unittest.TestCase):
 def test_consolidate_and_rollback(self):
  fn=lambda z:.25*z[0]*z[1]-.125*z[0]
  rng=random.Random(7)
  train=[((rng.uniform(-1,1),rng.uniform(-1,1)),0) for _ in range(500)]
  train=[(z,fn(z)) for z,_ in train]
  val=[((rng.uniform(-1,1),rng.uniform(-1,1)),0) for _ in range(180)]
  val=[(z,fn(z)) for z,_ in val]
  rolls=[]
  for _ in range(6):
   s=rng.uniform(-.3,.3);xs=[rng.uniform(-.7,.7) for _ in range(30)];end=s
   for x in xs:end=.6*end+fn((end,x))
   rolls.append((s,xs,end))
  m=ConsolidatedFunction(fn)
  result=m.propose(train,val,rolls,absolute_mse=1e-12,max_terms=6)
  self.assertTrue(result['accepted'],result)
  bad=[(z,rng.uniform(-3,3)) for z,_ in train]
  result=m.propose(bad,val,rolls,absolute_mse=1e-12,max_terms=6)
  self.assertFalse(result['accepted']);self.assertIsNotNone(m.symbolic)
 def test_evidence_gated_automatic_promotion_and_drift_rollback(self):
  fn=lambda z:.25*z[0]*z[1]-.125*z[0]
  r=random.Random(301)
  manager=ConsolidationManager(fn,min_samples=100,interval=50,absolute_mse=1e-12,drift_mse=.001)
  rolls=[]
  for _ in range(4):
   s=r.uniform(-.1,.1);xs=[r.uniform(-.4,.4) for _ in range(12)];end=s
   for x in xs:end=.6*end+fn((end,x))
   rolls.append((s,xs,end))
  # Targets arrive as real observed data; not provided to the proposal fitter.
  for _ in range(200):
   z=(r.uniform(-1,1),r.uniform(-1,1))
   manager.observe(z,fn(z),rollouts=rolls)
  self.assertIsNotNone(manager.function.symbolic)
  self.assertTrue(any(e["event"]=="accept" for e in manager.decisions))
  for _ in range(24):
   z=(r.uniform(-1,1),r.uniform(-1,1))
   manager.observe(z,fn(z)+3.)
  self.assertIsNone(manager.function.symbolic)
  self.assertTrue(any(e["event"]=="rollback" for e in manager.decisions))
 def test_inadequate_evidence_rejected(self):
  with self.assertRaises(ValueError):ConsolidatedFunction(lambda z:0).propose([],[],[])

class M16Tests(unittest.TestCase):
 def test_lru_and_arithmetic(self):
  m=AlgorithmicMemory(2)
  self.assertEqual(m.step(WRITE,2,4),4)
  self.assertEqual(m.step(ADD,2,3),7)
  self.assertEqual(m.step(WRITE,3,-2),-2)
  self.assertEqual(m.step(READ,2),7)
  m.step(WRITE,4,2)
  self.assertEqual(m.step(READ,3),0)
  self.assertEqual(m.step(ERASE,2),7)
 def test_invalid_operation(self):
  with self.assertRaises(ValueError):AlgorithmicMemory(0)
  with self.assertRaises(ValueError):AlgorithmicMemory().step(20,0)

if __name__=='__main__':unittest.main()

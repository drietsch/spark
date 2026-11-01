"""M16: bounded, program-controlled associative registers (not token recall).

The memory instruction policy is EXPLICITLY provided, not learned. Tests ask
whether a compiled symbolic machine can manage information across very long
intervening sequences. Entries are exact, finite and capacity-limited.
"""
from __future__ import annotations
import math
from dataclasses import dataclass,field

WRITE,READ,ADD,ERASE,NOOP=1,2,3,4,0

@dataclass
class AlgorithmicMemory:
    capacity:int=32
    data:dict=field(default_factory=dict)
    tick:int=0
    def __post_init__(self):
        if type(self.capacity) is not int or not 1<=self.capacity<=1024:
            raise ValueError('invalid capacity')
    def step(self,op,key,value=0.):
        if op not in (WRITE,READ,ADD,ERASE,NOOP) or not isinstance(key,int) or key<0:
            raise ValueError('invalid memory instruction')
        if not math.isfinite(value):raise ValueError('nonfinite memory payload')
        self.tick+=1
        if op in (WRITE,ADD):
            old=self.data[key][0] if key in self.data else 0.
            val=value if op==WRITE else old+value
            if key not in self.data and len(self.data)>=self.capacity:
                evict=min(self.data,key=lambda k:(self.data[k][1],k))
                del self.data[evict]
            self.data[key]=(val,self.tick)
            return val
        if op==READ:
            if key not in self.data:return 0.
            val,_=self.data[key];self.data[key]=(val,self.tick);return val
        if op==ERASE:
            return self.data.pop(key,(0.,0))[0]
        return 0.

    @staticmethod
    def cpp(capacity=32):
        if type(capacity) is not int or not 1<=capacity<=1024:raise ValueError('invalid capacity')
        return f'''// SPARK M16 bounded programmable memory, explicit instruction semantics.
#include <cstddef>
#include <cstdint>
struct SparkMemory {{
 long long keys[{capacity}]{{}}; double values[{capacity}]{{}};
 std::uint64_t age[{capacity}]{{}}; std::uint64_t tick{{0}};
 bool valid[{capacity}]{{}};
}};
extern "C" void spark_memory_init(SparkMemory* m) {{ *m=SparkMemory{{}}; }}
extern "C" double spark_memory_step(SparkMemory* m,int op,long long key,double value) {{
 if(!m||key<0)return 0.;
 ++m->tick;
 int hit=-1,free_slot=-1,evict=-1;
 for(int i=0;i<{capacity};++i) {{
  if(m->valid[i]&&m->keys[i]==key)hit=i;
  if(!m->valid[i]&&free_slot==-1)free_slot=i;
  if(m->valid[i]&&(evict==-1||m->age[i]<m->age[evict]||
       (m->age[i]==m->age[evict]&&m->keys[i]<m->keys[evict])))evict=i;
 }}
 if(op==1||op==3) {{
  int idx=hit==-1?(free_slot==-1?evict:free_slot):hit;
  double old=hit==-1?0.:m->values[hit];
  m->valid[idx]=true;m->keys[idx]=key;m->values[idx]=op==1?value:old+value;
  m->age[idx]=m->tick;return m->values[idx];
 }}
 if(op==2){{if(hit==-1)return 0.;m->age[hit]=m->tick;return m->values[hit];}}
 if(op==4){{if(hit==-1)return 0.;m->valid[hit]=false;return m->values[hit];}}
 return 0.;
}}
'''

"""SPARK M23 native C++17 Horn inference: compiled rules, NOT lookup answers.

The compiler embeds only asserted facts and rules. The C++ runtime performs
fixed-point forward chaining and polarity-aware query evaluation itself.
Python holds vocabulary IDs for the foreign-function boundary.
"""
from __future__ import annotations
import ctypes
import json
import shutil
import subprocess
from pathlib import Path
from spark.logic_program import LogicProgram,Atom,parse_question,variable,MAX_DERIVED,MAX_ROUNDS


def cpp_source(engine:LogicProgram)->tuple[str,dict,dict]:
    if not engine.facts and not engine.rules:raise ValueError('empty model')
    preds=sorted({a.pred for a in engine.facts}|{a.pred for r in engine.rules for a in (*r.body,r.head)})
    symbols=sorted({v for a in engine.facts for v in a.args}|
                   {v for r in engine.rules for a in (*r.body,r.head) for v in a.args if not variable(v)})
    predicates={p:i for i,p in enumerate(preds)}
    entities={v:i for i,v in enumerate(symbols)}
    if len(preds)>1000 or len(symbols)>4000:raise ValueError('native vocabulary too large')
    def encode(a:Atom,variables=None):
        variables={} if variables is None else variables
        def arg(s):
            if variable(s):
                if s not in variables: variables[s]=-(len(variables)+1)
                return variables[s]
            return entities[s]
        args=[arg(x) for x in a.args]
        return f'{{{predicates[a.pred]},{int(a.neg)},{len(a.args)},{args[0]},{args[1] if len(args)==2 else -9999}}}'
    facts=[encode(a) for a in engine.facts]
    rules=[]
    for r in engine.rules:
        variables={}
        body=[encode(a,variables) for a in r.body]
        head=encode(r.head,variables)
        rules.append((head,body))
    fact_txt=',\n'.join(facts) if facts else ''
    rule_src=[]
    for head,body in rules:
        items=','.join(body)
        rule_src.append(f'{{{head},{{{items}}}}}')
    rule_txt=',\n'.join(rule_src)
    source=r'''// SPARK M23. Compiled finite Horn rules; C++ performs actual deduction.
#include <array>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
struct Lit {int p,n,arity,a,b; bool operator==(const Lit& x) const {
 return p==x.p&&n==x.n&&arity==x.arity&&a==x.a&&b==x.b;}};
struct Hash {std::size_t operator()(const Lit& x) const {
 uint64_t h=1469598103934665603ULL;
 for(int v:{x.p,x.n,x.arity,x.a,x.b}){h^=uint32_t(v);h*=1099511628211ULL;}
 return static_cast<std::size_t>(h);}};
struct Rule {Lit head;std::vector<Lit> body;};
static const std::vector<Lit> input = {FACTS_PLACEHOLDER};
static const std::vector<Rule> rules = {RULES_PLACEHOLDER};
static constexpr int max_derived=25000,max_rounds=40;
static bool unify(const Lit& pattern,const Lit& fact,const std::array<int,4>& src,std::array<int,4>& dst) {
 if(pattern.p!=fact.p||pattern.n!=fact.n||pattern.arity!=fact.arity) return false;
 dst=src;
 const int ps[2]={pattern.a,pattern.b},fs[2]={fact.a,fact.b};
 for(int i=0;i<pattern.arity;++i){int v=ps[i],f=fs[i];
  if(v>=0){if(v!=f)return false;}else{
   int index=-v-1;if(index<0||index>=4)return false;
   if(dst[index]>=0&&dst[index]!=f)return false;
   dst[index]=f;
  }
 }return true;
}
static Lit subst(Lit p,const std::array<int,4>& env){
 if(p.a<0) p.a=env[-p.a-1];
 if(p.arity==2 && p.b<0)p.b=env[-p.b-1];
 return p;
}
static void match_rule(const Rule& r,int depth,const std::array<int,4>& env,
 const std::vector<Lit>& known,std::unordered_set<Lit,Hash>& output){
 if(depth==static_cast<int>(r.body.size())){
   auto q=subst(r.head,env);
   if(q.a>=0 && (q.arity==1||q.b>=0)) output.insert(q);
   return;
 }
 for(const auto& fact:known){
  std::array<int,4> next;
  if(unify(r.body[depth],fact,env,next))match_rule(r,depth+1,next,known,output);
 }
}
static std::unordered_set<Lit,Hash> closure(){
 std::unordered_set<Lit,Hash> known(input.begin(),input.end());
 for(int round=0;round<max_rounds;++round){
   std::vector<Lit> snapshot(known.begin(),known.end());
   std::unordered_set<Lit,Hash> additions;
   for(const auto& r:rules)match_rule(r,0,{{-1,-1,-1,-1}},snapshot,additions);
   std::size_t before=known.size();
   for(const auto& q:additions)known.insert(q);
   if(known.size()>max_derived)throw std::runtime_error("logic budget exceeded");
   if(known.size()==before)return known;
 }
 throw std::runtime_error("logic rounds exceeded");
}
extern "C" int spark_logic_answer_v1(int p,int a,int b,int neg){
 try {static auto known=closure();
   auto pos=known.count({p,neg, b==-9999?1:2,a,b})!=0;
   auto negative=known.count({p,1-neg,b==-9999?1:2,a,b})!=0;
   return pos&&negative?3:pos?1:negative?2:0;
 } catch(...) {return -1;}
}
'''
    source=source.replace('FACTS_PLACEHOLDER',fact_txt).replace('RULES_PLACEHOLDER',rule_txt)
    return source,predicates,entities


class NativeLogic:
    def __init__(self,lib,predicates,entities):
        self.lib=lib
        self.predicates=predicates
        self.entities=entities
        lib.spark_logic_answer_v1.argtypes=(ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)
        lib.spark_logic_answer_v1.restype=ctypes.c_int

    @classmethod
    def compile(cls,engine:LogicProgram,out:Path):
        out=Path(out)
        out.parent.mkdir(parents=True,exist_ok=True)
        source,preds,entities=cpp_source(engine)
        cpp=out.with_suffix('.cpp');cpp.write_text(source)
        compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if compiler is None:raise RuntimeError('C++17 compiler required')
        subprocess.run([compiler,'-O2','-std=c++17','-fPIC','-shared',str(cpp),'-o',str(out)],
                       check=True,capture_output=True,text=True)
        out.with_suffix('.vocabulary.json').write_text(json.dumps({'predicates':preds,'entities':entities},sort_keys=True)+'\n')
        return cls(ctypes.CDLL(str(out.resolve())),preds,entities)

    def ask(self,question):
        a=parse_question(question)
        if a.pred not in self.predicates or any(s not in self.entities for s in a.args):
            return 'unknown'
        args=[self.entities[s] for s in a.args]
        value=self.lib.spark_logic_answer_v1(self.predicates[a.pred],args[0],
                    args[1] if len(args)==2 else -9999,int(a.neg))
        if value<0:raise RuntimeError('native deduction resource budget exceeded')
        return ('unknown','yes','no','inconsistent')[value]

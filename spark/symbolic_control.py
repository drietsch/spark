"""Learn branching recurrent mathematical programs from transition observations.

Train-only guard search, rational polynomial leaves, and direct C++ compilation.
Constrained research grammar; no matrix/lookup in compiled inference.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from spark.symbolic_program import evaluate, synthesize, to_cpp, validate as validate_expr

FORMAT = "spark-conditional-program-v1"

def read_guard(g, s, x):
    z={"state":s,"input":x,"sum":s+x}[g["variable"]]
    return z == g["value"] if g["comparison"]=="eq" else z < g["value"]

def guard_cpp(g):
    v={"state":"s","input":"x","sum":"(s + x)"}[g["variable"]]
    op="==" if g["comparison"]=="eq" else "<"
    return f"({v} {op} {float(g['value']):.17g})"

def validate_program(n, depth=0):
    if depth>4 or not isinstance(n,dict):raise ValueError("malformed program")
    if n.get("op")=="leaf":
        if set(n)!={"op","expr"}:raise ValueError("malformed leaf")
        validate_expr(n["expr"])
    elif n.get("op")=="if":
        if set(n)!={"op","guard","then","else"}:raise ValueError("malformed if")
        g=n["guard"]
        if (not isinstance(g,dict) or set(g)!={"variable","comparison","value"}
            or g["variable"] not in ("state","input","sum")
            or g["comparison"] not in ("eq","lt") or type(g["value"]) not in (int,float)
            or not math.isfinite(g["value"]) or abs(g["value"])>10000):
            raise ValueError("invalid guard")
        if g["comparison"]=="eq" and g["variable"]!="input":
            raise ValueError("equality guards must refer to event inputs")
        validate_program(n["then"],depth+1);validate_program(n["else"],depth+1)
    else:raise ValueError("unsupported program operation")

def evaluate_program(n,s,x):
    if n["op"]=="leaf":return evaluate(n["expr"],s,x)
    return evaluate_program(n["then"] if read_guard(n["guard"],s,x) else n["else"],s,x)

def program_cpp(n):
    if n["op"]=="leaf":return to_cpp(n["expr"])
    return f"({guard_cpp(n['guard'])} ? {program_cpp(n['then'])} : {program_cpp(n['else'])})"

def compiled_source(n):
    validate_program(n)
    return ("// SPARK generated conditional mathematical recurrence.\n"
            "#include <cstddef>\n"
            "extern \"C\" double spark_control_step_v1(double s,double x) {\n"
            f"  return {program_cpp(n)};\n"
            "}\n"
            "extern \"C\" double spark_control_rollout_v1(double s,const double* xs,std::size_t n) {\n"
            "  for(std::size_t i=0;i<n;++i)s=spark_control_step_v1(s,xs[i]);\n"
            "  return s;\n}\n")

def node_cost(n):
    if n["op"]=="leaf":
        def count(e):return 1+sum(count(e[k]) for k in ("left","right","arg") if k in e)
        return count(n["expr"])
    return 4+node_cost(n["then"])+node_cost(n["else"])

def calculate_mse(rows,n):
    return sum((evaluate_program(n,s,x)-y)**2 for s,x,y in rows)/len(rows)

def leaf_fit(rows,max_degree):
    return {"op":"leaf","expr":synthesize(rows,max_degree=max_degree,denominator_limit=32).expression}

def guards_for(rows):
    out=[];values=sorted({x for _,x,_ in rows})
    if len(values)<=48:
        for x in values:
            if abs(x)<=20:out.append({"variable":"input","comparison":"eq","value":x})
    for variable in ("input","state","sum"):
        vals=sorted({round({"input":x,"state":s,"sum":s+x}[variable],7) for s,x,_ in rows})
        if len(vals)<=32:
            cuts=[(a+b)/2 for a,b in zip(vals,vals[1:])]
        else:
            lo,hi=vals[0],vals[-1]
            cuts=[float(k)+0.5 for k in range(max(-12,math.floor(lo)),min(13,math.ceil(hi)))]
            cuts += [vals[int(len(vals)*q)] for q in (.1,.25,.5,.75,.9)]
        for c in sorted(set(cuts)):
            out.append({"variable":variable,"comparison":"lt","value":float(c)})
    return out

@dataclass(frozen=True)
class LearnedControl:
    tree:dict
    train_mse:float
    branches:int
    complexity:int
    def step(self,s,x):return evaluate_program(self.tree,s,x)

def discover(rows,*,max_depth=2,max_degree=2,tolerance=1e-10,min_branch=24):
    if not (0<=max_depth<=3 and 0<=max_degree<=3 and min_branch>=12):
        raise ValueError("invalid configuration")
    if len(rows)<2*min_branch:raise ValueError("insufficient training data")
    rows=[tuple(map(float,r)) for r in rows]
    if any(len(r)!=3 or not all(math.isfinite(v) for v in r) for r in rows):
        raise ValueError("nonfinite or malformed transition")
    cache={}
    def fit(part):
        key=tuple(part)
        if key not in cache:
            tree=leaf_fit(part,max_degree)
            cache[key]=(tree,calculate_mse(part,tree))
        return cache[key]
    def search(part,depth):
        base,mse=fit(part)
        if depth==0 or mse<=tolerance:return base
        best=base;best_loss=mse
        for guard in guards_for(part):
            yes=[r for r in part if read_guard(guard,r[0],r[1])]
            if not min_branch<=len(yes)<=len(part)-min_branch:continue
            no=[r for r in part if not read_guard(guard,r[0],r[1])]
            a,ae=fit(yes);b,be=fit(no)
            loss=(len(yes)*ae+len(no)*be)/len(part)
            if loss<best_loss-max(tolerance,mse*0.0005):
                best_loss=loss
                best={"op":"if","guard":guard,"then":a,"else":b}
        if best["op"]=="if" and depth>1:
            g=best["guard"]
            yes=[r for r in part if read_guard(g,r[0],r[1])]
            no=[r for r in part if not read_guard(g,r[0],r[1])]
            best={"op":"if","guard":g,
                "then":search(yes,depth-1) if len(yes)>=2*min_branch else best["then"],
                "else":search(no,depth-1) if len(no)>=2*min_branch else best["else"]}
        return best
    tree=search(rows,max_depth)
    if max_depth>=2:
        ranked=[]
        for guard in guards_for(rows):
            yes=[r for r in rows if read_guard(guard,r[0],r[1])]
            if not min_branch<=len(yes)<=len(rows)-min_branch:continue
            no=[r for r in rows if not read_guard(guard,r[0],r[1])]
            _,ae=fit(yes);_,be=fit(no)
            ranked.append(((len(yes)*ae+len(no)*be)/len(rows),guard,yes,no))
        ranked.sort(key=lambda v:v[0])
        shortlist=ranked[:12]+[r for r in ranked if r[1]["comparison"]=="eq"]
        def quality(n):
            error=calculate_mse(rows,n)
            return (int(error>tolerance),error if error>tolerance else 0.0,node_cost(n))
        best_score=quality(tree)
        for _,guard,yes,no in shortlist:
            a=search(yes,max_depth-1) if len(yes)>=2*min_branch else fit(yes)[0]
            b=search(no,max_depth-1) if len(no)>=2*min_branch else fit(no)[0]
            candidate={"op":"if","guard":guard,"then":a,"else":b}
            q=quality(candidate)
            if q<best_score:tree,best_score=candidate,q
    validate_program(tree)
    def branches(n):return 0 if n["op"]=="leaf" else 1+branches(n["then"])+branches(n["else"])
    return LearnedControl(tree,calculate_mse(rows,tree),branches(tree),node_cost(tree))

def save(path,tree,metadata=None):
    validate_program(tree);path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"format":FORMAT,"tree":tree,"metadata":metadata or {}},
                               sort_keys=True,indent=2)+"\n")

def load(path):
    obj=json.loads(Path(path).read_text())
    if obj.get("format")!=FORMAT or set(obj)!={"format","tree","metadata"}:
        raise ValueError("unsupported SPARK control format")
    validate_program(obj["tree"])
    return obj["tree"]

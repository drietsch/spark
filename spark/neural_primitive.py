"""M12: a learnable gated dense *function* embedded in a typed symbolic program.

Training needs NumPy. Runtime is either Python scalar math or compiled C++17;
no neural inference framework, lookup index, or runtime weight matrices.
The graph surrounding the primitive is intentionally prescribed for this
first experiment; the neural primitive's parameters are learned from examples.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

FORMAT = 'spark-hybrid-neural-symbolic-v1'
WIDTH_LIMIT = 32


def _sigmoid(v: float) -> float:
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def _checked_number(x):
    if type(x) not in (float, int) or not math.isfinite(x) or abs(x) > 1e6:
        raise ValueError('invalid finite coefficient')


def validate_expression(e: dict, depth: int=0):
    if not isinstance(e, dict) or depth > 20:
        raise ValueError('malformed expression tree')
    op=e.get('op')
    if op == 'state':
        if set(e) != {'op','index'} or type(e['index']) is not int or not 0 <= e['index'] < 4:
            raise ValueError('invalid state register')
    elif op in ('input','context','neural'):
        if set(e) != {'op'}:
            raise ValueError('unexpected expression fields')
    elif op == 'const':
        if set(e) != {'op','num','den'} or type(e['num']) is not int or type(e['den']) is not int or not 1 <= e['den'] <= 128 or abs(e['num']) > 100000:
            raise ValueError('invalid rational constant')
    elif op in ('add','sub','mul'):
        if set(e) != {'op','left','right'}:
            raise ValueError('invalid binary operation')
        validate_expression(e['left'],depth+1)
        validate_expression(e['right'],depth+1)
    else:
        raise ValueError('unknown mathematical operation')


def validate(program: dict):
    if not isinstance(program,dict) or set(program) != {'format','registers','neural','args','updates'} or program['format']!=FORMAT or program['registers'] != 4:
        raise ValueError('malformed hybrid program')
    m=program['neural']
    if not isinstance(m,dict) or set(m) != {'type','width','w_hidden','b_hidden','w_gate','b_gate','w_out','b_out'} or m['type']!='gated_dense':
        raise ValueError('invalid neural function')
    w=m['width']
    if type(w) is not int or not 1<=w<=WIDTH_LIMIT:
        raise ValueError('invalid hidden width')
    for key in ('w_hidden','w_gate'):
        a=m[key]
        if not isinstance(a,list) or len(a)!=3 or any(not isinstance(row,list) or len(row)!=w for row in a):
            raise ValueError('invalid dense connection geometry')
        for row in a:
            for v in row:_checked_number(v)
    for key in ('b_hidden','b_gate','w_out'):
        a=m[key]
        if not isinstance(a,list) or len(a)!=w:raise ValueError('invalid vector geometry')
        for v in a:_checked_number(v)
    _checked_number(m['b_out'])
    if not isinstance(program['args'],list) or len(program['args'])!=3:
        raise ValueError('neural function takes exactly 3 symbolic arguments')
    if not isinstance(program['updates'],list) or len(program['updates'])!=4:
        raise ValueError('four symbolic register updates required')
    for e in program['args']+program['updates']:validate_expression(e)
    # The neural call is shared once, so cyclic references via args are forbidden.
    def has_neural(e):
        return e['op']=='neural' or any(has_neural(e[k]) for k in ('left','right') if k in e)
    if any(has_neural(e) for e in program['args']):
        raise ValueError('recursive neural function argument')


def neural_scalar(m: dict, z: tuple[float,float,float]):
    h=[]
    for j in range(m['width']):
        a=m['b_hidden'][j]+sum(z[i]*m['w_hidden'][i][j] for i in range(3))
        g=m['b_gate'][j]+sum(z[i]*m['w_gate'][i][j] for i in range(3))
        h.append(math.tanh(a)*_sigmoid(g))
    return m['b_out']+sum(h[j]*m['w_out'][j] for j in range(m['width']))


def _expression(e,s,x,c,q):
    op=e['op']
    if op=='state':return s[e['index']]
    if op=='input':return x
    if op=='context':return c
    if op=='neural':return q
    if op=='const':return e['num']/e['den']
    a=_expression(e['left'],s,x,c,q)
    b=_expression(e['right'],s,x,c,q)
    return a+b if op=='add' else a-b if op=='sub' else a*b


@dataclass(frozen=True)
class HybridProgram:
    payload:dict

    def __post_init__(self):validate(self.payload)

    @property
    def parameter_count(self):
        w=self.payload['neural']['width']
        return 3*w*2+3*w+1

    def step(self,state,x,context):
        if len(state)!=4:raise ValueError('wrong state vector dimension')
        s=tuple(float(v) for v in state); x=float(x); c=float(context)
        args=tuple(_expression(e,s,x,c,0.0) for e in self.payload['args'])
        q=neural_scalar(self.payload['neural'],args)
        return tuple(_expression(e,s,x,c,q) for e in self.payload['updates'])

    def rollout(self,state,inputs,contexts):
        current=tuple(state)
        for x,c in zip(inputs,contexts,strict=True):
            current=self.step(current,x,c)
        return current


def save(path:Path,model:HybridProgram):
    validate(model.payload)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(model.payload,sort_keys=True,separators=(',',':'))+'\n')


def load(path:Path):
    return HybridProgram(json.loads(path.read_text()))


def _cpp_num(v):
    _checked_number(v)
    return format(float(v),'.17g')


def _cpp_expr(e):
    op=e['op']
    if op=='state':return f's[{e["index"]}]'
    if op=='input':return 'x'
    if op=='context':return 'context'
    if op=='neural':return 'q'
    if op=='const':return f'({e["num"]}.0/{e["den"]}.0)'
    return f'({_cpp_expr(e["left"])} '+{'add':'+','sub':'-','mul':'*'}[op]+f' {_cpp_expr(e["right"])})'


def cpp_source(program:dict):
    validate(program)
    m=program['neural']; w=m['width']
    lines=['// Generated SPARK mathematical program with embedded learned function.',
           '// All trained constants are literals; no runtime tensors or parameter lookup.',
           '#include <cmath>','#include <cstddef>',
           'static inline double spark_sigmoid(double a) {',
           '  if (a>=0.0) return 1.0/(1.0+std::exp(-a));',
           '  const double t=std::exp(a); return t/(1.0+t);','}',
           'static inline double spark_neural(double z0,double z1,double z2) {',
           f'  double v = {_cpp_num(m["b_out"])};']
    for j in range(w):
        hid=' + '.join([_cpp_num(m['b_hidden'][j])]+[f'({_cpp_num(m["w_hidden"][i][j])} * z{i})' for i in range(3)])
        gate=' + '.join([_cpp_num(m['b_gate'][j])]+[f'({_cpp_num(m["w_gate"][i][j])} * z{i})' for i in range(3)])
        lines.extend([f'  const double h{j} = std::tanh({hid});',f'  const double g{j} = spark_sigmoid({gate});',
                      f'  v += ({_cpp_num(m["w_out"][j])}) * h{j} * g{j};'])
    lines.extend(['  return v;','}',
                  'extern "C" void spark_hybrid_step_v1(const double* s,double x,double context,double* out) {'])
    for i,e in enumerate(program['args']):lines.append(f'  const double z{i} = {_cpp_expr(e)};')
    lines.append('  const double q = spark_neural(z0,z1,z2);')
    for i,e in enumerate(program['updates']):lines.append(f'  out[{i}] = {_cpp_expr(e)};')
    lines.extend(['}',
          'extern "C" void spark_hybrid_rollout_v1(double* s,const double* xs,const double* contexts,std::size_t n) {',
          '  double next[4];',
          '  for (std::size_t i=0; i<n; ++i) {',
          '    spark_hybrid_step_v1(s,xs[i],contexts[i],next);',
          '    for (std::size_t j=0;j<4;++j) s[j]=next[j];',
          '  }','}'])
    return '\n'.join(lines)+'\n'


def _np_sigmoid(z):
    z=np.clip(z,-40.0,40.0)
    return 1.0/(1.0+np.exp(-z))


def _init(width,seed):
    rng=np.random.default_rng(seed)
    return dict(w_hidden=rng.normal(0,.45,(3,width)),b_hidden=np.zeros(width),
                w_gate=rng.normal(0,.3,(3,width)),b_gate=np.zeros(width),
                w_out=rng.normal(0,.25,width),b_out=np.zeros(1))


def _train_arrays(z,y, *,width,seed,epochs,lr):
    if np is None:raise ImportError('NumPy needed for offline neural function fitting')
    z=np.asarray(z,dtype=np.float64);y=np.asarray(y,dtype=np.float64).reshape(-1)
    if z.ndim!=2 or z.shape[1]!=3 or len(z)!=len(y) or len(z)<64 or not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError('invalid neural function training samples')
    if not 1<=width<=WIDTH_LIMIT or not 1<=epochs<=5000:
        raise ValueError('invalid training configuration')
    p=_init(width,seed);m={k:np.zeros_like(v) for k,v in p.items()};v={k:np.zeros_like(v) for k,v in p.items()}
    n=len(z)
    for step in range(1,epochs+1):
        a=z@p['w_hidden']+p['b_hidden'];h=np.tanh(a)
        g=_np_sigmoid(z@p['w_gate']+p['b_gate']);u=h*g
        pred=u@p['w_out']+p['b_out'][0]
        err=(pred-y)*(2.0/n)
        grads={}
        grads['w_out']=u.T@err
        grads['b_out']=np.array([err.sum()])
        du=err[:,None]*p['w_out'][None,:]
        da=du*g*(1-h*h)
        dg=du*h*g*(1-g)
        grads['w_hidden']=z.T@da
        grads['b_hidden']=da.sum(axis=0)
        grads['w_gate']=z.T@dg
        grads['b_gate']=dg.sum(axis=0)
        # Gradient clipping avoids pathological synthetic target instability.
        norm=math.sqrt(sum(float(np.sum(t*t)) for t in grads.values()))
        scale=min(1.,10./max(norm,1e-12))
        for k,gradient in grads.items():
            gradient=gradient*scale
            m[k]=.9*m[k]+.1*gradient
            v[k]=.999*v[k]+.001*(gradient**2)
            p[k]-=lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
    return {k:(a.tolist() if a.ndim else float(a)) for k,a in p.items()}


def fit_neural(z,y, *,width=12,epochs=900,lr=.016,seed=12):
    p=_train_arrays(z,y,width=width,seed=seed,epochs=epochs,lr=lr)
    return {'type':'gated_dense','width':width,'w_hidden':p['w_hidden'],
            'b_hidden':p['b_hidden'],'w_gate':p['w_gate'],'b_gate':p['b_gate'],
            'w_out':p['w_out'],'b_out':p['b_out'][0]}


def state(i):return {'op':'state','index':i}
def variable(op):return {'op':op}
def binary(op,left,right):return {'op':op,'left':left,'right':right}
def rational(num,den=1):return {'op':'const','num':num,'den':den}


def build_hybrid(neural):
    """M12 scaffold: learn neural q and embed it once in shared symbolic state updates.

    Coefficients in this scaffold are supplied, not discovered. M12 is about
    first-class neural-function composition, NOT automatic hybrid graph search.
    """
    q=variable('neural')
    return HybridProgram({'format':FORMAT,'registers':4,'neural':neural,
      'args':[binary('sub',state(0),state(1)),variable('input'),variable('context')],
      'updates':[variable('input'),state(0),
                 binary('add',binary('mul',rational(3,5),state(2)),q),
                 binary('add',binary('add',binary('mul',rational(1,2),state(3)),
                                       binary('mul',rational(1,5),state(2))),q)]})

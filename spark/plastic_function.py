"""M13: an online-plastic, dynamically sized neural FUNCTION inside a symbolic program.

The *algorithm* is compiled. Numerical coefficients and active units are live,
mutable data, updated after deployment from observed transitions; they are NOT
baked into generated machine code. No neural framework is needed in runtime.

The outer mathematical AST is typed and compiled but can be recompiled when
structural program search changes it. In this experiment it is prescribed.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

# Restricted grammar shared conceptually with M12; self-contained to make the
# online runtime independent of offline neural fitting dependencies.
def validate_expression(e, depth=0):
    if not isinstance(e, dict) or depth>20: raise ValueError('invalid expression')
    op=e.get('op')
    if op=='state':
        if set(e)!={'op','index'} or type(e['index']) is not int or not 0<=e['index']<4: raise ValueError('invalid register')
    elif op in ('input','context','neural'):
        if set(e)!={'op'}: raise ValueError('invalid variable')
    elif op=='const':
        if set(e)!={'op','num','den'} or type(e['num']) is not int or type(e['den']) is not int or not 1<=e['den']<=128 or abs(e['num'])>100000: raise ValueError('invalid rational')
    elif op in ('add','sub','mul'):
        if set(e)!={'op','left','right'}:raise ValueError('invalid binary operation')
        validate_expression(e['left'],depth+1);validate_expression(e['right'],depth+1)
    else: raise ValueError('unsupported operation')


def _expression(e,s,x,c,q):
    op=e['op']
    if op=='state':return s[e['index']]
    if op=='input':return x
    if op=='context':return c
    if op=='neural':return q
    if op=='const':return e['num']/e['den']
    a=_expression(e['left'],s,x,c,q);b=_expression(e['right'],s,x,c,q)
    return a+b if op=='add' else a-b if op=='sub' else a*b


def _cpp_expr(e):
    op=e['op']
    if op=='state':return f's[{e["index"]}]'
    if op=='input':return 'x'
    if op=='context':return 'context'
    if op=='neural':return 'q'
    if op=='const':return f'({e["num"]}.0/{e["den"]}.0)'
    return f'({_cpp_expr(e["left"])} '+{'add':'+','sub':'-','mul':'*'}[op]+f' {_cpp_expr(e["right"])})'


FORMAT = 'spark-plastic-function-v1'
MAX_UNITS = 32
PARAMS_PER_UNIT = 9  # hidden 3 + bias, gate 3 + bias, readout 1


def default_program() -> dict:
    def var(op): return {'op': op}
    def st(i): return {'op': 'state', 'index': i}
    def num(n, d=1): return {'op': 'const', 'num': n, 'den': d}
    def op(k, l, r): return {'op': k, 'left': l, 'right': r}
    q = var('neural')
    return {'format': FORMAT, 'registers': 4,
            'args': [op('sub', st(0), st(1)), var('input'), var('context')],
            'updates': [var('input'), st(0),
                        op('add', op('mul', num(3, 5), st(2)), q),
                        op('add', op('add', op('mul', num(1, 2), st(3)),
                                           op('mul', num(1, 5), st(2))), q)]}


def validate_program(p: dict) -> None:
    if not isinstance(p, dict) or set(p) != {'format', 'registers', 'args', 'updates'}:
        raise ValueError('invalid plastic program fields')
    if p['format'] != FORMAT or type(p['registers']) is not int or p['registers'] != 4:
        raise ValueError('invalid plastic program version/registers')
    if not isinstance(p['args'], list) or len(p['args']) != 3:
        raise ValueError('requires three mathematical arguments')
    if not isinstance(p['updates'], list) or len(p['updates']) != 4:
        raise ValueError('requires four register updates')
    for expr in p['args'] + p['updates']:
        validate_expression(expr)
    def contains_neural(expr):
        return expr['op'] == 'neural' or any(contains_neural(expr[k]) for k in ('left','right') if k in expr)
    if any(contains_neural(e) for e in p['args']):
        raise ValueError('recursive neural function invocation')
    if not any(contains_neural(e) for e in p['updates']):
        raise ValueError('neural primitive must participate in state updates')


def dual_eval(e, s, x, c, q):
    op = e['op']
    if op == 'neural': return q, 1.0
    if op in ('state', 'input', 'context', 'const'):
        return _expression(e, s, x, c, q), 0.0
    a,da = dual_eval(e['left'],s,x,c,q)
    b,db = dual_eval(e['right'],s,x,c,q)
    if op == 'add': return a+b, da+db
    if op == 'sub': return a-b, da-db
    return a*b, da*b+a*db


def _sigmoid(x):
    if x >= 0: return 1.0/(1.0+math.exp(-x))
    v=math.exp(x)
    return v/(1.0+v)


@dataclass
class PlasticFunction:
    """Tiny gated dense function with runtime topology and plastic coefficients."""
    capacity: int
    active: int
    units: list[list[float]]
    bias: float = 0.0

    @classmethod
    def initialize(cls, *,capacity=12,active=8,seed=17):
        if type(capacity) is not int or not 1<=capacity<=MAX_UNITS or type(active) is not int or not 1<=active<=capacity:
            raise ValueError('invalid capacity/active count')
        r=random.Random(seed)
        units=[[r.uniform(-.45,.45) for _ in range(3)] + [0.0] +
               [r.uniform(-.32,.32) for _ in range(3)] + [0.0,r.uniform(-.35,.35)]
               for _ in range(capacity)]
        return cls(capacity,active,units)

    @property
    def parameter_count(self):
        return self.capacity*PARAMS_PER_UNIT+1

    def validate(self):
        if type(self.capacity) is not int or not 1<=self.capacity<=MAX_UNITS or type(self.active) is not int or not 1<=self.active<=self.capacity:
            raise ValueError('invalid mutable network size')
        if len(self.units)!=self.capacity or any(len(row)!=PARAMS_PER_UNIT for row in self.units):
            raise ValueError('invalid dynamic dense geometry')
        if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>8.0 for row in self.units for v in row):
            raise ValueError('invalid network coefficients')
        if not math.isfinite(self.bias) or abs(self.bias)>8.0:
            raise ValueError('invalid readout bias')

    def set_active(self, count):
        if type(count) is not int or not 1<=count<=self.capacity:
            raise ValueError('active count outside allocated capacity')
        self.active=count

    def flatten(self):
        return [v for row in self.units for v in row]+[self.bias]

    def forward(self,z):
        if len(z)!=3 or not all(math.isfinite(v) for v in z):
            raise ValueError('nonfinite neural arguments')
        y=self.bias
        cache=[]
        for j in range(self.active):
            w=self.units[j]
            h=math.tanh(w[3]+sum(z[i]*w[i] for i in range(3)))
            g=_sigmoid(w[7]+sum(z[i]*w[4+i] for i in range(3)))
            y+=w[8]*h*g
            cache.append((h,g))
        return y,cache

    def update(self,z,gradient,lr,cache):
        if not math.isfinite(gradient) or not math.isfinite(lr) or not 0 < lr <= .15:
            raise ValueError('invalid online learning rate/gradient')
        gradient=max(-2.,min(2.,gradient))
        for j,(h,g) in enumerate(cache):
            w=self.units[j];old_out=w[8]
            a=gradient*old_out*g*(1-h*h)
            b=gradient*old_out*h*g*(1-g)
            for i in range(3):
                w[i]=max(-8.,min(8.,w[i]-lr*a*z[i]))
            w[3]=max(-8.,min(8.,w[3]-lr*a))
            for i in range(3):
                w[4+i]=max(-8.,min(8.,w[4+i]-lr*b*z[i]))
            w[7]=max(-8.,min(8.,w[7]-lr*b))
            w[8]=max(-8.,min(8.,w[8]-lr*gradient*h*g))
        self.bias=max(-8.,min(8.,self.bias-lr*gradient))


class PlasticProgram:
    def __init__(self, function: PlasticFunction, program=None):
        self.function=function
        self.program=default_program() if program is None else program
        validate_program(self.program)
        self.function.validate()
        self.steps=0

    def step(self,state,x,context):
        if len(state)!=4 or not all(math.isfinite(float(v)) for v in (*state,float(x),float(context))):
            raise ValueError('nonfinite symbolic state or input')
        s=tuple(float(v) for v in state); x=float(x); c=float(context)
        z=tuple(_expression(e,s,x,c,0.0) for e in self.program['args'])
        q,_=self.function.forward(z)
        return tuple(_expression(e,s,x,c,q) for e in self.program['updates'])

    def observe(self,state,x,context,target,*,lr=.04):
        """One immediate online update from the newly observed next state.

        Differentiate the compiled symbolic output expressions w.r.t. q,
        then differentiate q w.r.t. the mutable neural coefficients. No direct
        teacher-provided neural-function target is needed.
        """
        if len(target)!=4 or not all(math.isfinite(v) for v in target):
            raise ValueError('invalid observed state')
        s=tuple(float(v) for v in state); x=float(x);c=float(context)
        z=tuple(_expression(e,s,x,c,0.0) for e in self.program['args'])
        q,cache=self.function.forward(z)
        vals=[dual_eval(e,s,x,c,q) for e in self.program['updates']]
        numerator=sum((v-y)*d for (v,d),y in zip(vals,target))
        denom=sum(d*d for _,d in vals)
        if not denom>1e-15:
            raise ValueError('observations do not inform neural function')
        gradient=numerator/denom
        self.function.update(z,gradient,lr,cache)
        self.steps+=1
        return sum((v-y)**2 for (v,_),y in zip(vals,target))/4.0

    def save(self,path):
        self.function.validate();validate_program(self.program)
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'program':self.program,'capacity':self.function.capacity,
             'active':self.function.active,'units':self.function.units,'bias':self.function.bias,
             'online_steps':self.steps},sort_keys=True,separators=(',',':'))+'\n')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if set(d)!={'program','capacity','active','units','bias','online_steps'}:
            raise ValueError('unknown mutable model serialization')
        f=PlasticFunction(d['capacity'],d['active'],d['units'],d['bias']);f.validate()
        obj=cls(f,d['program'])
        if type(d['online_steps']) is not int or d['online_steps']<0:
            raise ValueError('bad online step count')
        obj.steps=d['online_steps']
        return obj


def _cpp_dual(e):
    """Compile exact derivative of the symbolic expression w.r.t. neural q."""
    op=e['op']
    if op=='neural':return '1.0'
    if op in ('state','input','context','const'):return '0.0'
    da=_cpp_dual(e['left']);db=_cpp_dual(e['right'])
    if op=='add':return f'({da}+{db})'
    if op=='sub':return f'({da}-{db})'
    return f'(({da})*({_cpp_expr(e["right"])})+({_cpp_expr(e["left"])})*({db}))'


def cpp_source(program,*,capacity=12):
    """Compile math and plastic gradient routines, NOT parameter values.

    Runtime stores a live array of coefficients and an active hidden width.
    The runtime network can update its behavior without recompilation.
    """
    validate_program(program)
    if type(capacity) is not int or not 1<=capacity<=MAX_UNITS:
        raise ValueError('invalid native capacity')
    lines=[
        '// M13: compiled symbolic math + LIVE plastic neural function.',
        '// Coefficients and active width are mutable at runtime; NOT literals.',
        '#include <cmath>', '#include <cstddef>', '#include <new>', '#include <algorithm>',
        f'struct SparkPlastic {{ double w[{capacity}][9]{{}}; double bias{{}}; int active{{1}}; }};',
        'static inline double sigmoid(double x) { if (x>=0) return 1.0/(1.0+std::exp(-x)); double e=std::exp(x);return e/(1.0+e); }',
        'extern "C" void* spark_plastic_create_v1() { return new(std::nothrow) SparkPlastic(); }',
        'extern "C" void spark_plastic_destroy_v1(void* p) { delete static_cast<SparkPlastic*>(p); }',
        f'extern "C" std::size_t spark_plastic_parameter_count_v1() {{ return {capacity*9+1}; }}',
        'extern "C" int spark_plastic_set_v1(void* v,const double* p,std::size_t n,int active) {',
        f'  if(!v || !p || n!={capacity*9+1} || active<1 || active>{capacity}) return 0;',
        f'  for(std::size_t i=0;i<n;++i)if(!std::isfinite(p[i]) || std::abs(p[i])>8.0)return 0;',
        '  auto& m=*static_cast<SparkPlastic*>(v);',
        f'  for(int j=0;j<{capacity};++j)for(int k=0;k<9;++k)m.w[j][k]=p[j*9+k];',
        f'  m.bias=p[{capacity*9}];m.active=active;return 1;',
        '}',
        'extern "C" int spark_plastic_get_v1(void* v,double* p,std::size_t n) {',
        f'  if(!v || !p || n!={capacity*9+1}) return 0;',
        '  auto& m=*static_cast<SparkPlastic*>(v);',
        f'  for(int j=0;j<{capacity};++j)for(int k=0;k<9;++k)p[j*9+k]=m.w[j][k];',
        f'  p[{capacity*9}]=m.bias;return 1;',
        '}',
        'extern "C" int spark_plastic_active_v1(void* v) { return v?static_cast<SparkPlastic*>(v)->active:0; }',
        'extern "C" int spark_plastic_resize_v1(void* v,int active) {',
        f'  if(!v||active<1||active>{capacity})return 0;',
        '  static_cast<SparkPlastic*>(v)->active=active;return 1;',
        '}',
        'static inline double neural(SparkPlastic& m,const double z[3],double h[],double g[]) {',
        '  double q=m.bias;',
        '  for(int j=0;j<m.active;++j){',
        '    double* w=m.w[j];',
        '    double a=w[3],b=w[7];',
        '    for(int i=0;i<3;++i){a+=w[i]*z[i];b+=w[4+i]*z[i];}',
        '    h[j]=std::tanh(a);g[j]=sigmoid(b);q+=w[8]*h[j]*g[j];',
        '  }return q;',
        '}',
        'static inline int run(SparkPlastic& m,const double* s,double x,double context,double* out,',
        '                      double z[3],double h[],double g[],double* derivative) {',
        '  if(!s||!out||!std::isfinite(x)||!std::isfinite(context))return 0;',
        '  for(int i=0;i<4;++i)if(!std::isfinite(s[i]))return 0;',
    ]
    for i,e in enumerate(program['args']):lines.append(f'  z[{i}] = {_cpp_expr(e)};')
    lines.extend(['  for(int i=0;i<3;++i)if(!std::isfinite(z[i]))return 0;',
                  '  const double q=neural(m,z,h,g);'])
    for i,e in enumerate(program['updates']):
        lines.append(f'  out[{i}] = {_cpp_expr(e)};')
        lines.append(f'  derivative[{i}] = {_cpp_dual(e)};')
    lines.append('  return 1;')
    lines.append('}')
    lines += [
        'extern "C" int spark_plastic_step_v1(void* v,const double* s,double x,double context,double* out) {',
        '  if(!v)return 0;',
        f'  double z[3],h[{capacity}],g[{capacity}],d[4];',
        '  return run(*static_cast<SparkPlastic*>(v),s,x,context,out,z,h,g,d);',
        '}',
        'extern "C" int spark_plastic_observe_v1(void* v,const double* s,double x,double context,',
        '                                           const double* target,double lr) {',
        '  if(!v||!target||!std::isfinite(lr)||lr<=0||lr>0.15)return 0;',
        '  for(int i=0;i<4;++i)if(!std::isfinite(target[i]))return 0;',
        '  auto& m=*static_cast<SparkPlastic*>(v);',
        f'  double z[3],h[{capacity}],g[{capacity}],d[4],out[4];',
        '  if(!run(m,s,x,context,out,z,h,g,d))return 0;',
        '  double numerator=0.0,denom=0.0;',
        '  for(int i=0;i<4;++i){numerator+=(out[i]-target[i])*d[i];denom+=d[i]*d[i];}',
        '  if(denom<=1e-15)return 0;',
        '  const double grad=std::clamp(numerator/denom,-2.0,2.0);',
        '  for(int j=0;j<m.active;++j){',
        '    double* w=m.w[j];double oldout=w[8];',
        '    const double a=grad*oldout*g[j]*(1-h[j]*h[j]);',
        '    const double b=grad*oldout*h[j]*g[j]*(1-g[j]);',
        '    for(int i=0;i<3;++i)w[i]=std::clamp(w[i]-lr*a*z[i],-8.0,8.0);',
        '    w[3]=std::clamp(w[3]-lr*a,-8.0,8.0);',
        '    for(int i=0;i<3;++i)w[4+i]=std::clamp(w[4+i]-lr*b*z[i],-8.0,8.0);',
        '    w[7]=std::clamp(w[7]-lr*b,-8.0,8.0);',
        '    w[8]=std::clamp(w[8]-lr*grad*h[j]*g[j],-8.0,8.0);',
        '  }',
        '  m.bias=std::clamp(m.bias-lr*grad,-8.0,8.0);return 1;',
        '}',
        'extern "C" double spark_plastic_native_stream_v1(void* v,const double* ss,const double* xs,',
        '                    const double* cs,const double* ys,std::size_t n,double lr) {',
        '  if(!v||!ss||!xs||!cs||!ys)return 0.0;',
        '  double checksum=0.0,out[4];',
        '  for(std::size_t i=0;i<n;++i){',
        '    if(!spark_plastic_step_v1(v,ss+4*i,xs[i],cs[i],out))return -1.0;',
        '    checksum+=out[2];',
        '    if(lr>0 && !spark_plastic_observe_v1(v,ss+4*i,xs[i],cs[i],ys+4*i,lr))return -1.0;',
        '  }return checksum;',
        '}',
    ]
    return '\n'.join(lines)+'\n'

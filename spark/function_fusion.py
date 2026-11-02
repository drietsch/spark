"""Audited exact map fusion; rules preserve scalar trees and reduction order.

The replay checker trusts these small local rule definitions, not the pass's
search decisions or logged claims. This is an engineering checker, not a
machine-checked proof of floating-point equivalence for arbitrary programs.
"""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from spark.function_ir import _broadcast, _expression, canonical, digest, require, validate
from spark.function_rewrite import read_program, readers_for
from spark.model_compiler import write_package


def fingerprint(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def _functions(program):return {**{f['name']:f for f in program['functions']},'$main':program['main']}


def _size(expression):
    return 1+sum(_size(a) for a in expression.get('args',[]))


def _count_arg(expression,index):
    return int(expression.get('op')=='arg' and expression.get('index')==index)+sum(_count_arg(a,index) for a in expression.get('args',[]))


def _expensive(expression):
    return expression['op'] in ('exp','erf','tanh','sqrt') or any(_expensive(a) for a in expression.get('args',[]))


def _substitute(expression,arguments,index_map=None):
    if expression['op']=='arg':return copy.deepcopy(arguments[expression['index']])
    if expression['op']=='index' and index_map is not None:return copy.deepcopy(index_map[expression['index']])
    return {k:([_substitute(a,arguments,index_map) for a in v] if k=='args' else copy.deepcopy(v)) for k,v in expression.items()}


def _replacement(function,types,producer_id,consumer_id):
    nodes={n['id']:n for n in function['nodes']};producer=nodes.get(producer_id);consumer=nodes.get(consumer_id)
    require(producer is not None and consumer is not None and producer['op']=='map','fusion requires an existing map producer')
    users={n['id'] for n in function['nodes'] if producer_id in n['inputs']}
    outputs=set(function['outputs'].values()) if 'outputs' in function else {function['output']}
    require(users=={consumer_id} and producer_id not in outputs,'producer must have one consumer and cannot be a public result')
    require(consumer['op'] in ('map','reduce','map_reduce'),'unsupported fusion consumer')
    if consumer['op']=='reduce':
        result=copy.deepcopy(consumer);result['op']='map_reduce';result['inputs']=list(producer['inputs'])
        result['attrs']={**consumer['attrs'],'expression':copy.deepcopy(producer['attrs']['expression'])}
        return result,'single-use-map-reduce'
    # A map_reduce's scalar expression uses its pre-reduction broadcast grid.
    target_shape=_broadcast([types[v][1] for v in consumer['inputs']])
    producer_shape=types[producer_id][1];shift=len(target_shape)-len(producer_shape)
    index_map={i:({'op':'literal','value':0} if dim==1 else {'op':'index','index':i+shift}) for i,dim in enumerate(producer_shape)}
    new_inputs=[]
    def operand(name):
        if name not in new_inputs:new_inputs.append(name)
        return {'op':'arg','index':new_inputs.index(name)}
    arguments=[]
    for i,name in enumerate(consumer['inputs']):
        if name!=producer_id:arguments.append(operand(name));continue
        occurrences=_count_arg(consumer['attrs']['expression'],i)
        require(not (_expensive(producer['attrs']['expression']) and occurrences>1),'fusion would duplicate a transcendental expression')
        arguments.append(_substitute(producer['attrs']['expression'],[operand(v) for v in producer['inputs']],index_map))
    expression=_substitute(consumer['attrs']['expression'],arguments)
    require(len(new_inputs)<=8 and _size(expression)<=96,'fusion scalar/operand cost budget exceeded')
    _expression(expression,len(new_inputs),len(target_shape))
    result=copy.deepcopy(consumer);result['inputs']=new_inputs;result['attrs']['expression']=expression
    require(_broadcast([types[v][1] for v in new_inputs])==target_shape,'fusion changes broadcast iteration space')
    return result,'single-use-map-map'


def fuse_program(program):
    validate(program);result=copy.deepcopy(program);records=[]
    for name,function in _functions(result).items():
        changed=True
        while changed:
            changed=False;all_types=validate(result);types=all_types['main'] if name=='$main' else all_types['$functions'][name]
            nodes={n['id']:n for n in function['nodes']}
            for consumer in list(function['nodes']):
                if consumer['op'] not in ('map','reduce','map_reduce'):continue
                for producer_id in dict.fromkeys(consumer['inputs']):
                    if producer_id not in nodes or nodes[producer_id]['op']!='map':continue
                    try:replacement,rule=_replacement(function,types,producer_id,consumer['id'])
                    except ValueError:continue
                    before={'producer':copy.deepcopy(nodes[producer_id]),'consumer':copy.deepcopy(consumer)}
                    function['nodes']=[replacement if n['id']==consumer['id'] else n for n in function['nodes'] if n['id']!=producer_id]
                    validate(result)
                    records.append({'function':name,'rule':rule,'producer':producer_id,'consumer':consumer['id'],
                        'before':before,'after':copy.deepcopy(replacement),'before_sha256':fingerprint(before),'after_sha256':fingerprint(replacement),
                        'preconditions':{'single_consumer':True,'not_public_output':True,'broadcast_index_mapping_checked':True,
                                         'scalar_tree_not_reassociated':True,'ascending_reduction_order_preserved':True},
                        'removed_materialization_shape':types[producer_id][1],'bytes_per_element':8})
                    changed=True;break
                if changed:break
    audit={'version':'spark-exact-fusion-audit-v1','source_sha256':fingerprint(program),'target_sha256':fingerprint(result),
           'records':records,'rule_counts':dict(Counter(r['rule'] for r in records)),
           'coefficient_bindings_unchanged':result['parameters']==program['parameters'],
           'floating_point_contract':'float64; no reassociation; reduction terms visited in original order; compiler -ffp-contract=off'}
    verify_fusion(program,result,audit)
    return result,audit


def verify_fusion(source,target,audit):
    """Replay every local rule and reject missing, altered or unjustified changes."""
    validate(source);validate(target)
    require(audit.get('version')=='spark-exact-fusion-audit-v1' and audit.get('source_sha256')==fingerprint(source),'audit source mismatch')
    replay=copy.deepcopy(source)
    for record in audit['records']:
        functions=_functions(replay);require(record['function'] in functions,'unknown audit function')
        function=functions[record['function']];nodes={n['id']:n for n in function['nodes']}
        actual={'producer':nodes.get(record['producer']),'consumer':nodes.get(record['consumer'])}
        require(actual==record['before'] and fingerprint(actual)==record['before_sha256'],'audit before-expression mismatch')
        all_types=validate(replay);types=all_types['main'] if record['function']=='$main' else all_types['$functions'][record['function']]
        replacement,rule=_replacement(function,types,record['producer'],record['consumer'])
        expected_conditions={'single_consumer':True,'not_public_output':True,'broadcast_index_mapping_checked':True,
                             'scalar_tree_not_reassociated':True,'ascending_reduction_order_preserved':True}
        require(record.get('preconditions')==expected_conditions and record.get('removed_materialization_shape')==types[record['producer']][1]
                and record.get('bytes_per_element')==8,'audit conditions or materialization claim mismatch')
        require(rule==record['rule'] and replacement==record['after'] and fingerprint(replacement)==record['after_sha256'],'invalid audit replacement')
        function['nodes']=[replacement if n['id']==record['consumer'] else n for n in function['nodes'] if n['id']!=record['producer']]
        validate(replay)
    require(canonical(replay)==canonical(target) and fingerprint(target)==audit['target_sha256'],'target contains unlogged changes')
    require(source['parameters']==target['parameters'] and source.get('coefficients_sha256')==target.get('coefficients_sha256'),'exact rewrite changed coefficients')
    require(audit.get('rule_counts')==dict(Counter(r['rule'] for r in audit['records'])) and audit.get('coefficient_bindings_unchanged') is True,'audit summary mismatch')
    return {'verified':True,'rewrites':len(audit['records']),'source_sha256':audit['source_sha256'],'target_sha256':audit['target_sha256']}


def compile_fused_program(source,destination):
    program=read_program(source);target,audit=fuse_program(program)
    tokenizer=Path(source)/'tokenizer'
    saved=write_package(target,readers_for(program,source),destination,tokenizer=tokenizer if tokenizer.exists() else None,optimize_program=False)
    require(canonical(saved)==canonical(target),'packaging changed the audited target')
    path=Path(destination)/'rewrite-audit.json';path.write_text(json.dumps(audit,indent=2)+'\n')
    verify_fusion(program,saved,audit)
    return saved,audit

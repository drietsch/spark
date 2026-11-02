"""Exact operation fusion, broadcast indices, audit replay and live captures."""
import copy
import ctypes.util
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from spark.function_ir import Builder,FORMAT,arg,expr,literal
from spark.function_fusion import fuse_program,verify_fusion,compile_fused_program
from spark.function_runtime import NativeFunctionProgram
from spark.model_compiler import typed,write_package

NATIVE=bool(shutil.which('c++') and ctypes.util.find_library('blas'))


def program(builder,output,functions=None):
    return {'format':FORMAT,'symbols':{'N':[1,64]},'parameters':[],'functions':functions or [],
            'main':{'inputs':builder.inputs,'nodes':builder.nodes,'outputs':{'result':output},'preconditions':[]}}


@unittest.skipUnless(NATIVE,'C++17 and BLAS required')
class FunctionFusionTests(unittest.TestCase):
    def compare(self,source,inputs):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);baseline=write_package(source,{},root/'baseline');target,audit=compile_fused_program(root/'baseline',root/'fused')
            self.assertTrue(verify_fusion(baseline,target,audit)['verified'])
            with NativeFunctionProgram(root/'baseline') as a,NativeFunctionProgram(root/'fused') as b:
                x=a.run(inputs)['result'];y=b.run(inputs)['result']
                self.assertEqual(x['shape'],y['shape']);self.assertEqual(bytes(x['values']),bytes(y['values']))
            return baseline,target,audit

    def test_map_reduction_preserves_noncontiguous_axis_order(self):
        b=Builder([typed('x',['N',3,2])]);t=b.node('transpose','x',axes=[2,0,1])
        squared=b.map(expr('mul',arg(0),arg(0)),t);out=b.node('reduce',squared,axis=1,kind='sum')
        source=program(b,out)
        _,target,audit=self.compare(source,{'x':[[[1.,2.],[3.,4.],[5.,6.]],[[7.,8.],[9.,10.],[11.,12.]]]})
        self.assertEqual(audit['rule_counts']['single-use-map-reduce'],1)
        self.assertEqual(target['main']['nodes'][-1]['op'],'map_reduce')

    def test_broadcast_producer_indices_are_remapped_in_fused_consumer(self):
        b=Builder([typed('x',[1,3]),typed('y',['N',1])])
        indexed=b.map(expr('add',arg(0),{'op':'index','index':0}),'x')
        out=b.map(expr('sub',arg(0),arg(1)),indexed,'y')
        _,_,audit=self.compare(program(b,out),{'x':[[1.,-0.,3.]],'y':[[10.],[20.]]})
        self.assertEqual(len(audit['records']),1)

    def test_rank_expansion_and_maximum_with_a_guard(self):
        b=Builder([typed('x',[3]),typed('y',['N',3])])
        indexed=b.map(expr('add',arg(0),{'op':'index','index':0}),'x')
        product=b.map(expr('mul',arg(0),arg(1)),indexed,'y')
        out=b.node('reduce',product,axis=1,kind='max')
        _,_,audit=self.compare(program(b,out),{'x':[1.,2.,3.],'y':[[1.,-2.,3.],[-1.,5.,.25]]})
        self.assertGreaterEqual(len(audit['records']),2)

    def test_shared_or_public_values_cannot_be_eliminated(self):
        b=Builder([typed('x',['N',2])]);p=b.map(expr('mul',arg(0),arg(0)),'x')
        a=b.node('reduce',p,axis=1,kind='sum');c=b.map(expr('add',arg(0),literal(1)),p)
        source=program(b,c);source['main']['outputs']['other']=a
        target,audit=fuse_program(source)
        self.assertEqual(audit['records'],[]);self.assertEqual(target,source)

    def test_audit_rejects_tampered_expression_and_unlogged_capture_change(self):
        b=Builder([typed('x',['N',2])]);p=b.map(expr('add',arg(0),literal(1)),'x');out=b.node('reduce',p,axis=1,kind='sum')
        source=program(b,out);target,audit=fuse_program(source)
        changed=copy.deepcopy(audit);changed['records'][0]['after']['attrs']['kind']='max'
        with self.assertRaises(ValueError):verify_fusion(source,target,changed)
        changed=copy.deepcopy(target);changed['symbols']['N']=[1,32]
        with self.assertRaises(ValueError):verify_fusion(source,changed,audit)

    def test_fusion_with_a_function_named_main(self):
        f=Builder([typed('x',['N',2])]);p=f.map(expr('mul',arg(0),arg(0)),'x');y=f.node('reduce',p,axis=1,kind='sum')
        main=Builder([typed('x',['N',2])]);out=main.node('call','x',function='main')
        _,_,audit=self.compare(program(main,out,[f.function('main',y)]),{'x':[[1.,2.],[3.,4.]]})
        self.assertEqual(len(audit['records']),1)

    def test_live_plastic_function_updates_preserve_parity(self):
        from spark.model_compiler import compile_plastic_function
        from spark.plastic_function import PlasticFunction
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);compile_plastic_function(PlasticFunction.initialize(capacity=8,active=5,seed=7),root/'baseline')
            _,audit=compile_fused_program(root/'baseline',root/'fused')
            self.assertGreater(len(audit['records']),0)
            with NativeFunctionProgram(root/'baseline') as a,NativeFunctionProgram(root/'fused') as b:
                inputs={'x':[[.1,.2,.3],[-1.,2.,-3.]]}
                for width in (1,8,5):
                    a.set_active_width('function0',width);b.set_active_width('function0',width)
                    self.assertEqual(bytes(a.run(inputs)['value']['values']),bytes(b.run(inputs)['value']['values']))
                a.set_function_coefficient('function0','readout_bias',0,.125);b.set_function_coefficient('function0','readout_bias',0,.125)
                self.assertEqual(bytes(a.run(inputs)['value']['values']),bytes(b.run(inputs)['value']['values']))


if __name__=='__main__':unittest.main()

"""Physical target rewrites, feature extraction and recovery statistics."""
import ctypes.util
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

NATIVE = bool(shutil.which('c++') and ctypes.util.find_library('blas'))
RESEARCH = all(importlib.util.find_spec(name) for name in ('numpy','torch','transformers'))


@unittest.skipUnless(NATIVE and RESEARCH, 'Optional source fixture and native dependencies required')
class FunctionRewriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import ModernBertConfig, ModernBertForMaskedLM
        from spark.model_compiler import compile_modernbert
        torch.set_num_threads(1); torch.manual_seed(61)
        config=ModernBertConfig(hidden_size=8,intermediate_size=12,num_hidden_layers=2,num_attention_heads=2,vocab_size=32,
                max_position_embeddings=64,local_attention=8,global_attn_every_n_layers=2,reference_compile=False,
                pad_token_id=0,cls_token_id=1,sep_token_id=2,mlp_bias=True)
        config._attn_implementation='eager';model=ModernBertForMaskedLM(config).eval()
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name.endswith('bias'):p.uniform_(-.02,.02)
        cls.temporary=tempfile.TemporaryDirectory();cls.source=Path(cls.temporary.name)/'source'
        compile_modernbert(model,cls.source,source={'fixture':'rewrite-test'})

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def inputs(self): return {'tokens':[1,4,5,6,7,3,2],'mask':[1]*7,'positions':list(range(7)),'queries':[5]}

    def test_physical_prefix_pruning_matches_live_width_and_reduces_storage(self):
        import numpy as np
        from spark.function_runtime import NativeFunctionProgram
        from spark.function_rewrite import prune_gated_function, read_program
        from spark.function_ir import digest
        original=read_program(self.source);checksum=digest(self.source/'coefficients.bin')
        with tempfile.TemporaryDirectory() as folder:
            target=prune_gated_function(self.source,folder,'function1',list(range(6)))
            self.assertTrue(target['metadata']['approximation'])
            self.assertEqual(original['coefficient_bytes']-target['coefficient_bytes'],6*(3*8+2)*4)
            with NativeFunctionProgram(self.source) as baseline,NativeFunctionProgram(folder) as modified:
                baseline.set_active_width('function1',6)
                expected=baseline.run(self.inputs())['logits']['values'];actual=modified.run(self.inputs())['logits']['values']
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        self.assertEqual(digest(self.source/'coefficients.bin'),checksum)

    def test_reordered_units_match_zeroed_discarded_readout_columns(self):
        import numpy as np
        from spark.function_runtime import NativeFunctionProgram
        from spark.function_rewrite import prune_gated_function
        keep=[7,2,11,0]
        with tempfile.TemporaryDirectory() as folder:
            prune_gated_function(self.source,folder,'function1',keep)
            with NativeFunctionProgram(self.source) as baseline,NativeFunctionProgram(folder) as modified:
                for row in range(8):
                    for col in range(12):
                        if col not in keep:baseline.set_function_coefficient('function1','output_weight',row*12+col,0.)
                np.testing.assert_allclose(modified.run(self.inputs())['logits']['values'],baseline.run(self.inputs())['logits']['values'],atol=1e-12,rtol=1e-12)

    def test_calibration_features_and_isolated_function_have_same_readout(self):
        import numpy as np
        from spark.function_runtime import NativeFunctionProgram
        from spark.function_rewrite import calibration_program,extract_function,parameter_array,read_program,gated_function
        program=read_program(self.source);_,spec=gated_function(program,'function1')
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);calibration_program(self.source,folder/'calibration','function1');extract_function(self.source,folder/'isolated','function1')
            with NativeFunctionProgram(folder/'calibration') as calibration,NativeFunctionProgram(folder/'isolated') as isolated:
                values=calibration.run(self.inputs())
                arr=lambda k:np.ctypeslib.as_array(values[k]['values']).reshape(values[k]['shape'])
                weight=parameter_array(program,self.source,spec['bindings']['output_weight']);bias=parameter_array(program,self.source,spec['bindings']['output_bias'])
                np.testing.assert_allclose(arr('features')@weight.T+bias,arr('target'),atol=1e-12,rtol=1e-12)
                actual=isolated.run({'x':arr('arguments').tolist()})['value']
                np.testing.assert_allclose(np.ctypeslib.as_array(actual['values']).reshape(actual['shape']),arr('target'),atol=1e-12,rtol=1e-12)

    def test_invalid_unit_lists_are_rejected(self):
        from spark.function_rewrite import prune_gated_function
        for keep in ([],[1,1],[-1],[12]):
            with tempfile.TemporaryDirectory() as folder,self.assertRaises(ValueError):prune_gated_function(self.source,folder,'function1',keep)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy required')
class ExperimentStatisticsTests(unittest.TestCase):
    def test_reconstruction_recovers_a_known_readout_from_training_pairs(self):
        import numpy as np
        from spark.function_experiment import fit_candidates
        rng=np.random.default_rng(71);train=rng.normal(size=(120,4));dev=rng.normal(size=(40,4))
        weight=np.asarray([[2.,0.,0.,0.],[0.,0.,3.,0.]])
        train_y=train@weight.T+[.1,-.2];dev_y=dev@weight.T+[.1,-.2]
        candidates=fit_candidates(train,train_y,dev,dev_y,weight,[2],[1e-6,1e-4])
        self.assertEqual(set(candidates[0]['keep']),{0,2});self.assertLess(candidates[0]['dev_nrmse'],1e-4)

    def test_quality_and_paired_confidence_intervals_have_expected_direction(self):
        import numpy as np
        from spark.function_experiment import quality_rows,summarize_quality,paired_speedup
        reference=np.asarray([[4.,1.,0.],[0.,1.,4.]])
        rows=quality_rows(reference,reference,[0,2],['article-a','article-b'])
        summary=summarize_quality(rows,samples=50)
        self.assertEqual(summary['gold_accuracy'],1.);self.assertEqual(summary['mean_teacher_kl_nats'],0.)
        estimate=paired_speedup([10,11,12,10,11],[5,5.5,6,5,5.5],samples=50)
        self.assertTrue(estimate['speedup_demonstrated']);self.assertEqual(estimate['median_paired_speedup'],2.)


if __name__=='__main__':unittest.main()

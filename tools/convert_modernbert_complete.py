#!/usr/bin/env python3
"""Convert and validate the complete pinned ModernBERT-base checkpoint on a runner."""
from __future__ import annotations
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def convert(args,report):
    import numpy as np
    import torch
    from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
    from spark.modernbert_model import export_model,NativeModernBert,sha256_file
    from modernbert_teacher import prepare_masked_inputs
    from run_converted_modernbert import encode
    torch.set_num_threads(2)
    report['stage']='download_checkpoint';started=time.perf_counter()
    config=AutoConfig.from_pretrained(args.model,revision=args.revision,token=False,trust_remote_code=False)
    if config.model_type!='modernbert' or not re.fullmatch('[0-9a-f]{40}',args.revision):raise ValueError('requires an immutable ModernBERT revision')
    config.reference_compile=False
    teacher=AutoModelForMaskedLM.from_pretrained(args.model,revision=args.revision,config=config,
        token=False,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager').cpu().eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,token=False,trust_remote_code=False)
    package=args.output/'package';package.mkdir(parents=True,exist_ok=True)
    report['stage']='export_complete_program'
    manifest=export_model(teacher,package,source={'checkpoint':args.model,'revision':args.revision,'pretrained':True})
    tokenizer.save_pretrained(package/'tokenizer')
    (package/'tokenizer/spark_tokenizer.json').write_text(json.dumps({'cls_id':tokenizer.cls_token_id,'sep_id':tokenizer.sep_token_id,'mask_id':tokenizer.mask_token_id,'pad_id':tokenizer.pad_token_id},indent=2)+'\n')
    manifest['tokenizer_sha256']=sha256_file(package/'tokenizer/tokenizer.json');(package/'model.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (package/'spark').mkdir(exist_ok=True);(package/'spark/__init__.py').write_text('"""Converted mathematical inference package."""\n')
    for src,dst in [('spark/modernbert_model.py','spark/modernbert_model.py'),('runtime/spark_modernbert.cpp','runtime/spark_modernbert.cpp'),('tools/run_converted_modernbert.py','tools/run_converted_modernbert.py')]:
        target=package/dst;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/src,target)
    (package/'README.txt').write_text('ModernBERT-base converted mathematical inference package.\n'
        'Source: '+args.model+' at '+args.revision+'\n'
        'Install a CPU BLAS runtime (libblas.so.3 / OpenBLAS), a C++17 compiler, and tokenizers==0.21.4 for text input.\n'
        'Run: OPENBLAS_NUM_THREADS=2 python tools/run_converted_modernbert.py --model . --text "The capital of France is [MASK]."\n'
        'No PyTorch or Transformers import is used for inference. Original dimensions and dense cost remain.\n'
        'This is a complete inference lowering, not a compact alternative architecture or a trained symbolic distillation.\n')
    report['stage']='compile_and_validate'
    rows=[json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    cases=[]
    for n,row in enumerate(rows):
        ids,pos,_=prepare_masked_inputs(tokenizer,row['text'],128)
        standalone,positions,_=encode(package,row['text'],config.max_position_embeddings)
        if ids!=standalone or positions!=[pos]:raise AssertionError('tokenizer boundary mismatch')
        cases.append({'name':f'prompt{n}','text':row['text'],'ids':ids,'queries':[pos]})
    multiple='The [MASK] of France is [MASK].';ids,qs,_=encode(package,multiple,config.max_position_embeddings)
    cases.append({'name':'multiple_masks','ids':ids,'queries':qs})
    base=cases[0]
    for side in ('left','right'):
        pad=5;ids=([tokenizer.pad_token_id]*pad+base['ids']) if side=='left' else (base['ids']+[tokenizer.pad_token_id]*pad)
        mask=([0]*pad+[1]*len(base['ids'])) if side=='left' else ([1]*len(base['ids'])+[0]*pad)
        cases.append({'name':side+'_padding','ids':ids,'mask':mask,'queries':[q+(pad if side=='left' else 0) for q in base['queries']]})
    words=tokenizer(' ordinary words about the river and a city',add_special_tokens=False)['input_ids']
    for length in (131,259):
        ids=[tokenizer.cls_token_id]+(words*((length//len(words))+1))[:length-2]+[tokenizer.sep_token_id]
        at=length//2;ids[at]=tokenizer.mask_token_id
        cases.append({'name':f'local_global_window_{length}','ids':ids,'queries':[at]})
    c=cases[-1];cases.append({**c,'name':'maximum_rotary_position','positions':list(range(config.max_position_embeddings-len(c['ids']),config.max_position_embeddings))})
    comparisons=[];top1=top5=0;logit_values=0;max_logit=max_hidden=0.;allclose=True
    absolute_tolerance=.002;relative_tolerance=.0001
    with NativeModernBert(package) as native:
        for case in cases:
            ids=case['ids'];mask=case.get('mask',[1]*len(ids));positions=case.get('positions',list(range(len(ids))))
            before=time.perf_counter()
            with torch.inference_mode():
                ref=teacher(input_ids=torch.tensor([ids]),attention_mask=torch.tensor([mask]),position_ids=torch.tensor([positions]),output_hidden_states=True)
                expected=ref.logits[0,case['queries']].float().numpy()
                expected_hidden=teacher.model.final_norm(ref.hidden_states[-1])[0].float().numpy()
            teacher_ms=(time.perf_counter()-before)*1000
            before=time.perf_counter();actual=native.run(ids,attention_mask=mask,position_ids=positions,query_positions=case['queries'],return_hidden=True);native_ms=(time.perf_counter()-before)*1000
            values=np.ctypeslib.as_array(actual['logits']).reshape(len(case['queries']),native.vocab)
            hidden=np.ctypeslib.as_array(actual['hidden']).reshape(len(ids),native.hidden)
            valid=np.asarray(mask,dtype=bool)
            le=float(np.max(np.abs(values-expected)));he=float(np.max(np.abs(hidden[valid]-expected_hidden[valid])))
            close=np.allclose(values,expected,atol=absolute_tolerance,rtol=relative_tolerance) and np.allclose(hidden[valid],expected_hidden[valid],atol=absolute_tolerance,rtol=relative_tolerance)
            native_ids=np.argmax(values,axis=1);teacher_ids=np.argmax(expected,axis=1)
            hits=int(np.sum(native_ids==teacher_ids));top1+=hits;top5+=sum(set(np.argsort(a)[-5:])==set(np.argsort(b)[-5:]) for a,b in zip(values,expected))
            logit_values+=values.size;max_logit=max(max_logit,le);max_hidden=max(max_hidden,he);allclose=allclose and close
            predictions=[{'position':position,'token':int(tok),'text':tokenizer.decode([int(tok)])} for position,tok in zip(case['queries'],native_ids)]
            difference=np.abs(hidden-expected_hidden);worst=np.unravel_index(np.argmax(np.where(valid[:,None],difference,-1)),difference.shape)
            comparisons.append({'case':case['name'],'sequence_length':len(ids),'queries':len(case['queries']),'max_logit_error':le,'max_hidden_error':he,
                'hidden_relative_l2':float(np.linalg.norm((hidden-expected_hidden)[valid])/max(np.linalg.norm(expected_hidden[valid]),1e-20)),
                'worst_hidden':{'position':int(worst[0]),'channel':int(worst[1]),'native':float(hidden[worst]),'reference':float(expected_hidden[worst])},
                'allclose':bool(close),'top1_correct':hits,'teacher_ms':teacher_ms,'native_ms':native_ms,'predictions':predictions})
            print(json.dumps(comparisons[-1]),flush=True)
        failed=next((c for c,r in zip(cases,comparisons) if not r['allclose']),None)
        if failed is not None:
            ids=failed['ids'];mask=failed.get('mask',[1]*len(ids));positions=failed.get('positions',list(range(len(ids))))
            with torch.inference_mode():ref=teacher(input_ids=torch.tensor([ids]),attention_mask=torch.tensor([mask]),position_ids=torch.tensor([positions]),output_hidden_states=True)
            result=native.run(ids,attention_mask=mask,position_ids=positions,query_positions=failed['queries'],return_hidden=True,trace=True)
            states=np.ctypeslib.as_array(result['trace']).reshape(2*native.layers+2,len(ids),native.hidden)
            layer_errors=[]
            for index,expected_state in enumerate(ref.hidden_states):
                actual_state=states[0 if index==0 else 2*index]
                wanted=expected_state[0].numpy()
                layer_errors.append({'layer':index,'max_abs':float(np.max(np.abs(actual_state-wanted))),
                    'relative_l2':float(np.linalg.norm(actual_state-wanted)/max(np.linalg.norm(wanted),1e-20))})
            with torch.inference_mode():
                higher=copy.deepcopy(teacher.model).double()
                reference64=higher(input_ids=torch.tensor([ids]),attention_mask=torch.tensor([mask]),position_ids=torch.tensor([positions])).last_hidden_state[0].numpy()
                reference32=teacher.model.final_norm(ref.hidden_states[-1])[0].numpy()
            del higher
            actual_hidden=np.ctypeslib.as_array(result['hidden']).reshape(len(ids),native.hidden)
            report['hidden_diagnostics']={'case':failed['name'],'stages':layer_errors,
                'torch32_vs_torch64_max_abs':float(np.max(np.abs(reference32-reference64))),
                'native32_vs_torch64_max_abs':float(np.max(np.abs(actual_hidden-reference64))),
                'torch32_vs_torch64_relative_l2':float(np.linalg.norm(reference32-reference64)/max(np.linalg.norm(reference64),1e-20))}
        # Batch masking is checked against a padded two-sequence teacher batch;
        # the mathematical runtime executes independent batch rows explicitly.
        pair=[cases[0],cases[-2]];length=max(len(c['ids']) for c in pair)
        batch_ids=[c['ids']+[tokenizer.pad_token_id]*(length-len(c['ids'])) for c in pair]
        masks=[[1]*len(c['ids'])+[0]*(length-len(c['ids'])) for c in pair]
        with torch.inference_mode():reference=teacher(input_ids=torch.tensor(batch_ids),attention_mask=torch.tensor(masks)).logits
        batch_error=0.
        for row,case in enumerate(pair):
            result=native.run(batch_ids[row],attention_mask=masks[row],query_positions=case['queries'])
            value=np.ctypeslib.as_array(result['logits']).reshape(len(case['queries']),native.vocab);expected=reference[row,case['queries']].numpy()
            batch_error=max(batch_error,float(np.max(np.abs(value-expected))))
            if not np.allclose(value,expected,atol=absolute_tolerance,rtol=relative_tolerance):raise AssertionError('padded batch validation failed')
        # Live function data mutations require no rebuild, and never alter the
        # mapped checkpoint file (MAP_PRIVATE); restoring data restores outputs.
        probe=cases[0];kw={'query_positions':probe['queries']};baseline=np.ctypeslib.as_array(native.run(probe['ids'],**kw)['logits']).copy()
        before_hash=sha256_file(package/'weights.sparkmb')
        native.set_active_width(0,max(1,native.intermediate//2));reduced=np.ctypeslib.as_array(native.run(probe['ids'],**kw)['logits']).copy()
        native.set_active_width(0,native.intermediate);restored=np.ctypeslib.as_array(native.run(probe['ids'],**kw)['logits']).copy()
        original_bias=0. if teacher.model.layers[0].mlp.Wo.bias is None else float(teacher.model.layers[0].mlp.Wo.bias[0])
        native.set_function_coefficient(0,'output_bias',0,original_bias+.1);updated=np.ctypeslib.as_array(native.run(probe['ids'],**kw)['logits']).copy()
        native.set_function_coefficient(0,'output_bias',0,original_bias);reset=np.ctypeslib.as_array(native.run(probe['ids'],**kw)['logits']).copy()
        live={'width_changed_output':bool(np.max(np.abs(reduced-baseline))>0),'width_restore_exact':bool(np.array_equal(restored,baseline)),
            'coefficient_changed_output':bool(np.max(np.abs(updated-baseline))>0),'coefficient_restore_exact':bool(np.array_equal(reset,baseline)),
            'checkpoint_unchanged':sha256_file(package/'weights.sparkmb')==before_hash}
    report['stage']='independent_package_inference'
    environment=os.environ.copy();environment['OPENBLAS_NUM_THREADS']='2'
    independent=subprocess.run([sys.executable,str(package/'tools/run_converted_modernbert.py'),'--model',str(package),
        '--text',rows[0]['text']],capture_output=True,text=True,check=True,env=environment)
    isolated=json.loads(independent.stdout)
    if any(isolated['framework_imports'].values()):raise AssertionError('converted inference imported a teacher framework')
    queries=sum(c['queries'].__len__() for c in cases)
    report.update(status='completed' if allclose and top1==queries and all(live.values()) else 'failed',stage='complete',
        checkpoint=args.model,revision=args.revision,parameter_count=sum(p.numel() for p in teacher.parameters()),
        layers=config.num_hidden_layers,program_nodes=len(manifest['program']),payload_bytes=manifest['parameter_payload_bytes'],
        weights_sha256=manifest['weights_sha256'],test_cases=len(cases),masked_queries=queries,logit_values_checked=logit_values,
        top1_agreement=top1,top5_set_agreement=int(top5),allclose=bool(allclose),max_logit_error=max_logit,max_hidden_error=max_hidden,
        absolute_tolerance=absolute_tolerance,relative_tolerance=relative_tolerance,padded_batch_max_logit_error=batch_error,
        live_function_checks=live,independent_inference=isolated,comparisons=comparisons,elapsed_seconds=time.perf_counter()-started,
        scope='Complete ModernBERT-base MLM inference lowering; original dimensions, attention and dense arithmetic retained.',
        unsupported='Training/autograd, custom RoPE scaling, arbitrary architectures, compact full-model replacement.')
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (package/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='completed':raise AssertionError('full conversion parity gate failed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='answerdotai/ModernBERT-base');p.add_argument('--revision',default='8949b909ec900327062f0ebf497f51aef5e6f0c8')
    p.add_argument('--inputs',type=Path,default=ROOT/'examples/modernbert_masks.jsonl');p.add_argument('--output',type=Path,default=ROOT/'artifacts/modernbert-complete')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report={'status':'started','stage':'initialization','workflow_url':f"https://github.com/{os.environ.get('GITHUB_REPOSITORY','drietsch/spark')}/actions/runs/{os.environ.get('GITHUB_RUN_ID','local')}"}
    try:convert(a,report)
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=re.sub(r'https?://\S+','<url>',str(e))[:1500]);raise SystemExit(1)
    finally:
        (a.output/'runner-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

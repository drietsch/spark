#!/usr/bin/env python3
"""Audited comparison: majority, SPARK signature index, sparse supervised postings."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.associative_classifier import SparseAssoc
from spark.modernbert import read_traces, load_index
from spark.semantic import SemanticEncoder
from spark.benchmark import audit_traces, teacher_top1, wilson_95_interval

PARAMS = [{"smoothing":s,"prior_weight":p,"exponent":e}
          for s in (2.0,10.0) for p in (0.0,0.1,0.4,1.0) for e in (0.6,1.0)]


def inner_split(rows):
    train, tune = [], []
    for r in rows:
        group = r.get("source_group") or r.get("source_text") or repr(r["input_ids"])
        bucket = int.from_bytes(hashlib.sha256(("cv:"+group).encode()).digest()[:4],"big") % 10
        (tune if bucket == 0 else train).append(r)
    if not train or not tune:
        raise ValueError("insufficient inner training/holdout examples")
    return train, tune


def compare(rows, predictions):
    truth=[teacher_top1(r) for r in rows]
    hits=sum(a==b for a,b in zip(truth,predictions))
    return {"examples":len(rows),"correct":hits,"agreement":hits/len(rows),
            "wilson95":wilson_95_interval(hits,len(rows))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces",type=Path)
    ap.add_argument("--student-prefix",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--require-boundaries",action="store_true")
    args=ap.parse_args()
    rows=read_traces(args.traces)
    audit_traces(rows)
    train=[r for r in rows if r["split"]=="train"]
    validation=[r for r in rows if r["split"]=="validation"]
    bad=[r for r in rows if (r["input_ids"][0]!=50281 or
         r["input_ids"][-1]!=50282 or r["input_ids"][r["mask_index"]]!=50284)]
    if args.require_boundaries and bad:
        ap.error(f"{len(bad)} traces lack required ModernBERT CLS/SEP/MASK structure")
    special={50281,50282,50283,50284}
    special_count=sum(teacher_top1(r) in special for r in validation)
    inner,tune=inner_split(train)
    model=SparseAssoc.fit(inner)
    best=None
    for params in PARAMS:
        predictions=[model.predict(r["input_ids"],r["mask_index"],**params) for r in tune]
        correct=sum(p==teacher_top1(r) for p,r in zip(predictions,tune))
        candidate=(correct,-params["prior_weight"],-params["smoothing"])
        if best is None or candidate>best[0]:
            best=(candidate,params)
    model=SparseAssoc.fit(train)
    predictions=[model.predict(r["input_ids"],r["mask_index"],**best[1]) for r in validation]
    majority=Counter(teacher_top1(r) for r in train).most_common(1)[0][0]
    report={
      "format":"spark-m6-supervised-v1","training":len(train),"validation":len(validation),
      "invalid_teacher_boundaries":len(bad),
      "teacher_special_top1_fraction":special_count/len(validation),
      "majority":compare(validation,[majority]*len(validation)),
      "sparse_supervised":compare(validation,predictions),
      "train_only_tuning":{"inner_train":len(inner),"inner_val":len(tune),"params":best[1],
                           "inner_val_hits":best[0][0]},
      "features":len(model.postings),
      "label_postings":sum(map(len,model.postings.values())),
      "notes":["Masked-LM teacher agreement, not ground-truth accuracy.",
               "Hyperparameters are selected on inner TRAIN examples only.",
               "Inference consists of sparse postings and scalar votes without matrix operations."]}
    if args.student_prefix:
        prefix=args.student_prefix
        enc=SemanticEncoder.load(Path(str(prefix)+".encoder.json"))
        idx=load_index(Path(str(prefix)+".smlm"),encoder=enc)
        report["spark_256bit"]=compare(validation,[
          idx.predict(r["input_ids"],r["mask_index"])["token"] for r in validation])
    rendered=json.dumps(report,indent=2,sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)


if __name__=="__main__":
    main()

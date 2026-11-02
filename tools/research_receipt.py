#!/usr/bin/env python3
"""Commit small, job-specific research receipts without model coefficients."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--slug',required=True)
    parser.add_argument('--stage',required=True);parser.add_argument('--final',action='store_true');args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True);path=args.root/'runner-report.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    report.update(stage=args.stage,workflow_url=f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    statuses={}
    for phase in ('build','equivalence','quality','benchmark','summary','prepare','evaluate'):
        p=args.root/(phase+'-report.json')
        if p.exists():statuses[phase]=json.loads(p.read_text()).get('status')
    report.update(status='failed' if 'failed' in statuses.values() else 'running',phase_statuses=statuses)
    if args.final:
        report.update(artifact_url=os.environ.get('ARTIFACT_URL',''),report_artifact_url=os.environ.get('REPORT_ARTIFACT_URL',''))
        report['status']='completed' if statuses.get('summary')=='completed' and all(s=='completed' for s in statuses.values()) and report['artifact_url'] else 'failed'
        summary=args.root/'summary-report.json'
        if summary.exists():report['conclusion']=json.loads(summary.read_text()).get('conclusion')
    path.write_text(json.dumps(report,indent=2)+'\n')
    target=args.root/'candidate/program.json'
    if target.exists():shutil.copyfile(target,args.root/'program.json')
    whitelist=('runner-report.json','protocol.json','program.json','audit.json','selection.json','dataset.json',
               'build-report.json','equivalence-report.json','quality-report.json','benchmark-report.json','summary-report.json','prepare-report.json','evaluate-report.json')
    files=[str(args.root/name) for name in whitelist if (args.root/name).exists()]
    subprocess.run(['git','add','-f','--',*files],check=True)
    subprocess.run(['git','commit','--allow-empty','-m','Return '+args.slug+': '+args.stage],check=True)
    branch=f"{os.environ['GITHUB_REF_NAME']}-{args.slug}-results-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],check=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Return only small experiment decisions, receipts and the target manifest."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True);parser.add_argument('--final',action='store_true')
    args=parser.parse_args();root=Path('artifacts/function-simplification');root.mkdir(parents=True,exist_ok=True)
    report_path=root/'runner-report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update(stage=args.stage,workflow_url=f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    phases={}
    for phase in ('prepare','evaluate','benchmark','summarize'):
        path=root/(phase+'-report.json')
        if path.exists():phases[phase]=json.loads(path.read_text()).get('status')
    report['phase_statuses']=phases
    report['status']='failed' if 'failed' in phases.values() else 'running'
    if args.final:
        artifact=os.environ.get('ARTIFACT_URL','');report['artifact_url']=artifact
        report['report_artifact_url']=os.environ.get('REPORT_ARTIFACT_URL','')
        report['status']='completed' if len(phases)==4 and all(v=='completed' for v in phases.values()) and artifact else 'failed'
        summary=root/'summarize-report.json'
        if summary.exists():report['conclusion']=json.loads(summary.read_text()).get('conclusion')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    files=['runner-report.json','protocol.json','dataset.json','selection.json','prepare-report.json','evaluate-report.json','benchmark-report.json','summarize-report.json']
    if (root/'recovered-program/program.json').exists():
        shutil.copyfile(root/'recovered-program/program.json',root/'program.json');files.append('program.json')
    paths=[str(root/name) for name in files if (root/name).exists()]
    subprocess.run(['git','add','-f','--',*paths],check=True)
    subprocess.run(['git','commit','--allow-empty','-m','Return function experiment: '+args.stage],check=True)
    ref=f"{os.environ['GITHUB_REF_NAME']}-results-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+ref],check=True)


if __name__=='__main__':main()

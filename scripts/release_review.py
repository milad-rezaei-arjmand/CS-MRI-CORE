"""Audit and stage only this explicit public-file allowlist. Never publishes."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from release_paths import REPOSITORY

FILES = [
 '.gitattributes','.gitignore','README.md','LICENSE','THIRD_PARTY_NOTICES.md','CITATION.cff',
 'requirements.txt','requirements-test.txt','PUBLIC_FILE_ALLOWLIST.txt','RELEASE_MANIFEST.sha256',
 'assets/Figure_1_CS_MRI_CORE.svg','assets/Figure_1_CS_MRI_CORE.png','assets/Figure_3_locked_repeat_results.png',
 'configs/locked_protocol.json','configs/upstream_required_files.json',
 'docs/DATA_AND_REPRODUCTION.md','docs/PORTABILITY.md','docs/VALIDATION.md','docs/source_provenance.json',
 'results/published_aggregate.json','results/published_per_repeat.csv',
 'scripts/release_paths.py','scripts/run.py','scripts/check_upstream.py','scripts/audit_portability.py',
 'scripts/release_review.py','scripts/train_baseline_repeat00.py','scripts/train_mean_fusion_repeat00.py',
 'scripts/crossslice_dataset.py','scripts/build_mricore_cache.py','scripts/build_final_holdout_cache_seed2026.py',
 'scripts/evaluate_final_holdout_repeat00_04.py','tests/test_release.py',
]

PATTERNS = {
 'github_token':re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
 'private_key':re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
 'aws_access_key':re.compile(rb'AKIA[0-9A-Z]{16}'),
 'real_case_identifier':re.compile(rb'BraTS-MEN-RT-\d{4}-\d'),
 'private_windows_path':re.compile(rb'[A-Za-z]:[\\/]Users[\\/]|/mnt/[a-z]/python files|D:[\\/]python files',re.I),
 'assigned_secret_literal':re.compile(rb'(?:api_key|access_token|password|secret_key)\s*[:=]\s*[\x22\x27][A-Za-z0-9_\-/+]{12,}',re.I),
}

def git(*args,check=True,input=None):
    return subprocess.run(['git','-C',str(REPOSITORY),*args],capture_output=True,text=True,check=check,input=input)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',action='store_true',help='Initialize/stage the explicit reviewed files locally. No commit, remote, or publication.')
    args=parser.parse_args()
    paths=sorted(FILES)
    (REPOSITORY/'PUBLIC_FILE_ALLOWLIST.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8',newline='\n')
    for name in paths:
        path=REPOSITORY/name
        if path.is_file() and path.suffix!='.png':
            data=path.read_bytes()
            normalized=data.replace(b'\r\n',b'\n')
            if data!=normalized: path.write_bytes(normalized)
    provenance=REPOSITORY/'docs/source_provenance.json'
    records=json.loads(provenance.read_text())
    for row in records:
        row['release_sha256']=hashlib.sha256((REPOSITORY/row['release_path']).read_bytes()).hexdigest()
    provenance.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8',newline='\n')
    manifest=[]
    for name in paths:
        if name!='RELEASE_MANIFEST.sha256':
            manifest.append(hashlib.sha256((REPOSITORY/name).read_bytes()).hexdigest()+'  '+name)
    (REPOSITORY/'RELEASE_MANIFEST.sha256').write_text('\n'.join(manifest)+'\n',encoding='utf-8',newline='\n')
    findings=[]
    sizes={}
    for name in paths:
        file=REPOSITORY/name
        if file.is_symlink() or file.is_junction(): findings.append({'file':name,'check':'link'})
        data=file.read_bytes(); sizes[name]=len(data)
        if len(data)>2*1024*1024: findings.append({'file':name,'check':'unexpected_large_file'})
        if file.suffix!='.png':
            for label,pattern in PATTERNS.items():
                if pattern.search(data): findings.append({'file':name,'check':label})
    # Verify the exact allowed scientific assets rather than permitting arbitrary PNGs.
    expected_assets={
     'assets/Figure_1_CS_MRI_CORE.png':'cedd51b314cf7f82749997e7ad71f81a1ce6ee3f4bb28b3a7dbbdc90f89f77ec',
     'assets/Figure_3_locked_repeat_results.png':'e9fd5ebdcca0e7af55f34be80ecad867181d718afe6ed042684e23fdc284297e'}
    for name,expected in expected_assets.items():
        if hashlib.sha256((REPOSITORY/name).read_bytes()).hexdigest()!=expected:
            findings.append({'file':name,'check':'unapproved_image_hash'})
    report={'allowlisted_files':paths,'file_count':len(paths),'total_bytes':sum(sizes.values()),
            'sizes':sizes,'security_scan_findings':findings,'secret_values_emitted':False,
            'medical_data_weights_case_lists_and_experiment_archives_allowlisted':False,
            'scan_limit':'Pattern scan, explicit file review, and asset hashes; not a guarantee against all possible secrets.'}
    if findings:
        print(json.dumps(report,indent=2)); raise SystemExit(1)
    if args.stage:
        if not (REPOSITORY/'.git').exists(): git('init','-b','main')
        # No wildcard or git add -A. Git reads only these exact repository-relative paths.
        git('add','--pathspec-from-file=PUBLIC_FILE_ALLOWLIST.txt')
        tracked=git('ls-files').stdout.splitlines()
        if sorted(tracked)!=paths: raise RuntimeError('Staged paths differ from allowlist')
        git('diff','--cached','--check')
        denied=['.runtime/patient.nii.gz','.runtime/test.png','.external/weights.pth','.env','datasets/test.nii.gz','experiments/test.json','checkpoints/test.pt','.venv-test/test.py']
        ignored=git('check-ignore','--stdin','-z',input='\0'.join(denied)+'\0').stdout.rstrip('\0').split('\0')
        if sorted(ignored)!=sorted(denied): raise RuntimeError('Gitignore exclusion check failed')
        report['staged_paths_match_allowlist']=True
        report['gitignore_prohibited_path_probes_passed']=True
        report['staged_diff_stat']=git('diff','--cached','--stat').stdout
        report['git_status']=git('status','--short').stdout
        report['remotes']=git('remote','-v').stdout
        report['commit_count']=git('rev-list','--count','HEAD',check=False).stdout.strip() or '0 (unborn HEAD)'
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()

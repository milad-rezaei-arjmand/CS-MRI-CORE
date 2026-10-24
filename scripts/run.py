"""Portable launcher. Without --execute it prints a plan and does no research work."""
import argparse
import json
import os
import runpy
import sys
from pathlib import Path

from release_paths import REPOSITORY, release_workspace, release_upstream

def plan(args):
    scripts = {
        'train': 'train_baseline_repeat00.py' if args.method == 'single' else 'train_mean_fusion_repeat00.py',
        'preprocess': 'build_mricore_cache.py',
        'preprocess-final': 'build_final_holdout_cache_seed2026.py',
        'prepare-evaluation': 'evaluate_final_holdout_repeat00_04.py',
        'evaluate': 'evaluate_final_holdout_repeat00_04.py',
    }
    return {'action': args.action, 'method': args.method, 'repeat': args.repeat,
            'script': scripts[args.action], 'workspace': str(release_workspace()),
            'upstream': str(release_upstream()), 'execute': args.execute,
            'evaluation_scope': 'paired Repeat00-04, locked 50-case protocol' if 'eval' in args.action else None}

def preflight(args):
    root = release_workspace()
    if args.action == 'train':
        folder = root/'datasets/BraTS-MEN-RT/processed_mricore/lists'
        sets = []
        for role in ('train','val'):
            path = folder/f'repeat_{args.repeat:02d}_{role}.txt'
            rows = [r.strip() for r in path.read_text(encoding='utf-8-sig').splitlines() if r.strip()]
            if len(rows) != 5 or len(set(rows)) != 5:
                raise RuntimeError(f'{role} must contain exactly five distinct center records')
            # Check case separation without printing identifiers.
            sets.append({r.split(',')[0].rsplit('-z',1)[0] for r in rows})
        if sets[0] & sets[1]:
            raise RuntimeError('Training and validation cases overlap')
    if args.action in ('train','evaluate'):
        upstream = release_upstream()
        for relative in ('cfg.py','models/sam/__init__.py','utils/dataset.py',
                         'pretrained_weights/mri_foundation.pth','pretrained_weights/sam_vit_b_01ec64.pth'):
            if not (upstream/relative).is_file():
                raise FileNotFoundError(f'Obtain the upstream dependency/weights separately: {relative}')
        from check_upstream import verify
        errors = verify(upstream)
        if errors:
            raise RuntimeError('Upstream file verification failed: ' + ', '.join(errors))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train','preprocess','preprocess-final','prepare-evaluation','evaluate'])
    parser.add_argument('--method', choices=['single','cross'], default='cross', help='Training method; evaluation always covers both methods.')
    parser.add_argument('--repeat', type=int, choices=range(5), default=0, help='Training support repeat; evaluation always covers 00-04.')
    parser.add_argument('--execute', action='store_true', help='Actually process local authorized data or train/evaluate. Omit to inspect the plan.')
    args = parser.parse_args()
    result = plan(args)
    print(json.dumps(result, indent=2))
    if not args.execute:
        return
    preflight(args)
    os.environ['CS_MRI_CORE_REPEAT'] = str(args.repeat)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    flags = {'prepare-evaluation':['--prepare-only'],'evaluate':['--evaluate']}.get(args.action,[])
    script = REPOSITORY/'scripts'/result['script']
    # Upstream cfg.parse_args must not see launcher options.
    sys.argv = [str(script)] + flags
    runpy.run_path(str(script), run_name='__main__')

if __name__ == '__main__':
    main()

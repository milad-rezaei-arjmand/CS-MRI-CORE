"""Read-only verification of separately acquired MRI-CORE source files."""
import argparse
import hashlib
import json
import concurrent.futures
import urllib.request
from pathlib import Path
from release_paths import REPOSITORY, release_upstream

def verify(root):
    spec = json.loads((REPOSITORY/'configs/upstream_required_files.json').read_text())
    errors = []
    for name, expected in spec['files'].items():
        path = Path(root)/name
        if not path.is_file():
            errors.append(name + ' (missing)')
        else:
            # Git autocrlf may change checkout bytes. Recorded research files use LF.
            data = path.read_bytes().replace(b'\r\n', b'\n')
            if hashlib.sha256(data).hexdigest() != expected:
                errors.append(name + ' (hash mismatch)')
    return errors

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=release_upstream())
    parser.add_argument('--remote',action='store_true',help='Read and hash the pinned remote source in memory; no files are downloaded to disk.')
    args = parser.parse_args()
    if args.remote:
        spec = json.loads((REPOSITORY/'configs/upstream_required_files.json').read_text())
        def remote_check(item):
            name,expected = item
            url = f'https://raw.githubusercontent.com/mazurowski-lab/mri_foundation/{spec["inspected_commit"]}/{name}'
            try:
                request = urllib.request.Request(url,headers={'User-Agent':'CS-MRI-CORE-read-only-audit'})
                with urllib.request.urlopen(request,timeout=40) as stream:
                    data = stream.read().replace(b'\r\n',b'\n')
                return None if hashlib.sha256(data).hexdigest()==expected else name+' (hash mismatch)'
            except Exception as error:
                return name+' ('+type(error).__name__+')'
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            errors = [value for value in pool.map(remote_check,spec['files'].items()) if value]
    else:
        errors = verify(args.root)
    print(json.dumps({'matched':not errors,'errors':errors},indent=2))
    raise SystemExit(bool(errors))

if __name__=='__main__': main()

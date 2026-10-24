"""Private-author audit: compare copies to the read-only research source.

The original source root is a CLI input and is never written. No source modules
are imported. Output goes to stdout only. Not needed for normal reproduction.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from release_paths import REPOSITORY

ALLOWED_PATHS = {'PROJECT','MRI_CORE','SCRIPT_DIR','TRAIN_LIST','VAL_LIST','RUN_DIR'}

class Normalize(ast.NodeTransformer):
    def visit_Name(self,node):
        if isinstance(node.ctx,ast.Load) and node.id=='RELEASE_REPEAT':
            return ast.copy_location(ast.Constant(value=0),node)
        return node

    def visit_Assign(self,node):
        if any(isinstance(target,ast.Name) and target.id in ALLOWED_PATHS|{'RELEASE_REPEAT'} for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Import(self,node): return None
    def visit_ImportFrom(self,node): return None

    def visit_Assert(self,node):
        if any(isinstance(value,ast.Name) and value.id=='RELEASE_REPEAT' for value in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

def normal(path):
    tree = Normalize().visit(ast.parse(path.read_text(encoding='utf-8-sig')))
    return ast.dump(tree,include_attributes=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',required=True,type=Path)
    args = parser.parse_args()
    records = json.loads((REPOSITORY/'docs/source_provenance.json').read_text())
    report = []
    for row in records:
        original = args.archive_root/row['archive_relative_path']
        release = REPOSITORY/row['release_path']
        report.append({'file':row['release_path'],'original_hash_matches_provenance':hashlib.sha256(original.read_bytes()).hexdigest()==row['archive_sha256'],
                       'computational_ast_unchanged_except_declared_paths_and_repeat_metadata':normal(original)==normal(release)})
    print(json.dumps(report,indent=2))
    if not all(row['original_hash_matches_provenance'] and row['computational_ast_unchanged_except_declared_paths_and_repeat_metadata'] for row in report):
        raise SystemExit(1)

if __name__=='__main__': main()

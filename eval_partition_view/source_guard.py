"""Check audited Baseline functions without importing CUDA/BART modules."""

import ast
import hashlib
import json
from pathlib import Path


def check_sources(root):
    audit = json.loads(Path(__file__).with_name('source_audit.json').read_text(encoding='utf-8'))
    trees = {}
    local_sha256 = {}
    for relative in audit['reviewed_blob_shas']:
        path = root / relative
        raw = path.read_bytes()
        local_sha256[relative] = hashlib.sha256(raw).hexdigest()
        if path.suffix == '.py':
            trees[relative] = ast.parse(raw.decode('utf-8'), filename=str(path))
    for entry in audit['function_signatures']:
        nodes = trees[entry['path']].body
        try:
            for part in entry['function'].split('.'):
                node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                            and n.name == part)
                nodes = node.body
        except StopIteration as exc:
            raise RuntimeError(f"Missing audited function {entry['path']}::{entry['function']}") from exc
        canonical = ast.dump(node, include_attributes=False).replace(', type_params=[]', '')
        actual = hashlib.sha256(canonical.encode()).hexdigest()
        if actual != entry['ast_sha256']:
            raise RuntimeError(f"Audited Baseline implementation changed: {entry['path']}::{entry['function']}. "
                               'Review the implementation before using this capture program.')
    return {'reviewed_repository': audit['reviewed_repository'],
            'reviewed_branch': audit['reviewed_branch'], 'reviewed_commit': audit['reviewed_commit'],
            'reviewed_blob_shas': audit['reviewed_blob_shas'], 'local_file_sha256': local_sha256,
            'audited_function_count': len(audit['function_signatures']),
            'audited_functions_match': True,
            'note': 'Function ASTs match the reviewed implementation. No Git command was executed.'}

"""Run an existing PaDIS eval entry point with NGS partition support only in this process.
Example: python eval/run_ngs_eval.py eval/recon.py [original arguments]
"""
import os
import sys
import runpy
from pathlib import Path

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python eval/run_ngs_eval.py eval/<entry>.py [args]')
    target = Path(sys.argv[1]).resolve()
    if not target.is_file() or target == Path(__file__).resolve():
        raise SystemExit('Choose an existing evaluation entry point')
    sys.path[:0] = [str(target.parent), str(root), str(root/'train/padis-mri'), str(root/'eval')]
    import denoise_padding as legacy
    from denoise_padding_ngs import denoisedFromPatches as ngs_denoise
    original = legacy.denoisedFromPatches
    def dispatch(net, *args, **kwargs):
        if getattr(net, 'use_ngs', False):
            return ngs_denoise(net, *args, **kwargs)
        return original(net, *args, **kwargs)
    legacy.denoisedFromPatches = dispatch
    import importlib
    qualified = importlib.import_module('eval.denoise_padding')
    qualified.denoisedFromPatches = dispatch
    sys.argv = [str(target)] + sys.argv[2:]
    runpy.run_path(str(target), run_name='__main__')

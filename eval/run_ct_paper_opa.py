"""Run existing PaDIS evaluator with the established 1/.003 DPS schedule."""
import sys, runpy, inspect, functools
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'eval'),str(root),str(root/'train/padis-mri')]
import recon
original=recon.dps2
sig=inspect.signature(original)
calls=0
@functools.wraps(original)
def wrapped(*args,**kwargs):
    global calls
    b=sig.bind(*args,**kwargs)
    net=b.arguments['net']
    if not getattr(net,'use_joint_global',False) or net.model.__class__.__name__!='CTDDIMUNet':
        raise TypeError('Expected CT DDIM + EDM OPA checkpoint')
    b.arguments.update(sigma_min=0.003,sigma_max=1.0)
    calls+=1
    print('[CT+OPA DPS] sigma_max=1, sigma_min=0.003',flush=True)
    return original(*b.args,**b.kwargs)
recon.dps2=wrapped
sys.argv[0]=str(root/'eval/run.py')
runpy.run_path(sys.argv[0],run_name='__main__')
assert calls>0,'DPS was not called'

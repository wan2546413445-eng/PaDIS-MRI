"""Run existing PaDIS DPS using a VP prior adapter, without modifying eval sources."""
import sys
import runpy
import inspect
import functools
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'eval'),str(root),str(root/'train/padis-mri')]
import recon
from training.ct_paper_2d import CTPaperPrior

original=recon.dps2
signature=inspect.signature(original)
calls=0

@functools.wraps(original)
def wrapped(*args,**kwargs):
    global calls
    b=signature.bind(*args,**kwargs)
    net=b.arguments['net']
    if not isinstance(net, CTPaperPrior): raise TypeError('Expected CTPaperPrior checkpoint')
    b.arguments['sigma_min']=net.sigma_min
    b.arguments['sigma_max']=1.0
    calls+=1
    print(f'[CTPaper VP->DPS] sigma_min={net.sigma_min:.8f}, sigma_max=1.0; call={calls}',flush=True)
    return original(*b.args,**b.kwargs)

recon.dps2=wrapped
sys.argv[0]=str(root/'eval/run.py')
runpy.run_path(sys.argv[0],run_name='__main__')
assert calls>0, 'DPS was not called'

"""CPU-only grid scheduling regression tests, independent of model/GPU."""
import sys
from pathlib import Path
import random
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval_pula.flex_grids import build_grid_plan, grid_summary
from eval_pula.flex_schedule import make_schedule, counts
rows=make_schedule(47,pattern='PPPPD')
old=build_grid_plan(rows,64,123)
new=build_grid_plan(rows,64,123,pula_mode='cycle_k',dps_mode='random_step')
assert all(a[:4]==b[:4] for a,b in zip(old,new))
assert new==build_grid_plan(rows,64,123,pula_mode='cycle_k',dps_mode='random_step')
rng=random.Random(123)
ref=[(rng.randrange(64),rng.randrange(64)) for _ in range(235)]
assert [r[4] for r in new]==ref[4::5]
unique=build_grid_plan(rows,64,123,pula_mode='cycle_k',dps_mode='random_unique')
assert grid_summary(rows,unique)['D']==dict(updates=47,unique_shifts=47)
assert grid_summary(rows,old)['D']['unique_shifts']==1
assert counts(rows)['logical_updates']==235
for pattern in ['PPPPP','DDDDD','PPDPD','PDPDD']:
    r=[pattern]*47
    for mode in ['cycle_k','fixed','random_step','random_unique']:
        g=build_grid_plan(r,64,123,pula_mode=mode,dps_mode=mode)
        assert all(len(a)==len(b) for a,b in zip(r,g))
        assert all(0<=v<64 for line in g for shift in line for v in shift)
        assert sum(v['updates'] for v in grid_summary(r,g).values())==235
print('PASS legacy:',grid_summary(rows,old))
print('PASS separate:',grid_summary(rows,new))
print('PASS unique:',grid_summary(rows,unique))
print('PASS P shifts unchanged; native DPS global-slot RNG; deterministic; P/D counts unchanged')

"""Pair the closest available EMA snapshots inside the selected experiment runs."""
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path


def snapshot(root, target):
    candidates = list(root.rglob('network-snapshot-*.pkl'))
    if not candidates:
        raise SystemExit(f'No EMA snapshots in {root}')
    def kimg(path):
        return int(path.stem.split('-')[-1])
    delta = min(abs(kimg(p)-target) for p in candidates)
    best = [p for p in candidates if abs(kimg(p)-target) == delta]
    if len({p.parent for p in best}) != 1:
        raise SystemExit(f'Select one run directory under {root}')
    path = min(best, key=kimg)
    return path, kimg(path)


base = Path(os.environ.get('RESULT_ROOT', '/mnt/SSD2/wsy/PaDIS-MRI'))
runs = base / 'PaDIS-MRI-runs'
wdr = snapshot(Path(os.environ.get('WDR_RUN', runs/'WDR/wdr_train_500k')), 500)
pattern = '*symmetric_lam2p5_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform'
opa_root = os.environ.get('OPA_RUN')
if not opa_root:
    matches = list((runs/'overlap_independent_training-runs/brain/32dB').glob(pattern))
    if len(matches) != 1:
        sys.exit('Set OPA_RUN to the verified C128 OPA effective lambda=1.25 baseline directory.')
    opa_root = matches[0]
opa = snapshot(Path(opa_root), wdr[1])
print(f'WDR {wdr}; OPA {opa}; difference {opa[1]-wdr[1]:+d} kimg', flush=True)
out = Path(os.environ.get('OUT_ROOT', base/'PaDIS-MRI-recon/new_data/WDR_paired'))
for mask in (7, 4):
    tables = []
    for name, (model, k) in [('WDR', wdr), ('OPA', opa)]:
        group = out/f'R{mask}'/name/model.parent.name/model.stem
        group.mkdir(parents=True, exist_ok=True)
        marker = group/'model_path.txt'
        if marker.exists() and marker.read_text() != str(model.resolve()):
            sys.exit(f'Output belongs to another model: {group}')
        marker.write_text(str(model.resolve()))
        env = dict(os.environ, MODEL_PATH=str(model.resolve()), METHOD=name, GPU=os.getenv('GPU','0'),
                   CKPT_KIMG=str(k), MASKS=str(mask), OUT_ROOT=str(group.resolve()),
                   PROJECT_DIR=os.getcwd(), SAMPLES='0,4,8,13,18,22,26,31', DPS_STEPS='40', DPS_INNER='6')
        subprocess.run(['bash','bash/eval_paired_sigma1_core.sh'], env=env, check=True)
        with (group/'per_image.csv').open() as f:
            tables.append({r['sample']: r for r in csv.DictReader(f)})
    rows = []
    for i, row in tables[0].items():
        rows.append({key: float(row[key])-float(tables[1][i][key]) for key in row if key != 'sample'})
    with (out/f'R{mask}'/f'paired_WDR{wdr[1]}_OPA{opa[1]}.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', *rows[0]])
        writer.writerows([i,*r.values()] for i,r in zip(tables[0],rows))
        writer.writerow(['mean',*(statistics.mean(r[k] for r in rows) for k in rows[0])])

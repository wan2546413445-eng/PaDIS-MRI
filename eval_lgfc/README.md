# LGFC Stage 1 v2

This directory contains an isolated implementation of Late-Stage Global
Frequency Correction (LGFC) for PaDIS-MRI. The protected baseline files under
`eval/` are not modified.

LGFC constructs either a `last` or EMA temporal reference from patch-denoiser
outputs already produced within the current outer diffusion step. Inner-loop
states and noise vary continuously, so this reference is not a strict
same-state, multi-partition consensus. It is reset at the beginning of every
outer step, remains detached, and is never carried between outer steps.

## Correction and safeguards

At a scheduled late outer step, LGFC encodes the current center FOV and its
temporal reference with the existing coil sensitivity maps. It uses only the
reference-to-state difference at unmeasured frequencies, combines the
multi-coil correction with an unregularized SENSE formula, and updates only the
center FOV of the padded state. It does not set unmeasured frequencies to zero
and does not treat the reference as true unmeasured k-space.

Two safeguards permit a meaningful update strength:

1. A per-sample relative-update limit clips the candidate correction to 5% by
   default.
2. A sampled-data residual guard accepts at most 2% DC-residual growth. Failed
   candidates are retried at scales `0.5`, `0.25`, and `0.125`; if all fail,
   the current image is returned unchanged.

The correction is executed under `torch.no_grad()`. It receives no ground
truth, brain mask, or fully sampled k-space. It adds no denoiser call, network
call, partition, or random draw. When LGFC is disabled, `dps2_lgfc()` directly
dispatches to the original `eval/recon.py::dps2()`.

## Fixed stage-1 experiment

- sample3 only, mask 7, seed 123, zeta 3
- image size 384, padding 64, patch size 64
- 78 outer steps and 10 inner loops, totaling 780 denoiser updates
- correction on outer steps 40 through 78, for at most 39 attempts
- A1: `last`, weight 0.25, relative limit 0.05
- A2: EMA beta 0.8, weight 0.25, relative limit 0.05
- A3: EMA beta 0.8, weight 0.40, relative limit 0.08
- all LGFC runs: DC growth limit 1.02 and at most 3 backtracks

## Run

Set the four required environment variables and execute the script. It resolves
the repository root automatically and adds `train/padis-mri` to `PYTHONPATH` so
checkpoints pickled as `training.*` can be loaded. Set optional `GPU` to choose a
physical device; the script exposes it as logical GPU 0.

```bash
GPU=0 \
MODEL_PATH=/path/to/network-snapshot.pkl \
VAL_DIR=/path/to/validation \
SAVE_ROOT=/path/to/lgfc-stage1-output \
SAMPLE_INDICES=3 \
bash bash/eval_lgfc_stage1.sh
```

The script first validates paths, syntax, imports, and smoke tests. It then runs
B0 with the original entry point, followed by B1 through the disabled LGFC entry
point. It checks reconstruction shape and numerical equality before
allowing A1, A2, and A3 to run. All five runs are serial and have separate
stdout and stderr logs. LGFC diagnostics contain CPU scalars only.

If the desired sample has been copied into a one-file validation directory as
`sample_0.pt`, use `SAMPLE_INDICES=0`.

## Checks

```bash
python -m py_compile eval_lgfc/*.py
bash -n bash/eval_lgfc_stage1.sh
python eval_lgfc/smoke_test_lgfc.py
```

The smoke test covers invalid configurations, direct baseline bypass,
reference formulas and detachment, mask broadcasting, multi-coil complex FFTs,
identity cases, finite outputs, relative-update clipping, all DC-backtracking
outcomes, absence of ground-truth inputs, denoiser-call parity, and CUDA peak
memory reporting when CUDA is available.

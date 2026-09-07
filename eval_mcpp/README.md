# MCPP Stage-1

**Measurement-Calibrated Patch Prior (MCPP)** keeps the pretrained PaDIS backbone frozen and adapts only the final SongUNet `*_aux_norm` + `*_aux_conv` for the current MRI subject.

Baseline PaDIS uses

```text
L = || A Agg[D_theta(P_i x)] - y ||^2
```

and updates only `x`. MCPP uses the same measurement loss and the same denoiser forward graph to obtain both

```text
grad_x   = dL/dx
grad_phi = dL/dphi / (||y||^2 + eps)
```

then performs

```text
x_k   -> original PaDIS posterior update
phi_k -> phi_{k+1} = phi_k - mcpp_lr * grad_phi
```

The updated `phi` is first used by the **next inner loop**. All patches share the same subject-specific `phi`, so the acquired whole-FOV MRI residual calibrates future local patch predictions without adding a full-image diffusion model or global-image condition.

## Important properties

- one subject at a time (`batch size = 1`)
- each subject starts from pretrained `phi0` and resets after reconstruction
- GT / fully sampled k-space are not passed to MCPP adaptation; they are used only by the existing post-evaluation metrics
- one `denoisedFromPatches()` call per inner loop, same as baseline
- no LGFC, LoRA, extra network, gate, regularizer, backtracking, or second partition in Stage-1
- diagnostics are optional; when enabled, GPU scalars are transferred to CPU only once after the sample loop

## Run

The reference code may live outside the repository. Point it to the real PaDIS-MRI root with `REPO_ROOT`:

```bash
REPO_ROOT=/path/to/PaDIS-MRI \
MODEL_PATH=/path/to/network-snapshot.pkl \
VAL_DIR=/path/to/val_t1-flair_subsamp/32dB \
SAVE_DIR=/path/to/mcpp_output \
GPU_ID=0 \
MCPP_LR=1e-4 \
SAMPLE_INDICES=0,4,8,12,16,20,24,28 \
bash bash/eval_mcpp_stage1.sh
```

The shell uses the current formal paired-evaluation policy by default:

```text
seed=123
mask_select=7
zeta=3
steps=78
inner_loops=10
psize=64
pad=64
fixed_seed_per_sample=True
```

For a first single-sample mechanism test, override e.g. `SAMPLE_INDICES=3`.

## Diagnostics

When enabled, each update records only:

```text
global_update
outer_step
inner_step
sigma
relative_measurement_sse
phi_grad_norm
phi_relative_delta
prior_dc_alignment
```

`prior_dc_alignment` is measured on the central real FOV and is diagnostic only; it never gates the update.

## Smoke test

```bash
python eval_mcpp/smoke_test_mcpp.py
```

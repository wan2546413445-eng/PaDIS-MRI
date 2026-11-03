# Simplified LGFC evaluation

This directory keeps the original PaDIS sampler and the current LGFC method in a small, independent implementation. It does not import code from `eval_lgfc/`.

## Where to look

- `run_lgfc.py`: command-line entry, model loading, one-sample evaluation, `config.json`, and final normalized metrics.
- `evaluator_lgfc.py`: reuses `eval/evaluator.py::DPSHyperEvaluator` and replaces only its PaDIS reconstruction entry.
- `recon_lgfc.py`: the PaDIS outer/inner reconstruction loop and the three marked LGFC insertion points.
- `lgfc_core.py`: current-outer reference update and the frequency correction itself.
- `config.py`: the single source of fixed PaDIS settings and LGFC defaults.

## Execution order

For every outer diffusion step, `reference` is reset to `None`. Each of the 10 inner loops first produces a denoised prediction; its center FOV updates a detached `last` or EMA reference. The reference therefore aggregates only predictions from the current outer step and never crosses an outer-step boundary.

After all 10 inner loops finish, LGFC runs when

```text
outer_step >= start_outer
and (outer_step - start_outer) % correction_every == 0
```

The corrected center FOV is then embedded back into the padded state used by the next outer step.

## LGFC update

For state `x`, current-outer reference `r`, sensitivity maps `S`, and sampling mask `M`:

```text
delta_k = (1 - M) [FFT(S r) - FFT(S x)]
delta_x = lambda * sum_c conj(S_c) IFFT(delta_k_c)
                    / (sum_c |S_c|^2 + eps)
```

The update is scaled down only when `||delta_x|| / (||x|| + eps)` exceeds `max_relative_update`. FFT and IFFT use orthonormal normalization, matching the existing MRI operators.

## Defaults

PaDIS is fixed at image size 384, pad/patch size 64, 78 outer steps, 10 inner loops, `zeta=3.0`, seed 123, `sigma=[0.003, 10.0]`, and `rho=7.0`.

LGFC defaults to EMA reference, `ema_beta=0.8`, `start_outer=60`, `correction_every=1`, `unsampled_weight=0.6`, and `max_relative_update=0.08`. This triggers 19 corrections at outer steps 60 through 78.

## Run

GPU selection stays outside Python:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_lgfc_simply/run_lgfc.py \
  --model_path /path/to/network-snapshot.pkl \
  --val_dir /path/to/validation/32dB \
  --save_dir /path/to/output/sample26 \
  --sample_index 26 \
  --mask_select 4 \
  --save_diagnostics
```

## Output

```text
save_dir/
├── config.json
└── evaluate/
    ├── recons/recon_patch_<sample_index>.npy
    ├── comp_plots/results.json
    └── lgfc_diagnostics/<sample_index>_patch_lgfc.csv
        # only with --save_diagnostics
```

`comp_plots/results.json` is the final normalized metric file. The optional CSV contains only the reference/state distance, unmeasured-frequency difference, raw and clipped update norms, clipping flag, and final update norm for each correction.

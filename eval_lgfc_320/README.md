# LGFC 320-ROI inference

This isolated experiment keeps the original 384x384 MRI measurement problem unchanged while reducing the PaDIS target ROI to the central 320x320 region.

## Fixed geometry

Original PaDIS canvas:

- 384 MRI FOV + 64 zero padding per side = 512x512.

LGFC-320 canvas:

- take the central 448x448 coordinate region of the original 512 canvas;
- within 448, `[32:416]` is the complete original 384 MRI FOV;
- within 448, `[64:384]` is the central 320 PaDIS target;
- the outer 32 pixels are zero padding.

The 32-pixel real context ring inside the original 384 FOV is retained for the original MRI likelihood and LGFC but does not receive the learned PaDIS patch prior. Its x0 estimate uses the current posterior state. The central 320 target receives the learned patch denoiser output.

No 384 MRI quantity is regenerated or cropped for data consistency:

- full k-space: unchanged;
- undersampling mask: unchanged;
- sensitivity maps: unchanged;
- measurement: unchanged;
- MRI forward/adjoint operator: unchanged.

Position conditioning is the central `[32:480]` crop of the original 512x512 training coordinate grid, so anatomical positions keep their original PaDIS coordinates.

With psize=64:

- baseline 384 target: 7x7 = 49 patches/update;
- 320 target: 6x6 = 36 patches/update.

## Fair metric comparison

`utils_320.py` always evaluates the same central 320 GT ROI. It accepts either a 320 reconstruction from this experiment or a 384 baseline reconstruction and center-crops the baseline first. The normalization constant is still computed from the original fully sampled 384 k-space.

This directory is intentionally separate from `eval_lgfc/`.

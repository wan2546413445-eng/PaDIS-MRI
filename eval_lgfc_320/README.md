# LGFC 320-ROI inference

This isolated experiment keeps the original 384x384 MRI measurement problem (full k-space, mask, sensitivity maps and data consistency) unchanged, while reducing the PaDIS target ROI to the central 320x320 region.

Implementation principle:
- original measurement FOV remains 384x384;
- PaDIS inference canvas is the central 448x448 crop of the original 512x512 padded canvas;
- the central 384x384 slice of this 448 canvas is used for MRI data consistency and LGFC;
- only the central 320x320 region is the reported reconstruction target;
- positional encoding is cropped from the original 512x512 training coordinate grid instead of being renormalized on 448;
- patch size remains 64, so a fixed partition uses 6x6=36 patches instead of 7x7=49.

This directory is intentionally separate from `eval_lgfc/`.

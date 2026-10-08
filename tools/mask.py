from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Configuration
# ============================================================

SAMPLE_PATH = Path(
    "/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/"
    "val_t1-flair_subsamp/32dB/sample_0.pt"
)

OUT_DIR = Path(
    "/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-matlab/"
    "sampling_mask_visualization"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load validation sample
# ============================================================

data = torch.load(
    SAMPLE_PATH,
    map_location="cpu",
    weights_only=False,
)

print(f"Validation sample: {SAMPLE_PATH}")


# ============================================================
# Export R4 / R7 masks
# ============================================================

for R in [4, 7]:

    key = f"mask_{R}"

    if key not in data:
        raise KeyError(
            f"{key} not found. Available keys: {list(data.keys())}"
        )

    mask = data[key]

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    # (1, 384, 384) -> (384, 384)
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise RuntimeError(
            f"Unexpected {key} shape after squeeze: {mask.shape}"
        )

    # Binary mask for visualization only.
    # Do not alter the original validation data.
    mask_vis = (mask > 0).astype(np.float32)

    output_path = OUT_DIR / f"sampling_mask_R{R}.png"

    plt.imsave(
        output_path,
        mask_vis,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    sampled = int(mask_vis.sum())
    total = int(mask_vis.size)

    # Cartesian column mask: count acquired columns as an extra check.
    acquired_columns = int(
        np.any(mask_vis > 0, axis=0).sum()
    )

    print(
        f"R={R}: "
        f"shape={mask_vis.shape}, "
        f"acquired_columns={acquired_columns}, "
        f"sampled_pixels={sampled}/{total}"
    )

    print(f"Saved: {output_path}")


print("\nDone.")
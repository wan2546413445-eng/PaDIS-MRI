#!/usr/bin/env python3
"""After paired GPU runs: check exact reconstruction and full effective configuration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


# The runner stores all sampler settings in config.json. The ON run alone adds
# these read-only diagnostic options; out_root is excluded by the runner already.
IGNORED_CONFIG_KEYS = frozenset({
    "diagnose_dps_jacobian", "jacobian_levels", "jacobian_inner_steps", "out_root",
})
COUNT_KEYS = (
    "logical_updates", "logical_nfe", "network_forwards", "main_network_forwards",
    "diagnostic_network_forwards", "terminal_network_forwards", "network_backprops",
    "dps_updates", "dps_a_calls", "langevin_updates", "cg_solves", "cg_iterations",
    "a_calls", "ah_calls",
)
_MISSING = object()


def _difference(on, off, ignored=()):
    """Compare every remaining JSON field, including keys absent on one side."""
    mismatches = {}
    for key in sorted((set(on) | set(off)) - set(ignored)):
        a, b = on.get(key, _MISSING), off.get(key, _MISSING)
        if a != b:
            mismatches[key] = {
                "on": "<missing>" if a is _MISSING else a,
                "off": "<missing>" if b is _MISSING else b,
            }
    return mismatches


def compare(on_root, off_root, sample=0):
    def file(root, name):
        return Path(root) / "random_step" / f"sample_{sample}" / "evaluate" / name

    outputs = {}
    for mode, folder in (("on", on_root), ("off", off_root)):
        recon_path = file(folder, f"recons/recon_patch_{sample}.npy")
        arr = np.load(recon_path, allow_pickle=False)
        config = json.loads(file(folder, "config.json").read_text())
        timing = json.loads(file(folder, "timing.json").read_text())
        outputs[mode] = (arr, config, timing, recon_path)

    a, ac, at, ap = outputs["on"]
    b, bc, bt, bp = outputs["off"]
    same_bits = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    max_error = float(np.max(np.abs(a-b))) if a.shape == b.shape else None
    config_diff = _difference(ac, bc, IGNORED_CONFIG_KEYS)
    count_diff = _difference({k: at.get(k, _MISSING) for k in COUNT_KEYS},
                             {k: bt.get(k, _MISSING) for k in COUNT_KEYS})
    # Missing counters on both sides are not an acceptance success.
    for key in COUNT_KEYS:
        if key not in at or key not in bt:
            count_diff[key] = {
                "on": at.get(key, "<missing>"), "off": bt.get(key, "<missing>")
            }

    report = {
        "bitwise_equal": same_bits,
        "max_abs_difference": max_error,
        "config_same": not config_diff,
        "config_keys_compared": len((set(ac) | set(bc)) - IGNORED_CONFIG_KEYS),
        "excluded_config_keys": sorted(IGNORED_CONFIG_KEYS),
        "config_mismatches": config_diff,
        "network_and_cg_counts_same": not count_diff,
        "count_mismatches": count_diff,
        "on_recon_sha256": hashlib.sha256(ap.read_bytes()).hexdigest(),
        "off_recon_sha256": hashlib.sha256(bp.read_bytes()).hexdigest(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not (same_bits and not config_diff and not count_diff):
        raise SystemExit("Phase A acceptance FAILED: inspect the paired runs.")
    print("Phase A acceptance PASSED")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-root", required=True)
    parser.add_argument("--off-root", required=True)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    compare(args.on_root, args.off_root, args.sample)

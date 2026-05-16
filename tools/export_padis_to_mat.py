import os
import argparse
import glob
import numpy as np
import torch
from scipy.io import savemat


def fftmod_np(x: np.ndarray) -> np.ndarray:
    """
    Same logic as eval/utils.py fftmod, but for numpy.
    Input shape can be [Nc, H, W].
    """
    y = x.copy()
    y[..., ::2, :] *= -1
    y[..., :, ::2] *= -1
    return y


def make_zero_filled_adjoint(ksp: np.ndarray, s_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Reproduce the adjoint / zero-filled visualization approximately:
        adjoint = sum_c conj(s_map_c) * ifft2(mask * ksp_c)

    ksp:   [Nc, H, W], complex
    s_map: [Nc, H, W], complex
    mask:  [1, H, W] or [H, W]
    return: [H, W], complex
    """
    if mask.ndim == 3:
        mask2d = mask[0]
    else:
        mask2d = mask

    ksp_shifted = fftmod_np(ksp)
    us_ksp = ksp_shifted * mask2d[None, :, :]

    coil_imgs = np.fft.ifft2(us_ksp, axes=(-2, -1), norm="ortho")
    adj = np.sum(np.conj(s_map) * coil_imgs, axis=0)
    return adj


def find_recon_file(recon_dir: str, idx: int) -> str:
    candidates = [
        os.path.join(recon_dir, f"recon_patch_{idx}.npy"),
        os.path.join(recon_dir, f"recon_whole_{idx}.npy"),
        os.path.join(recon_dir, f"recon_admm_{idx}.npy"),
        os.path.join(recon_dir, f"recon_{idx}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No recon file found for idx={idx} in {recon_dir}")


def export_one(
    idx: int,
    recon_dir: str,
    val_dir: str,
    out_dir: str,
    mask_select: int = 7,
    save_kspace_maps: bool = False,
):
    recon_path = find_recon_file(recon_dir, idx)
    sample_path = os.path.join(val_dir, f"sample_{idx}.pt")

    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Missing sample file: {sample_path}")

    recon = np.load(recon_path)
    recon = np.squeeze(recon)

    data = torch.load(sample_path, map_location="cpu", weights_only=False)

    gt = data["gt"].cpu().numpy()
    ksp = data["ksp"].cpu().numpy()
    s_map = data["s_map"].cpu().numpy()

    mask_key = f"mask_{mask_select}"
    if mask_key not in data:
        raise KeyError(f"{mask_key} not found in {sample_path}")

    mask = data[mask_key].cpu().numpy()

    zf = make_zero_filled_adjoint(ksp=ksp, s_map=s_map, mask=mask)

    recon_abs = np.abs(recon).astype(np.float32)
    gt_abs = np.abs(gt).astype(np.float32)
    zf_abs = np.abs(zf).astype(np.float32)
    err_abs = np.abs(recon_abs - gt_abs).astype(np.float32)

    # 读取作者预处理时保存的 ACS 99th percentile normalization constant
    # sample_*.pt 里一般有 norm_consts_99
    if "norm_consts_99" in data:
        norm_const = data["norm_consts_99"]
        if hasattr(norm_const, "cpu"):
            norm_const = norm_const.cpu().numpy()
        norm_const = np.asarray(norm_const).squeeze().astype(np.float32)
    else:
        norm_const = np.array(np.nan, dtype=np.float32)

    mat_dict = {
        "sample_idx": np.array(idx, dtype=np.int32),
        "mask_select": np.array(mask_select, dtype=np.int32),

        "recon_complex": recon.astype(np.complex64),
        "gt_complex": gt.astype(np.complex64),
        "zf_complex": zf.astype(np.complex64),

        "recon_abs": recon_abs,
        "gt_abs": gt_abs,
        "zf_abs": zf_abs,
        "err_abs": err_abs,

        # 给 MATLAB 旧代码用的别名
        "recon": recon_abs,
        "gt": gt_abs,
        "zf": zf_abs,

        # 作者预处理/后处理中使用的统一归一化常数
        "norm_const_acs99": norm_const,
        "norm_consts_99": norm_const,

        "mask": np.squeeze(mask).astype(np.float32),
    }

    # Optional: save full k-space and sensitivity maps. These files will be much larger.
    if save_kspace_maps:
        # Convert [Nc, H, W] -> [H, W, Nc], easier to use in MATLAB.
        mat_dict["ksp_hwc"] = np.transpose(ksp, (1, 2, 0)).astype(np.complex64)
        mat_dict["smap_hwc"] = np.transpose(s_map, (1, 2, 0)).astype(np.complex64)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sample_{idx}.mat")
    savemat(out_path, mat_dict, do_compression=True)

    print(f"[OK] idx={idx} -> {out_path}")


def parse_indices(idx_arg: str, recon_dir: str):
    if idx_arg.lower() == "all":
        files = sorted(glob.glob(os.path.join(recon_dir, "recon_patch_*.npy")))
        if not files:
            files = sorted(glob.glob(os.path.join(recon_dir, "recon_*.npy")))

        indices = []
        for f in files:
            name = os.path.basename(f)
            stem = name.replace(".npy", "")
            idx = int(stem.split("_")[-1])
            indices.append(idx)
        return sorted(indices)

    return [int(x.strip()) for x in idx_arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--idx", type=str, default="all", help="'all' or comma-separated indices, e.g. 0,1,2")
    parser.add_argument("--mask_select", type=int, default=7)
    parser.add_argument("--save_kspace_maps", action="store_true")
    args = parser.parse_args()

    indices = parse_indices(args.idx, args.recon_dir)
    print(f"Exporting {len(indices)} samples:", indices)

    for idx in indices:
        export_one(
            idx=idx,
            recon_dir=args.recon_dir,
            val_dir=args.val_dir,
            out_dir=args.out_dir,
            mask_select=args.mask_select,
            save_kspace_maps=args.save_kspace_maps,
        )


if __name__ == "__main__":
    main()

import os
import glob
import argparse
import numpy as np
import torch
from scipy.io import savemat


def fftmod_np(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[..., ::2, :] *= -1
    y[..., :, ::2] *= -1
    return y


def make_zero_filled_adjoint(ksp: np.ndarray, s_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Generate zero-filled / adjoint image:
        zf = sum_c conj(s_map_c) * ifft2(mask * ksp_c)
    """
    if mask.ndim == 3:
        mask2d = mask[0]
    else:
        mask2d = mask

    ksp_shifted = fftmod_np(ksp)
    us_ksp = ksp_shifted * mask2d[None, :, :]
    coil_imgs = np.fft.ifft2(us_ksp, axes=(-2, -1), norm="ortho")
    zf = np.sum(np.conj(s_map) * coil_imgs, axis=0)
    return zf


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


def parse_indices(idx_arg: str, recon_dir: str):
    if idx_arg.lower() == "all":
        patterns = [
            os.path.join(recon_dir, "recon_patch_*.npy"),
            os.path.join(recon_dir, "recon_whole_*.npy"),
            os.path.join(recon_dir, "recon_admm_*.npy"),
            os.path.join(recon_dir, "recon_*.npy"),
        ]

        files = []
        for pat in patterns:
            files = sorted(glob.glob(pat))
            if files:
                break

        indices = []
        for f in files:
            stem = os.path.basename(f).replace(".npy", "")
            idx = int(stem.split("_")[-1])
            indices.append(idx)

        return sorted(indices)

    return [int(x.strip()) for x in idx_arg.split(",") if x.strip()]


def export_one(idx: int, recon_dir: str, val_dir: str, out_dir: str, mask_select: int):
    recon_path = find_recon_file(recon_dir, idx)
    sample_path = os.path.join(val_dir, f"sample_{idx}.pt")

    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Missing validation sample: {sample_path}")

    recon = np.squeeze(np.load(recon_path))

    data = torch.load(sample_path, map_location="cpu", weights_only=False)

    gt = data["gt"].cpu().numpy()
    ksp = data["ksp"].cpu().numpy()
    s_map = data["s_map"].cpu().numpy()

    mask_key = f"mask_{mask_select}"
    if mask_key not in data:
        raise KeyError(f"{mask_key} not found in {sample_path}")

    mask = data[mask_key].cpu().numpy()

    zf = make_zero_filled_adjoint(ksp=ksp, s_map=s_map, mask=mask)

    # 这里直接保存 magnitude，契合你的 evaluate_cold_diffusion_matlab.m
    gt_abs = np.abs(gt).astype(np.float32)
    recon_abs = np.abs(recon).astype(np.float32)
    zf_abs = np.abs(zf).astype(np.float32)

    file_name = f"sample_{idx:03d}"
    slice_idx = idx

    mat_dict = {
        "gt": gt_abs,
        "recon": recon_abs,
        "zf": zf_abs,
        "file_name": file_name,
        "slice_idx": np.array(slice_idx, dtype=np.int32),
    }

    os.makedirs(out_dir, exist_ok=True)

    # 必须包含 _slice，契合 MATLAB 里的 *_slice*.mat
    out_name = f"{file_name}_slice{slice_idx:03d}.mat"
    out_path = os.path.join(out_dir, out_name)

    savemat(out_path, mat_dict, do_compression=True)
    print(f"[OK] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--idx", type=str, default="all")
    parser.add_argument("--mask_select", type=int, default=7)
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
        )


if __name__ == "__main__":
    main()
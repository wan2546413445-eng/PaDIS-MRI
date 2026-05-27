import random
import torch
from denoise_padding import getIndices as cross_getIndices


def _coord_token(i, j, ps, H, W, device):
    cy = i + ps / 2.0 - 0.5
    cx = j + ps / 2.0 - 0.5
    return torch.tensor([(cx / (W - 1) - 0.5) * 2.0, (cy / (H - 1) - 0.5) * 2.0], device=device, dtype=torch.float32)


def _sample_context(target_i, target_j, H, W, ps, cp_k, cp_local_k, cp_global_k):
    coords = [(target_i, target_j)]
    max_i, max_j = H - ps, W - ps
    jitter = max(1, ps // 2)

    tries = 0
    while len(coords) < 1 + cp_local_k and tries < 200:
        tries += 1
        ni = max(0, min(max_i, target_i + random.randint(-jitter, jitter)))
        nj = max(0, min(max_j, target_j + random.randint(-jitter, jitter)))
        if (ni, nj) not in coords:
            coords.append((ni, nj))
    while len(coords) < 1 + cp_local_k:
        coords.append((target_i, target_j))

    far = ps
    tries = 0
    while len(coords) < cp_k and tries < 400:
        tries += 1
        gi = random.randint(0, max_i)
        gj = random.randint(0, max_j)
        if (gi, gj) in coords:
            continue
        if abs(gi - target_i) + abs(gj - target_j) < far:
            continue
        coords.append((gi, gj))
    while len(coords) < cp_k:
        coords.append((target_i, target_j))
    return coords


def denoisedFromCrossPatchSets(net, x_hat, t_hat, latents_pos, class_labels, indices,
                               cp_k=8, cp_local_k=3, cp_global_k=4, cp_eval_batch_size=2, cp_debug=False):
    assert cp_k == 1 + cp_local_k + cp_global_k
    device = x_hat.device
    _, C, H, W = x_hat.shape
    ps = indices[0][1] - indices[0][0]

    out = torch.zeros_like(x_hat)
    wgt = torch.zeros_like(x_hat)

    for start in range(0, len(indices), cp_eval_batch_size):
        chunk = indices[start:start + cp_eval_batch_size]
        b = len(chunk)
        patch_set = torch.empty(b, cp_k, C, ps, ps, device=device)
        patch_pos = torch.empty(b, cp_k, 2, ps, ps, device=device)
        patch_tok = torch.empty(b, cp_k, 2, device=device)

        for bi, z in enumerate(chunk):
            ti, tj = z[0], z[2]
            coords = _sample_context(ti, tj, H, W, ps, cp_k, cp_local_k, cp_global_k)
            for k, (i, j) in enumerate(coords):
                patch_set[bi, k] = x_hat[0, :, i:i + ps, j:j + ps]
                patch_pos[bi, k] = latents_pos[0, :, i:i + ps, j:j + ps]
                patch_tok[bi, k] = _coord_token(i, j, ps, H, W, device)

        t_batch = t_hat.reshape(1).repeat(b).reshape(b, 1, 1, 1, 1)
        pred = net(patch_set, t_batch, x_pos=patch_pos, patch_coords=patch_tok, class_labels=class_labels)

        if cp_debug:
            assert patch_set.shape == (b, cp_k, C, ps, ps)
            assert patch_pos.shape == (b, cp_k, 2, ps, ps)
            assert patch_tok.shape == (b, cp_k, 2)
            assert pred.shape == (b, cp_k, C, ps, ps)

        for bi, z in enumerate(chunk):
            out[0, :, z[0]:z[1], z[2]:z[3]] += pred[bi, 0]
            wgt[0, :, z[0]:z[1], z[2]:z[3]] += 1.0

    D_real = out / torch.clamp(wgt, min=1.0)
    if cp_debug:
        assert D_real.shape == x_hat.shape
    return D_real

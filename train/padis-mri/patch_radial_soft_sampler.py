from pathlib import Path
import time

p = Path("training/patch_loss.py")
s = p.read_text()

if "anatomy-gated gradient-aware sampler" in s:
    raise RuntimeError(
        "training/patch_loss.py already contains anatomy-gated gradient-aware sampler. "
        "Please restore a clean patch_loss.py before patching again."
    )

bak = p.with_suffix(p.suffix + f".bak_agg_{time.strftime('%Y%m%d_%H%M%S')}")
bak.write_text(s)
print("Backup:", bak)

old_init = """        self.sigma_data = sigma_data
"""

new_init = """        self.sigma_data = sigma_data

        # Anatomy-gated gradient-aware sampler.
        #
        # This sampler first estimates an anatomical support region from each MRI slice,
        # then uses both radial anatomical proximity and local gradient strength to
        # sample patches. It is designed to avoid wasting too much probability on
        # far-background / external-padding areas while increasing the chance of
        # sampling structural and detail-rich regions.
        #
        # It is NOT a pure high-gradient sampler:
        #   - radial/anatomy gate suppresses far-background and padding;
        #   - gradient score emphasizes local structure;
        #   - clipping prevents skull / extreme boundaries from dominating;
        #   - base probability and uniform fallback keep background modeled.
        #
        # Compatible with old radial env names:
        #   PADIS_RADIAL_ENABLE=1 will also enable this sampler.
        #
        # Recommended:
        #   PADIS_AGG_ENABLE=1
        #   PADIS_AGG_SAMPLE_PROB=0.8
        #   PADIS_AGG_THR_RATIO=0.05
        #   PADIS_AGG_BASE=0.05
        #   PADIS_AGG_RADIUS_SCALE=1.15
        #   PADIS_AGG_TAU_SCALE=0.30
        #   PADIS_AGG_GRAD_ALPHA=2.0
        #   PADIS_AGG_GRAD_CLIP_Q=0.95
        #   PADIS_AGG_GATE_BASE=0.20
        self.agg_enable = int(os.environ.get('PADIS_AGG_ENABLE', os.environ.get('PADIS_RADIAL_ENABLE', '0')))
        self.agg_sample_prob = float(os.environ.get('PADIS_AGG_SAMPLE_PROB', os.environ.get('PADIS_RADIAL_SAMPLE_PROB', '0.8')))
        self.agg_thr_ratio = float(os.environ.get('PADIS_AGG_THR_RATIO', os.environ.get('PADIS_RADIAL_THR_RATIO', '0.05')))
        self.agg_base = float(os.environ.get('PADIS_AGG_BASE', os.environ.get('PADIS_RADIAL_BASE', '0.05')))
        self.agg_radius_scale = float(os.environ.get('PADIS_AGG_RADIUS_SCALE', os.environ.get('PADIS_RADIAL_RADIUS_SCALE', '1.15')))
        self.agg_tau_scale = float(os.environ.get('PADIS_AGG_TAU_SCALE', os.environ.get('PADIS_RADIAL_TAU_SCALE', '0.30')))
        self.agg_min_pixels = int(os.environ.get('PADIS_AGG_MIN_PIXELS', os.environ.get('PADIS_RADIAL_MIN_PIXELS', '64')))

        # Gradient-aware part.
        self.agg_grad_alpha = float(os.environ.get('PADIS_AGG_GRAD_ALPHA', '2.0'))
        self.agg_grad_clip_q = float(os.environ.get('PADIS_AGG_GRAD_CLIP_Q', '0.95'))
        self.agg_gate_base = float(os.environ.get('PADIS_AGG_GATE_BASE', '0.20'))

        if self.agg_enable:
            print(
                f"[Patch_EDMLoss] anatomy-gated gradient-aware sampler enabled: "
                f"prob={self.agg_sample_prob}, "
                f"thr={self.agg_thr_ratio}, "
                f"base={self.agg_base}, "
                f"radius_scale={self.agg_radius_scale}, "
                f"tau_scale={self.agg_tau_scale}, "
                f"grad_alpha={self.agg_grad_alpha}, "
                f"grad_clip_q={self.agg_grad_clip_q}, "
                f"gate_base={self.agg_gate_base}"
            )
"""

if old_init not in s:
    raise RuntimeError("Cannot find sigma_data init block. Restore a clean training/patch_loss.py first.")

s = s.replace(old_init, new_init, 1)

old_block = """        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device).long()
            j = torch.zeros((batch_size,), device=device).long()
        else:
            i = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j = torch.randint(0, w - tw + 1, (batch_size,), device=device)
"""

new_block = """        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device).long()
            j = torch.zeros((batch_size,), device=device).long()
        else:
            # Original uniform sampler over the whole padded canvas.
            i_uniform = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j_uniform = torch.randint(0, w - tw + 1, (batch_size,), device=device)
            i = i_uniform.clone()
            j = j_uniform.clone()

            # Anatomy-gated gradient-aware sampler.
            # With probability agg_sample_prob, replace the uniform top-left
            # location by a sample from:
            #   score = base + radial_score * anatomy_gate * (1 + alpha * grad_score)
            if self.agg_enable > 0:
                use_agg = torch.rand(batch_size, device=device) < self.agg_sample_prob

                # Magnitude image. For MRI complex input, first two channels are real/imag.
                use_ch = min(2, padded.size(1))
                mag = padded[:, :use_ch].float().square().sum(dim=1).sqrt()  # [B,H,W]

                yy = torch.arange(h, device=device, dtype=torch.float32).view(h, 1).expand(h, w)
                xx = torch.arange(w, device=device, dtype=torch.float32).view(1, w).expand(h, w)

                max_i = h - th
                max_j = w - tw

                # Candidate patch centers corresponding to all possible top-left positions.
                ci = torch.arange(0, max_i + 1, device=device, dtype=torch.float32) + th / 2.0
                cj = torch.arange(0, max_j + 1, device=device, dtype=torch.float32) + tw / 2.0
                grid_cy, grid_cx = torch.meshgrid(ci, cj, indexing='ij')  # [H-th+1, W-tw+1]

                for b in range(batch_size):
                    if not bool(use_agg[b].item()):
                        continue

                    mb = mag[b]
                    peak = mb.amax().clamp_min(1e-8)

                    # Anatomical support is only used to estimate center/radius
                    # and an anatomy gate. It is not a hard mask.
                    support = (mb > self.agg_thr_ratio * peak).float()
                    mass = support.sum()

                    if mass < self.agg_min_pixels:
                        continue

                    cy = (support * yy).sum() / mass
                    cx = (support * xx).sum() / mass

                    # Equivalent radius from support area.
                    radius = torch.sqrt(mass / 3.141592653589793) * self.agg_radius_scale
                    tau = torch.clamp(radius * self.agg_tau_scale, min=8.0)

                    dist = torch.sqrt((grid_cy - cy) ** 2 + (grid_cx - cx) ** 2)
                    outside = torch.clamp(dist - radius, min=0.0)

                    # Radial anatomical proximity.
                    # Inside estimated anatomical envelope: close to 1.
                    # Outside: decays smoothly.
                    radial_score = torch.exp(-0.5 * (outside / tau) ** 2)

                    # Local gradient magnitude.
                    # Use simple finite differences to avoid adding dependencies.
                    gx = torch.zeros_like(mb)
                    gy = torch.zeros_like(mb)
                    gx[:, 1:] = mb[:, 1:] - mb[:, :-1]
                    gy[1:, :] = mb[1:, :] - mb[:-1, :]
                    grad = torch.sqrt(gx.square() + gy.square())

                    # Patch-level average gradient for every candidate top-left.
                    grad_patch = torch.nn.functional.avg_pool2d(
                        grad[None, None],
                        kernel_size=(th, tw),
                        stride=1
                    )[0, 0]  # [H-th+1, W-tw+1]

                    # Patch-level support fraction. This prevents pure padding or far
                    # background gradients from dominating, but does not fully remove background.
                    support_frac = torch.nn.functional.avg_pool2d(
                        support[None, None],
                        kernel_size=(th, tw),
                        stride=1
                    )[0, 0].clamp(0.0, 1.0)

                    # Normalize gradient robustly.
                    # q=0.95 clips very strong boundaries, reducing skull-edge domination.
                    gp = grad_patch.flatten()
                    positive = gp[gp > 0]

                    if positive.numel() < 8:
                        grad_norm = torch.zeros_like(grad_patch)
                    else:
                        q = torch.quantile(positive, self.agg_grad_clip_q).clamp_min(1e-8)
                        grad_norm = (grad_patch / q).clamp(0.0, 1.0)

                    # Anatomy gate:
                    #   support_frac high  -> keep gradient emphasis;
                    #   support_frac low   -> still keep gate_base probability.
                    anatomy_gate = self.agg_gate_base + (1.0 - self.agg_gate_base) * support_frac

                    # Final sampling score.
                    # base keeps uniform-like coverage for background and stability.
                    score = (
                        self.agg_base
                        + radial_score * anatomy_gate * (1.0 + self.agg_grad_alpha * grad_norm)
                    )

                    score = score.flatten().clamp_min(1e-8)

                    idx = torch.multinomial(score, 1)[0].long()
                    ii = torch.div(idx, max_j + 1, rounding_mode='floor')
                    jj = idx - ii * (max_j + 1)

                    i[b] = ii
                    j[b] = jj
"""

if old_block not in s:
    raise RuntimeError(
        "Cannot find original uniform crop block in training/patch_loss.py. "
        "Restore a clean training/patch_loss.py first."
    )

s = s.replace(old_block, new_block, 1)

p.write_text(s)
print("Patched:", p)

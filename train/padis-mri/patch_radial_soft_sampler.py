from pathlib import Path
import time

p = Path("training/patch_loss.py")
s = p.read_text()

bak = p.with_suffix(p.suffix + f".bak_radial_{time.strftime('%Y%m%d_%H%M%S')}")
bak.write_text(s)
print("Backup:", bak)

old_init = """        self.sigma_data = sigma_data
"""

new_init = """        self.sigma_data = sigma_data

        # Anatomy-centered radial soft sampler.
        # This sampler estimates an anatomical support center and radius
        # from each MRI slice, then softly downweights far-background and
        # external-padding regions.
        #
        # It does NOT use high-gradient selection.
        # It does NOT exclude skull or image-internal background.
        # It only changes the probability of patch-center sampling.
        #
        # Recommended:
        #   PADIS_RADIAL_ENABLE=1
        #   PADIS_RADIAL_SAMPLE_PROB=0.8
        #   PADIS_RADIAL_THR_RATIO=0.05
        #   PADIS_RADIAL_BASE=0.08
        #   PADIS_RADIAL_RADIUS_SCALE=1.20
        #   PADIS_RADIAL_TAU_SCALE=0.35
        self.radial_enable = int(os.environ.get('PADIS_RADIAL_ENABLE', '0'))
        self.radial_sample_prob = float(os.environ.get('PADIS_RADIAL_SAMPLE_PROB', '0.8'))
        self.radial_thr_ratio = float(os.environ.get('PADIS_RADIAL_THR_RATIO', '0.05'))
        self.radial_base = float(os.environ.get('PADIS_RADIAL_BASE', '0.08'))
        self.radial_radius_scale = float(os.environ.get('PADIS_RADIAL_RADIUS_SCALE', '1.20'))
        self.radial_tau_scale = float(os.environ.get('PADIS_RADIAL_TAU_SCALE', '0.35'))
        self.radial_min_pixels = int(os.environ.get('PADIS_RADIAL_MIN_PIXELS', '64'))

        if self.radial_enable:
            print(
                f"[Patch_EDMLoss] radial-soft sampler enabled: "
                f"prob={self.radial_sample_prob}, "
                f"thr={self.radial_thr_ratio}, "
                f"base={self.radial_base}, "
                f"radius_scale={self.radial_radius_scale}, "
                f"tau_scale={self.radial_tau_scale}"
            )
"""

if old_init not in s:
    raise RuntimeError("Cannot find sigma_data init block.")
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

            # Anatomy-centered radial soft sampler.
            # With probability radial_sample_prob, sample patch center from a
            # soft radial distribution estimated from the current slice.
            if self.radial_enable > 0:
                use_radial = torch.rand(batch_size, device=device) < self.radial_sample_prob

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
                grid_cy, grid_cx = torch.meshgrid(ci, cj, indexing='ij')

                for b in range(batch_size):
                    if not bool(use_radial[b].item()):
                        continue

                    mb = mag[b]
                    peak = mb.amax().clamp_min(1e-8)

                    # Estimate anatomical support from magnitude.
                    # This is only for center/radius estimation, not a hard sampling mask.
                    support = (mb > self.radial_thr_ratio * peak).float()
                    mass = support.sum()

                    if mass < self.radial_min_pixels:
                        continue

                    cy = (support * yy).sum() / mass
                    cx = (support * xx).sum() / mass

                    # Equivalent radius from support area.
                    # radius_scale > 1 expands to include skull/boundary nearby background.
                    radius = torch.sqrt(mass / 3.141592653589793) * self.radial_radius_scale
                    tau = torch.clamp(radius * self.radial_tau_scale, min=8.0)

                    dist = torch.sqrt((grid_cy - cy) ** 2 + (grid_cx - cx) ** 2)
                    outside = torch.clamp(dist - radius, min=0.0)

                    # Inside estimated anatomical envelope: high weight.
                    # Outside: gradually decays, but never zero due to radial_base.
                    score = self.radial_base + torch.exp(-0.5 * (outside / tau) ** 2)
                    score = score.flatten().clamp_min(1e-8)

                    idx = torch.multinomial(score, 1)[0].long()
                    ii = torch.div(idx, max_j + 1, rounding_mode='floor')
                    jj = idx - ii * (max_j + 1)

                    i[b] = ii
                    j[b] = jj
"""

if old_block not in s:
    raise RuntimeError("Cannot find original uniform crop block. The file may not be clean original.")
s = s.replace(old_block, new_block, 1)

p.write_text(s)
print("Patched:", p)

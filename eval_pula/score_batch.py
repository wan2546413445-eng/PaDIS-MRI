"""Pack trajectory partitions together; retain the legacy patch residual stitching."""

from collections import defaultdict
from contextlib import contextmanager
import os
from time import perf_counter

import torch
import torch.nn.functional as F

from eval_pula.score_opa import PatchScore, real_channels


@contextmanager
def synchronized_stage(device, totals, name):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    totals[name] += perf_counter() - start


class TrajectoryPatchScore(PatchScore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_stats()

    def reset_stats(self):
        self.stages = defaultdict(float)
        self.network_batch_calls = 0
        self.patches_processed = 0

    @torch.no_grad()
    def denoise(self, state, sigma, shifts):
        # state is [trajectories,1,H,W]; each shift belongs to exactly one row.
        with synchronized_stage(state.device, self.stages, "patch_preparation_seconds"):
            query = real_channels(F.pad(state, (self.pad,) * 4))
            images, positions, metadata = [], [], []
            for trajectory, (row, col) in enumerate(shifts):
                indices = [(a + row, b + row, c + col, d + col) for a, b, c, d in self.indices]
                start = len(images)
                for a, b, c, d in indices:
                    images.append(query[trajectory, :, a:b, c:d])
                    positions.append(self.positions[0, :, a:b, c:d])
                metadata.append((trajectory, start, len(images), indices))
            # Legacy denoisedFromPatches allocates network inputs as float32.
            image_batch = torch.stack(images).to(torch.float32)
            position_batch = torch.stack(positions).to(torch.float32)
            sigma_arg = query.new_tensor(sigma)
        with synchronized_stage(state.device, self.stages, "network_forward_seconds"):
            microbatch = int(os.environ.get("PADIS_PATCH_BATCH", len(images)))
            if microbatch < 1:
                raise ValueError("PADIS_PATCH_BATCH must be positive.")
            outputs = []
            for start in range(0, len(images), microbatch):
                stop = start + microbatch
                outputs.append(self.net(image_batch[start:stop], sigma_arg,
                                        position_batch[start:stop], None).to(torch.float64))
                self.network_batch_calls += 1
            predictions = torch.cat(outputs)
            self.patches_processed += len(images)
        with synchronized_stage(state.device, self.stages, "patch_stitching_seconds"):
            residual = torch.zeros_like(query)
            for trajectory, start, stop, indices in metadata:
                for prediction, (a, b, c, d) in zip(predictions[start:stop], indices):
                    # Keep += prediction; -= input; input + residual, including
                    # the legacy float64 -> float32 rounding at the first +=.
                    residual[trajectory, :, a:b, c:d] += prediction
                    residual[trajectory, :, a:b, c:d] -= query[trajectory, :, a:b, c:d]
            denoised = query + residual
            fov = slice(self.pad, self.pad + self.image_size)
            denoised = denoised[:, :, fov, fov]
            # The old helper's randn_like(...)*0 only fills discarded padding;
            # it has no numerical role and does not draw from trajectory RNGs.
            result = torch.complex(denoised[:, 0], denoised[:, 1]).unsqueeze(1)
        return result

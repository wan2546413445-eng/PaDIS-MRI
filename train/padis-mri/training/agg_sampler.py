"""Anatomy-gradient guided sampling for PaDIS-MRI patches.

The sampler uses three fixed method definitions:

* anatomical support is extracted at 5% of the image peak magnitude;
* gradient scores are clipped at their 95th percentile;
* overlapping pairs use a half-patch displacement.

Only ``rho`` and ``gamma`` remain adjustable.  ``rho`` mixes uniform and
structure-guided sampling, while ``gamma`` controls the gradient contribution.
"""

import torch
import torch.nn.functional as F


SUPPORT_THRESHOLD_RATIO = 0.05
GRADIENT_CLIP_QUANTILE = 0.95


def _structure_maps(image):
    """Return anatomical support and gradient magnitude for one image."""
    use_channels = min(2, image.shape[0])
    magnitude = image[:use_channels].float().square().sum(dim=0).sqrt()
    peak = magnitude.amax()
    if not torch.isfinite(peak) or float(peak.item()) <= 0.0:
        return None

    support = (magnitude > SUPPORT_THRESHOLD_RATIO * peak).float()
    if float(support.sum().item()) <= 0.0:
        return None

    grad_x = torch.zeros_like(magnitude)
    grad_y = torch.zeros_like(magnitude)
    grad_x[:, 1:] = magnitude[:, 1:] - magnitude[:, :-1]
    grad_y[1:, :] = magnitude[1:, :] - magnitude[:-1, :]
    gradient = torch.sqrt(grad_x.square() + grad_y.square())
    return support, gradient


def _normalize_candidate_gradient(gradient_score):
    """Normalize candidate gradient scores with a fixed robust scale."""
    positive = gradient_score[gradient_score > 0]
    if positive.numel() == 0:
        return torch.zeros_like(gradient_score)
    clip_value = torch.quantile(
        positive, GRADIENT_CLIP_QUANTILE
    ).clamp_min(torch.finfo(gradient_score.dtype).eps)
    return (gradient_score / clip_value).clamp(0.0, 1.0)


def _sampling_probabilities(score, rho):
    """Mix a normalized structure score with the uniform distribution."""
    flat = torch.nan_to_num(score.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = flat.clamp_min(0.0)
    count = flat.numel()
    if count == 0:
        raise ValueError('AGG candidate set is empty')

    total = flat.sum()
    uniform = torch.full_like(flat, 1.0 / count)
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return uniform
    guided = flat / total
    return (1.0 - rho) * uniform + rho * guided


def _region_score(support, gradient, kernel_size, gamma):
    """Compute S(u)=A(u)[1+gamma*G(u)] for candidate regions."""
    anatomy = F.avg_pool2d(
        support[None, None], kernel_size=kernel_size, stride=1
    )[0, 0]
    gradient_mean = F.avg_pool2d(
        gradient[None, None], kernel_size=kernel_size, stride=1
    )[0, 0]
    gradient_normalized = _normalize_candidate_gradient(gradient_mean)
    return anatomy * (1.0 + gamma * gradient_normalized)


def sample_single_locations(images, patch_size, rho, gamma):
    """Sample one patch location per image from the AGG mixture."""
    patch_size = int(patch_size)
    batch, _, height, width = images.shape
    max_top = height - patch_size
    max_left = width - patch_size
    if min(max_top, max_left) < 0:
        raise ValueError('patch_size is larger than the input image')

    top = torch.empty(batch, device=images.device, dtype=torch.long)
    left = torch.empty(batch, device=images.device, dtype=torch.long)
    for index in range(batch):
        maps = _structure_maps(images[index])
        if maps is None:
            flat_index = torch.randint(
                0, (max_top + 1) * (max_left + 1), [], device=images.device
            )
            score_width = max_left + 1
        else:
            support, gradient = maps
            score = _region_score(
                support, gradient, (patch_size, patch_size), gamma
            )
            probabilities = _sampling_probabilities(score, rho)
            flat_index = torch.multinomial(probabilities, 1)[0]
            score_width = score.shape[1]

        top[index] = torch.div(flat_index, score_width, rounding_mode='floor')
        left[index] = flat_index % score_width
    return top, left


def _overlap_score(support, gradient, patch_size, horizontal, gamma):
    """Score the physical overlap strip shared by every candidate pair."""
    overlap = patch_size // 2
    stride = patch_size - overlap
    if horizontal:
        score = _region_score(
            support, gradient, (patch_size, overlap), gamma
        )
        return score[:, stride: support.shape[1] - patch_size + 1]

    score = _region_score(
        support, gradient, (overlap, patch_size), gamma
    )
    return score[stride: support.shape[0] - patch_size + 1, :]


def sample_overlap_locations(images, patch_size, rho, gamma):
    """Sample half-shift pairs using their shared overlap-region scores."""
    patch_size = int(patch_size)
    overlap = patch_size // 2
    stride = patch_size - overlap
    if overlap <= 0:
        raise ValueError('patch_size must be at least 2')

    device = images.device
    batch, _, height, width = images.shape
    max_top_h = height - patch_size
    max_left_h = width - patch_size - stride
    max_top_v = height - patch_size - stride
    max_left_v = width - patch_size
    if min(max_top_h, max_left_h, max_top_v, max_left_v) < 0:
        raise ValueError(
            'image resolution is too small for half-shift overlapping patches'
        )

    horizontal = torch.rand(batch, device=device) < 0.5
    top1 = torch.empty(batch, device=device, dtype=torch.long)
    left1 = torch.empty(batch, device=device, dtype=torch.long)

    for index in range(batch):
        is_horizontal = bool(horizontal[index].item())
        candidate_height = (max_top_h + 1) if is_horizontal else (max_top_v + 1)
        candidate_width = (max_left_h + 1) if is_horizontal else (max_left_v + 1)
        maps = _structure_maps(images[index])

        if maps is None:
            flat_index = torch.randint(
                0, candidate_height * candidate_width, [], device=device
            )
        else:
            support, gradient = maps
            score = _overlap_score(
                support, gradient, patch_size, is_horizontal, gamma
            )
            if score.shape != (candidate_height, candidate_width):
                raise RuntimeError('AGG overlap score has inconsistent geometry')
            probabilities = _sampling_probabilities(score, rho)
            flat_index = torch.multinomial(probabilities, 1)[0]

        top1[index] = torch.div(
            flat_index, candidate_width, rounding_mode='floor'
        )
        left1[index] = flat_index % candidate_width

    top2 = torch.where(horizontal, top1, top1 + stride)
    left2 = torch.where(horizontal, left1 + stride, left1)
    return top1, left1, top2, left2, horizontal, overlap

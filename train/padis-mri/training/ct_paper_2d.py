"""2D complex MRI port of Yang et al., arXiv:2512.18161, Sec 3.1 / App A-B.
Architecture follows the DDIM-style residual U-Net (Song et al. 2020).
No NGS, GC gates, separate global encoder, or EDM training objective.
The paper does not specify D; this port uses the existing PaDIS JointCT
8x8 average-pooling convention. The original CT implementation is not public.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def downsample_global(x, patch_size=64):
    return F.adaptive_avg_pool2d(x, (patch_size, patch_size))


def extract_patch(x, top, left, p):
    b = x.shape[0]
    rows = top[:, None, None] + torch.arange(p, device=x.device)[None, :, None]
    cols = left[:, None, None] + torch.arange(p, device=x.device)[None, None, :]
    return x.permute(0, 2, 3, 1)[torch.arange(b, device=x.device)[:, None, None], rows, cols].permute(0, 3, 1, 2).contiguous()


def position_patch(top, left, p, height, width):
    rows = top[:, None, None] + torch.arange(p, device=top.device)[None, :, None]
    cols = left[:, None, None] + torch.arange(p, device=top.device)[None, None, :]
    yy = (2 * rows.float() / (height - 1) - 1).expand(-1, p, p)
    xx = (2 * cols.float() / (width - 1) - 1).expand(-1, p, p)
    return torch.stack([xx, yy], dim=1)


def timestep_embedding(t, dim):
    f = torch.exp(-math.log(10000.0) * torch.arange(dim // 2, device=t.device, dtype=torch.float32) / (dim // 2 - 1))
    phase = t.float().reshape(-1, 1) * f[None]
    return torch.cat([phase.sin(), phase.cos()], dim=1)


class Residual(nn.Module):
    def __init__(self, cin, cout, temb, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin, eps=1e-6)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(temb, cout)
        self.norm2 = nn.GroupNorm(32, cout, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x))) + self.time(F.silu(emb))[:, :, None, None]
        return self.skip(x) + self.conv2(self.dropout(F.silu(self.norm2(h))))


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, y, z = h.shape
        q = self.q(h).flatten(2).transpose(1, 2)
        k = self.k(h).flatten(2)
        v = self.v(h).flatten(2).transpose(1, 2)
        a = (q @ k * c ** -0.5).softmax(dim=-1)
        return x + self.proj((a @ v).transpose(1, 2).reshape(b, c, y, z))


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Level(nn.Module):
    def __init__(self, blocks, attention, resize):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.attention = nn.ModuleList(attention)
        self.resize = resize


class CTDDIMUNet(nn.Module):
    """Appendix B: width 64, mult 1/2/4/4, 2 blocks, attention at 8x8."""
    def __init__(self, patch_size=64, dropout=0.05, base=64):
        super().__init__()
        self.base = base
        self.patch_size = patch_size
        self.time = nn.Sequential(nn.Linear(base, 4*base), nn.SiLU(), nn.Linear(4*base, 4*base))
        self.input = nn.Conv2d(6, base, 3, padding=1)
        mult = (1, 2, 4, 4)
        c = base
        skip_channels = [c]
        self.down = nn.ModuleList()
        for level, m in enumerate(mult):
            res = patch_size // 2**level
            blocks, attn = [], []
            for _ in range(2):
                blocks.append(Residual(c, base*m, 4*base, dropout))
                c = base*m
                attn.append(Attention(c) if res == 8 else nn.Identity())
                skip_channels.append(c)
            resize = Downsample(c) if level < len(mult)-1 else nn.Identity()
            if level < len(mult)-1:
                skip_channels.append(c)
            self.down.append(Level(blocks, attn, resize))
        self.mid1 = Residual(c, c, 4*base, dropout)
        self.midattn = Attention(c)
        self.mid2 = Residual(c, c, 4*base, dropout)
        self.up = nn.ModuleList()
        for level in reversed(range(len(mult))):
            res = patch_size // 2**level
            blocks, attn = [], []
            for _ in range(3):
                blocks.append(Residual(c+skip_channels.pop(), base*mult[level], 4*base, dropout))
                c = base*mult[level]
                attn.append(Attention(c) if res == 8 else nn.Identity())
            self.up.append(Level(blocks, attn, Upsample(c) if level else nn.Identity()))
        assert not skip_channels
        self.output = nn.Sequential(nn.GroupNorm(32, c, eps=1e-6), nn.SiLU(), nn.Conv2d(c, 2, 3, padding=1))

    def forward(self, x, t):
        if x.shape[1:] != (6, self.patch_size, self.patch_size):
            raise ValueError(f'Expected [B,6,{self.patch_size},{self.patch_size}], got {tuple(x.shape)}')
        emb = self.time(timestep_embedding(t, self.base))
        h = self.input(x)
        skips = [h]
        for i, level in enumerate(self.down):
            for block, attn in zip(level.blocks, level.attention):
                h = attn(block(h, emb))
                skips.append(h)
            if i != len(self.down)-1:
                h = level.resize(h)
                skips.append(h)
        h = self.mid2(self.midattn(self.mid1(h, emb)), emb)
        for level in self.up:
            for block, attn in zip(level.blocks, level.attention):
                h = attn(block(torch.cat([h, skips.pop()], dim=1), emb))
            h = level.resize(h)
        assert not skips
        return self.output(h)


class CTPaperPrior(nn.Module):
    """VP epsilon network with a separate adapter for PaDIS VE/DPS evaluation."""
    def __init__(self, dropout=0.05):
        super().__init__()
        self.patch_size = 64
        self.img_resolution = 512
        self.img_channels = 4 # external PaDIS interface: complex + position
        self.out_channels = 2
        self.label_dim = 0
        self.use_joint_global = True
        self.joint_context_size = 64
        self.sigma_data = 0.5 # interface metadata only; not used in VP training
        self.model = CTDDIMUNet(dropout=dropout)
        alpha = (1 - torch.linspace(1e-4, 2e-2, 1000, dtype=torch.float64)).cumprod(0)
        self.register_buffer('alpha_bar', alpha.float())
        self.register_buffer('ve_sigma', ((1-alpha)/alpha).sqrt().float())
        self.sigma_min = float(self.ve_sigma[0]) * math.sqrt(2)
        self.sigma_max = float(self.ve_sigma[-1]) * math.sqrt(2)

    def predict_noise(self, local_vp, context_vp, pos, t):
        if local_vp.shape != context_vp.shape or local_vp.shape != pos.shape:
            raise ValueError('Local, global and position shapes must match')
        return self.model(torch.cat([local_vp, context_vp, pos], dim=1).float(), t)

    def round_sigma(self, sigma):
        # Keep the requested VE schedule; use interpolation for timestep label.
        return torch.as_tensor(sigma)

    def sigma_to_time(self, s):
        grid = self.ve_sigma.to(device=s.device, dtype=s.dtype)
        clipped = s.clamp(grid[0], grid[-1])
        hi = torch.searchsorted(grid, clipped.contiguous()).clamp(1, len(grid)-1)
        lo = hi - 1
        return lo.to(s.dtype) + (clipped-grid[lo])/(grid[hi]-grid[lo])

    def forward(self, x, sigma, x_pos=None, class_labels=None, force_fp32=False, global_context=None, **kwargs):
        # This is the PaDIS DPS adapter, not the CT paper's DDIM sampler.
        # torch.randn_like(complex64) has real/imag variance 1/2.
        if global_context is None or x_pos is None:
            raise ValueError('CT paper prior requires full noisy context and coordinates')
        x = x.float()
        s = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1)
        if s.numel() == 1:
            s = s.expand(x.shape[0])
        if s.numel() != x.shape[0]:
            raise ValueError('sigma batch mismatch')
        s = s / math.sqrt(2.0)
        t = self.sigma_to_time(s)
        scale = (1+s.square()).rsqrt()[:, None, None, None]
        eps = self.predict_noise(x*scale, global_context.float()*scale, x_pos.float(), t)
        return x - s[:, None, None, None]*eps


def vp_training_loss(net, clean_full):
    """Algorithm 2: noise full field, crop, downsample, predict local epsilon."""
    b, c, h, w = clean_full.shape
    p = net.patch_size
    t = torch.randint(0, 1000, (b,), device=clean_full.device)
    alpha = net.alpha_bar[t, None, None, None]
    eps = torch.randn_like(clean_full)
    xt = alpha.sqrt()*clean_full + (1-alpha).sqrt()*eps
    top = torch.randint(0, h-p+1, (b,), device=clean_full.device)
    left = torch.randint(0, w-p+1, (b,), device=clean_full.device)
    local = extract_patch(xt, top, left, p)
    target = extract_patch(eps, top, left, p)
    context = downsample_global(xt, p)
    pos = position_patch(top, left, p, h, w)
    pred = net.predict_noise(local, context, pos, t)
    return F.mse_loss(pred, target)

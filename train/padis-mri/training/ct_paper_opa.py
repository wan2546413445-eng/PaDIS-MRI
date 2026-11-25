"""CT-inspired DDIM backbone + unchanged PaDIS EDM/OPA training.
See README for the distinction from the original VP CT method.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_utils import persistence


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
    """Configurable CT-style DDIM UNet; defaults preserve the original port."""
    def __init__(self, patch_size=64, dropout=0.05, base=64,
                 channel_mult=(1, 2, 4, 4), num_res_blocks=2,
                 attn_resolutions=(8,), noise_embedding="fourier", init_mode="edm"):
        super().__init__()
        mult = tuple(channel_mult)
        if base < 32 or base % 32:
            raise ValueError('base must be a positive multiple of 32 (GroupNorm)')
        if not mult or any(not isinstance(m, int) or m < 1 for m in mult):
            raise ValueError('channel_mult must contain positive integers')
        if len(mult) > 7 or patch_size % (2**(len(mult)-1)):
            raise ValueError('Incompatible channel_mult depth for patch_size')
        if num_res_blocks < 1:
            raise ValueError('num_res_blocks must be positive')
        self.channel_mult = mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = tuple(attn_resolutions)
        if not set(self.attn_resolutions).issubset({patch_size//2**i for i in range(len(mult))}):
            raise ValueError('Attention resolution must be one of the UNet feature sizes')
        if noise_embedding not in ('fourier', 'ddim'):
            raise ValueError('noise_embedding must be fourier or ddim')
        if init_mode not in ('edm', 'legacy'):
            raise ValueError('init_mode must be edm or legacy')
        self.noise_embedding = noise_embedding
        self.init_mode = init_mode
        self.base = base
        self.patch_size = patch_size
        self.time = nn.Sequential(nn.Linear(base, 4*base), nn.SiLU(), nn.Linear(4*base, 4*base))
        self.input = nn.Conv2d(6, base, 3, padding=1)
        c = base
        skip_channels = [c]
        self.down = nn.ModuleList()
        for level, m in enumerate(mult):
            res = patch_size // 2**level
            blocks, attn = [], []
            for _ in range(num_res_blocks):
                blocks.append(Residual(c, base*m, 4*base, dropout))
                c = base*m
                attn.append(Attention(c) if res in self.attn_resolutions else nn.Identity())
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
            for _ in range(num_res_blocks + 1):
                blocks.append(Residual(c+skip_channels.pop(), base*mult[level], 4*base, dropout))
                c = base*mult[level]
                attn.append(Attention(c) if res in self.attn_resolutions else nn.Identity())
            self.up.append(Level(blocks, attn, Upsample(c) if level else nn.Identity()))
        assert not skip_channels
        self.output = nn.Sequential(nn.GroupNorm(32, c, eps=1e-6), nn.SiLU(), nn.Conv2d(c, 2, 3, padding=1))
        if noise_embedding == 'fourier':
            # Same frequency distribution as training.networks.FourierEmbedding
            # (NCSN++ option in EDM): fixed Gaussian frequencies, scale=16.
            self.register_buffer('noise_freqs', torch.randn(base // 2) * 16)
        if init_mode == 'edm':
            self.init_edm()

    @torch.no_grad()
    def init_edm(self):
        # PaDIS SongUNet initialization policy, applied to CT's existing layers.
        # Do not alter CT residual, attention, skip or resampling topology.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for layer in self.modules():
            if isinstance(layer, Residual):
                layer.conv2.weight.mul_(1e-5)
            elif isinstance(layer, Attention):
                for qkv in (layer.q, layer.k, layer.v):
                    qkv.weight.mul_(math.sqrt(0.2))
                layer.proj.weight.mul_(1e-5)
        self.output[-1].weight.mul_(1e-5)

    def embed_noise(self, t):
        # getattr also permits legacy objects lacking the new attribute.
        if getattr(self, 'noise_embedding', 'ddim') == 'ddim':
            return timestep_embedding(t, self.base)
        phase = t.float().reshape(-1, 1) * (2 * math.pi * self.noise_freqs[None])
        # sin/cos ordering matches SongUNet after its embedding swap.
        return torch.cat([phase.sin(), phase.cos()], dim=1)


    def forward(self, x, t):
        if x.shape[1:] != (6, self.patch_size, self.patch_size):
            raise ValueError(f'Expected [B,6,{self.patch_size},{self.patch_size}], got {tuple(x.shape)}')
        emb = self.time(self.embed_noise(t))
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


@persistence.persistent_class
class CTPaperEDM(torch.nn.Module):
    """CT local/global/XY concatenation and DDIM UNet, with original PaDIS EDM interface."""
    def __init__(self, img_resolution, img_channels, out_channels=2, label_dim=0,
                 sigma_data=0.5, dropout=0.05, use_fp16=False,
                 model_channels=64, channel_mult=(1, 2, 4, 4),
                 num_res_blocks=2, attn_resolutions=(8,),
                 noise_embedding="fourier", init_mode="edm"):
        super().__init__()
        if img_channels != 4 or out_channels != 2 or label_dim != 0:
            raise ValueError('Expected complex MRI with XY maps and no class labels')
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.use_fp16 = use_fp16
        self.sigma_min = 0.0
        self.sigma_max = float('inf')
        self.use_joint_global = True
        self.joint_context_size = 64
        self.model = CTDDIMUNet(
            patch_size=64, dropout=dropout, base=model_channels,
            channel_mult=channel_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions, noise_embedding=noise_embedding,
            init_mode=init_mode)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, sigma, x_pos=None, class_labels=None, force_fp32=False,
                global_context=None, augment_labels=None, **kwargs):
        x = x.float()
        if x_pos is None or x_pos.shape != x.shape:
            raise ValueError('Expected matching XY position maps')
        if global_context is None:
            # Existing training loop calls a no-grad, full-FOV module summary
            # before optimization. This is not a training or sampling path.
            if torch.is_grad_enabled() or x.shape[-2:] == (64, 64):
                raise ValueError('CT+OPA requires the noisy full-field context')
            global_context = downsample_global(x, 64)
            h, w = x.shape[-2:]
            if min(h,w) < 64:
                raise ValueError('Expected a full-FOV summary input')
            i,j = (h-64)//2,(w-64)//2
            x = x[:,:,i:i+64,j:j+64]
            x_pos = x_pos[:,:,i:i+64,j:j+64]
        if x.shape[-2:] != (64,64) or global_context.shape != x.shape:
            raise ValueError('CT+OPA expects fixed 64x64 local and global tensors')
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1,1,1,1)
        c_skip = self.sigma_data**2 / (sigma.square()+self.sigma_data**2)
        c_out = sigma*self.sigma_data / (sigma.square()+self.sigma_data**2).sqrt()
        c_in = (sigma.square()+self.sigma_data**2).rsqrt()
        c_noise = sigma.log().flatten()/4
        if c_noise.numel() == 1:
            c_noise = c_noise.expand(x.shape[0])
        # Same input ordering as the CT-style port: local, global, XY.
        z = torch.cat([c_in*x,c_in*global_context.float(),x_pos.float()],dim=1)
        return c_skip*x+c_out*self.model(z,c_noise).float()

import numpy as np
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from torch_utils import persistence
from torch.nn.functional import silu
from training.networks import Linear, UNetBlock, SongUNet

def _is_unet_block(block):
    return block.__class__.__name__ == "UNetBlock"


@persistence.persistent_class
class CrossPatchSongUNet(SongUNet):
    def __init__(self, *args, cp_num_heads=4, cp_depth=2, cp_ffn_mult=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp_num_heads = cp_num_heads
        self.cp_depth = cp_depth
        self.cp_ffn_mult = cp_ffn_mult

        bottleneck = [m.out_channels for m in self.enc.values() if hasattr(m, 'out_channels')][-1]
        self.coord_proj = Linear(2, bottleneck)
        self.noise_proj = Linear(self.map_layer1.out_features, bottleneck)
        self.token_ln = torch.nn.LayerNorm(bottleneck)

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=bottleneck,
            nhead=cp_num_heads,
            dim_feedforward=bottleneck * cp_ffn_mult,
            batch_first=True
        )
        self.cross_attn = torch.nn.TransformerEncoder(enc_layer, num_layers=cp_depth)
        self.film = Linear(bottleneck, bottleneck * 2)
    def forward(self, x_in, noise_labels, patch_coords, class_labels=None, augment_labels=None):
        B, K, C, H, W = x_in.shape
        x = x_in.reshape(B * K, C, H, W)

        emb = self.map_noise(noise_labels.reshape(B, 1).repeat(1, K).reshape(B * K))
        if self.map_label is not None and class_labels is not None:
            emb = emb + self.map_label(class_labels).repeat_interleave(K, dim=0)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        skips = []
        for block in self.enc.values():
            x = block(x, emb) if _is_unet_block(block) else block(x)
            skips.append(x)

        D, h, w = x.shape[1], x.shape[2], x.shape[3]
        tokens = x.reshape(B, K, D, h, w).mean(dim=(-1, -2))
        tokens = tokens + self.coord_proj(patch_coords.to(tokens.dtype))
        noise_tok = self.noise_proj(emb.reshape(B, K, -1).mean(dim=1)).unsqueeze(1)
        tokens = tokens + noise_tok
        tokens = self.cross_attn(self.token_ln(tokens))

        gamma, beta = self.film(tokens.reshape(B * K, D)).chunk(2, dim=1)
        x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        for block in self.dec.values():
            if _is_unet_block(block):
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
            else:
                x = block(x)

        return x.reshape(B, K, x.shape[1], x.shape[2], x.shape[3])

@persistence.persistent_class
class CrossPatch_EDMPrecond(torch.nn.Module):
    def __init__(self, img_resolution=64,
                 img_channels=4,
                 out_channels=2,
                 label_dim=0,
                 use_fp16=False,
                 sigma_data=0.5,
                 cp_patch_size=64,
                 cp_num_heads=4,
                 cp_ffn_mult=4,
                 cp_depth=2,
                 **model_kwargs):
        super().__init__()
        self.img_resolution = cp_patch_size
        self.img_channels = 2
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data

        model_kwargs = dict(model_kwargs)
        for k in ['img_channels', 'out_channels', 'use_fp16', 'hash_channels', 'model_type', 'cp_patch_size']:
            model_kwargs.pop(k, None)

        self.model = CrossPatchSongUNet(
            img_resolution=cp_patch_size,
            in_channels=4,
            out_channels=2,
            label_dim=label_dim,
            cp_num_heads=cp_num_heads,
            cp_depth=cp_depth,
            cp_ffn_mult=cp_ffn_mult,
            **model_kwargs,
        )

    def forward(self, x, sigma, x_pos=None, patch_coords=None, class_labels=None, force_fp32=False, **kwargs):
        assert x.ndim == 5
        sigma = sigma.reshape(x.shape[0], 1, 1, 1, 1).to(x.dtype)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log().reshape(x.shape[0]) / 4
        x_in = torch.cat([c_in * x, x_pos.to(x.dtype)], dim=2)
        F_x = self.model(x_in, c_noise, patch_coords=patch_coords, class_labels=class_labels)
        return c_skip * x + c_out * F_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

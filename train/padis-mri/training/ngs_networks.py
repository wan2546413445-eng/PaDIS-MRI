"""Independent NGS EDM model; original training.networks is never edited.
SongUNet forward follows PaDIS-MRI main-WSY, with NGS added at the bottleneck.
"""
import numpy as np
import torch
from torch.nn.functional import silu
from torch_utils import persistence
from training.networks import SongUNet, UNetBlock
from training.nonlocal_group_shrinkage import NonlocalGroupShrinkage


@persistence.persistent_class
class NGSEDMPrecond(torch.nn.Module):
    def __init__(self, img_resolution, img_channels, out_channels=None, label_dim=0,
                 use_fp16=False, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5,
                 model_type='SongUNet', four_channels=1, hash_channels=1, **model_kwargs):
        super().__init__()
        if model_type != 'SongUNet' or img_channels != 4 or out_channels not in (None, 2):
            raise ValueError('NGS expects the complex PaDIS SongUNet with 4 input / 2 output channels')
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.out_channels = 2
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min, self.sigma_max, self.sigma_data = sigma_min, sigma_max, sigma_data
        self.four_channels, self.hash_channels = four_channels, hash_channels
        self.use_global_fusion = True
        self.use_ngs = True
        self.model = SongUNet(img_resolution=img_resolution, in_channels=img_channels,
                              out_channels=2, label_dim=label_dim, **model_kwargs)
        channels = model_kwargs.get('model_channels', 128) * model_kwargs.get('channel_mult', [1,2,2,2])[-1]
        self.ngs = NonlocalGroupShrinkage(channels)

    def prepare_global(self, full, sigma):
        sigma = torch.as_tensor(sigma, device=full.device, dtype=torch.float32).reshape(-1,1,1,1)
        if full.shape[0] == 1:
            sigma = sigma[:1]
        if sigma.shape[0] not in (1, full.shape[0]):
            raise ValueError('full context and sigma batch mismatch')
        return self.ngs.encode(full.float() / (self.sigma_data ** 2 + sigma.square()).sqrt())

    def _song_forward(self, x, noise_labels, class_labels, augment_labels, context, coords):
        # Original SongUNet mapping and encoder/decoder, unchanged except one residual.
        m = self.model
        local = x[:, :2]
        emb = m.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        if m.map_label is not None:
            tmp = class_labels
            if self.training and m.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0],1],device=x.device) >= m.label_dropout).to(tmp.dtype)
            emb = emb + m.map_label(tmp * np.sqrt(m.map_label.in_features))
        if m.map_augment is not None and augment_labels is not None:
            emb = emb + m.map_augment(augment_labels)
        emb = silu(m.map_layer0(emb))
        emb = silu(m.map_layer1(emb))
        skips, aux = [], x
        for name, block in m.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x) if name.endswith("_conv") else block(x, emb)
                skips.append(x)
        if local.shape[-2:] == (64,64) and context is not None:
            x = x + self.ngs(local, context, coords, noise_labels, x.shape[-2:]).to(x.dtype)
        else:
            x = self.ngs.inactive(x)
        aux, tmp = None, None
        for name, block in m.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

    def forward(self, x, sigma, x_pos=None, class_labels=None, force_fp32=False,
                augment_labels=None, full_noisy=None, full_noisy_sigma=None,
                patch_coords=None, prepared_global=None, global_batch_indices=None):
        x = x.float()
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1,1,1,1)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 and x.is_cuda else torch.float32
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros([x.shape[0],self.label_dim],device=x.device)
        else:
            class_labels = class_labels.float().reshape(-1,self.label_dim)
        c_skip = self.sigma_data**2 / (sigma.square()+self.sigma_data**2)
        c_out = sigma*self.sigma_data / (sigma.square()+self.sigma_data**2).sqrt()
        c_in = (sigma.square()+self.sigma_data**2).rsqrt()
        c_noise = sigma.log().flatten()/4
        if x_pos is None:
            raise ValueError('PaDIS requires position maps')
        x_in = torch.cat([c_in*x, x_pos.float()],dim=1).to(dtype)
        context = prepared_global
        if full_noisy is not None:
            context = self.prepare_global(full_noisy, sigma if full_noisy_sigma is None else full_noisy_sigma)
        if context is not None and global_batch_indices is not None and context.shape[0] != 1:
            context = context.index_select(0,global_batch_indices.long())
        if context is not None and patch_coords is None:
            raise ValueError('NGS context requires patch coordinates')
        f = self._song_forward(x_in,c_noise,class_labels,augment_labels,context,patch_coords)
        return c_skip*x+c_out*f.float()

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

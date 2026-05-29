import os, sys, time, copy, json, pickle, re
import numpy as np
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats, misc
from .cross_patch_loss import CrossPatch_EDMLoss
from .dataset import ImageFolderDatasetX

def training_loop(run_dir='.', dataset_kwargs={}, data_loader_kwargs={}, network_kwargs={}, loss_kwargs={}, optimizer_kwargs={}, augment_kwargs=None,
    seed=0, batch_size=1, batch_gpu=1, total_kimg=200000, ema_halflife_kimg=500, ema_rampup_ratio=0.05, lr_rampup_kimg=10000,
    loss_scaling=1, kimg_per_tick=50, snapshot_ticks=50, state_dump_ticks=500, resume_pkl=None, resume_state_dump=None, resume_kimg=0,


    cudnn_benchmark=True, pad_width=0, device=torch.device('cuda'), cp_patch_size=64, cp_k=8, cp_local_k=3, cp_global_k=4, cp_debug=False, patch_list=None, patch_probs=None, **kwargs):
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark

    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu

    mypath = dataset_kwargs.get('path') + '/noisy.pt'
    dataset_obj = ImageFolderDatasetX(mypath, 384 + 2 * pad_width, pad=pad_width, channels=2, imsize=384)
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    net = dnnlib.util.construct_class_by_name(**network_kwargs, img_resolution=cp_patch_size, img_channels=4, out_channels=2, label_dim=dataset_obj.label_dim)
    net.train().requires_grad_(True).to(device)

    # Load network weights from a snapshot. This is used both for weight-only
    # transfer and as the EMA initialization when --resume points to a
    # training-state-xxxxxx.pt file.
    if resume_pkl is not None:
        if not os.path.isfile(resume_pkl):
            raise FileNotFoundError(f'resume_pkl not found: {resume_pkl}')
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        with open(resume_pkl, 'rb') as f:
            resume_data = pickle.load(f)
        src = resume_data['ema'] if 'ema' in resume_data else resume_data['net']
        misc.copy_params_and_buffers(src_module=src, dst_module=net, require_all=False)
        del resume_data, src
        if resume_kimg == 0:
            match = re.search(r'network-snapshot-(\d+)\.pkl$', os.path.basename(resume_pkl))
            if match is not None:
                resume_kimg = int(match.group(1))
        dist.print0(f'Network weights loaded. resume_kimg={resume_kimg}')

    if patch_list is None:
        patch_list = [16, 32, 64]
    if patch_probs is None:
        patch_probs = [0.2, 0.3, 0.5]

    loss_kwargs = dict(loss_kwargs)
    loss_kwargs.update(cp_k=cp_k, cp_local_k=cp_local_k, cp_global_k=cp_global_k, cp_patch_size=cp_patch_size, cp_debug=cp_debug)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    assert isinstance(loss_fn, CrossPatch_EDMLoss)

    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    augment_pipe = None if augment_kwargs is None else dnnlib.util.construct_class_by_name(**augment_kwargs)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Full training-state resume: restore optimizer state and raw net weights.
    # cross_train.py maps --resume training-state-xxxxxx.pt to both
    # resume_state_dump and the matching network-snapshot-xxxxxx.pkl.
    if resume_state_dump is not None:
        if not os.path.isfile(resume_state_dump):
            raise FileNotFoundError(f'resume_state_dump not found: {resume_state_dump}')
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        state_data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=state_data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(state_data['optimizer_state'])
        del state_data
        dist.print0('Training state loaded.')

    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    start_time = time.time()

    while True:
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                patch_size = int(np.random.choice(patch_list, p=patch_probs))
                batch_mul = 1
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32)
                labels = labels.to(device)

                loss = loss_fn(net=ddp, images=images, patch_size=patch_size, resolution=dataset_obj.resolution, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total / cp_k).backward()

        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        cur_nimg += int(batch_size)
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        if dist.get_rank() == 0 and (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=copy.deepcopy(ema).eval().requires_grad_(False), loss_fn=loss_fn)
            with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                pickle.dump(data, f)

        if dist.get_rank() == 0 and (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        if done:
            break

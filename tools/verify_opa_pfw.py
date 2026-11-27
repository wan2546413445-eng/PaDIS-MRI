"""CPU regression against this checkout's actual OPA, including raw sample identity."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "train/padis-mri")]
from training.patch_overlap_independent_loss import IndependentNoiseOverlapPatch_EDMLoss
from training.opa_pfw_loss import OPAPFWLoss, IndexedDataset, fractions, load_integral


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, 1)

    def forward(self, noisy, sigma, x_pos, **kwargs):
        return self.conv(torch.cat((noisy, x_pos), dim=1)) / (1 + sigma)


def run():
    torch.manual_seed(112)
    torch.set_num_threads(1)
    pad = 64
    x = torch.randn(3, 80, 96, dtype=torch.complex64)
    mask = x.abs() > 0.9
    padded = F.pad(mask, (pad,)*4)
    table = F.pad(padded.cumsum(1).cumsum(2), (1, 0, 1, 0)).int()
    with TemporaryDirectory() as tmp:
        data, cache = Path(tmp)/"noisy.pt", Path(tmp)/"cache.pt"
        torch.save(dict(x_est_gt=x), data)
        torch.save(dict(integral=table, padding=pad), cache)
        ds = IndexedDataset(str(data), 208, pad=pad)
        table = load_integral(cache, ds)
        # Exercise the dataset's real image flip and shuffle/index mapping.
        ds._xflip[1] = 1
        images, labels, ids, flips = next(iter(torch.utils.data.DataLoader(
            ds, batch_size=3, sampler=[2, 0, 1])))
    assert ids.tolist() == [2, 0, 1]
    for s in (16, 32, 64):
        for row, col in [(0, 0), (pad-7, pad+5), (pad+7, pad+13)]:
            coords = torch.full((3,), row), torch.full((3,), col)
            ratio = fractions(table, ids, flips, *coords, s)
            direct = []
            for idx, flip in zip(ids, flips):
                m = padded[idx].flip([1]) if flip else padded[idx]
                direct.append(m[row:row+s, col:col+s].sum())
            assert torch.equal(ratio*s*s, torch.stack(direct))

    baseline = IndependentNoiseOverlapPatch_EDMLoss(lambda_overlap=1.25, noise_mode="independent",
        alignment_mode="symmetric", pair_geometry="identical", spatial_weight_mode="uniform")
    net = Probe()
    # Deliberately construct full, empty and 50% coverage at every even patch size.
    masks = torch.zeros_like(padded)
    masks[2] = True
    masks[1, :, ::2] = True
    cache = F.pad(masks.cumsum(1).cumsum(2), (1, 0, 1, 0)).int()
    for s in (16, 32, 64):
        args = dict(net=net, images=images, patch_size=s, resolution=images.shape[-2],
                    labels=labels, augment_pipe=None)
        torch.manual_seed(123)
        original = baseline(**args)
        state = torch.random.get_rng_state()
        for limit in (1., 2.):
            torch.manual_seed(123)
            output = OPAPFWLoss(limit)(**args, integral=cache, raw_idx=ids, flipped=flips)
            assert torch.equal(state, torch.random.get_rng_state())
            weights = torch.tensor([limit, 1., (limit+1)/2]).view(3, 1, 1, 1)
            expected = original * weights
            torch.testing.assert_close(output, expected)
            reference = torch.autograd.grad(expected.sum(), net.parameters(), retain_graph=True)
            actual = torch.autograd.grad(output.sum(), net.parameters())
            for a, e in zip(actual, reference):
                torch.testing.assert_close(a, e)
            if limit == 2:
                assert not torch.isclose(output.mean(), original.mean()*weights.mean())
            print(f"P={s} Wmax={limit}: loss, gradients and RNG passed")
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    before = net.conv.weight.detach().clone()
    loss = OPAPFWLoss(2.)(**args, integral=cache, raw_idx=ids, flipped=flips)
    loss.sum().backward()
    opt.step()
    assert not torch.equal(before, net.conv.weight)
    print("Shuffled indices, flipped windows, padding and optimizer step passed")


if __name__ == "__main__":
    run()

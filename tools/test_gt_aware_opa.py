"""Run from the repository root with python tools/test_gt_aware_opa.py."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'train/padis-mri')]
from training.patch_overlap_independent_loss import IndependentNoiseOverlapPatch_EDMLoss as Loss, gt_weight


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([0.7, 0.9]).reshape(1, 2, 1, 1))

    def forward(self, x, sigma, **kwargs):
        self.prediction = x * self.scale
        self.sigma = sigma
        return self.prediction


def evaluate(cls, images, size, lam, **options):
    net = Probe()
    fn = cls(lambda_overlap=lam, alignment_mode='symmetric', pair_geometry='identical',
             spatial_weight_mode='uniform', **options)
    torch.manual_seed(123)
    loss = fn(net, images, size, images.shape[-1])
    grad, = torch.autograd.grad(loss.sum(), net.scale)
    return loss.detach(), grad


def test_gt_opa():
    a = torch.tensor([.16, .18, .20, .24, .20], requires_grad=True).reshape(1, 1, 1, 5)
    b = torch.tensor([.24, .26, .28, .32, .20], requires_grad=True).reshape_as(a)
    gt = torch.full_like(a, .2)
    w = gt_weight(a, b, gt)
    torch.testing.assert_close(w.flatten(), torch.tensor([1., .8, .5, .2, 0.]))
    assert not w.requires_grad and ((w >= 0) & (w <= 1)).all()
    torch.testing.assert_close(w, gt_weight(b, a, gt), rtol=0, atol=0)
    aux = w * (a - b).square()
    assert aux.flatten()[-1] == 0
    for term in (aux, ((a - gt).square() + (b - gt).square()) / 2):
        for gradient in torch.autograd.grad(term.sum(), (a, b)):
            assert gradient.isfinite().all() and gradient.abs().sum() > 0

    images = torch.full((3, 2, 96, 96), .2)
    net = Probe()
    fn = Loss(lambda_overlap=1., alignment_mode='symmetric', pair_geometry='identical',
              spatial_weight_mode='uniform', consistency_weight='gt_aware')
    torch.manual_seed(123)
    actual = fn(net, images, 64, 96)
    a, b = net.prediction.chunk(2)
    center = ((a + b) / 2 - .2).square().mean(1, keepdim=True)
    spread = ((a - b) / 2).square().mean(1, keepdim=True)
    gate = (spread / (center + spread + 1e-12 / 4)).detach()
    sigma = net.sigma.chunk(2)[0]
    weight = (sigma.square() + .25) / (.5 * sigma).square()
    expected = weight * ((a - .2).square() + (b - .2).square()) / 2
    expected = expected + (weight * gate * (a - b).square()).mean((1, 2, 3), keepdim=True)
    torch.testing.assert_close(actual, expected)
    for loss_grad, ref_grad in zip(torch.autograd.grad(actual.mean(), net.prediction, retain_graph=True),
                                   torch.autograd.grad(expected.mean(), net.prediction)):
        torch.testing.assert_close(loss_grad, ref_grad)

    source = subprocess.check_output(['git', 'show',
        '2d39f8e:train/padis-mri/training/patch_overlap_independent_loss.py'], cwd=ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'original_loss.py'
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('original_loss', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for size in (16, 32, 64):
            for lam in (0., 1.):
                reference = evaluate(module.IndependentNoiseOverlapPatch_EDMLoss, images, size, lam)
                modes = ['uniform', 'gt_aware'] if size != 64 or lam == 0 else ['uniform']
                for mode in modes:
                    observed = evaluate(Loss, images, size, lam, consistency_weight=mode)
                    for x, y in zip(observed, reference):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
    print('PASS: five weights, symmetry, both gradients, two-channel GT objective and exact regressions')


if __name__ == '__main__':
    test_gt_opa()

"""Manual CPU-only verification: python tests/test_opa_agga_cpu.py -v."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'train/padis-mri'), str(ROOT)]
import torch
from training import agg_sampler
from training import patch_opa_agga_loss as new
from training import patch_overlap_independent_loss as old
from training.patch_loss import Patch_EDMLoss
from training.patch_overlap_loss import _extract_patches, _position_maps

spec = importlib.util.spec_from_file_location('opa_agga_entry', ROOT / 'train/padis-mri/train_opa_agga.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
torch.set_num_threads(1)


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.31))
        self.calls = []

    def forward(self, x, sigma, x_pos, class_labels=None, augment_labels=None):
        self.calls.append((x.detach().clone(), sigma.detach().clone(),
                           x_pos.detach().clone(), class_labels))
        return self.scale * x + 0.07 * x_pos[:, :1] + 0.03 * sigma


def make_images(batch=3, size=101):
    return torch.linspace(-1, 1, batch * 2 * size * size).reshape(batch, 2, size, size)


class OPAAGGATests(unittest.TestCase):
    def test_pair_content_position_sigma_and_noise(self):
        images = make_images()
        top, left = torch.tensor([0, 17, 37]), torch.tensor([37, 3, 0])
        clean = _extract_patches(images, top, left, 64)
        net = TinyDenoiser()
        labels = torch.arange(3).view(3, 1)
        with patch.object(new, 'sample_single_locations', return_value=(top, left)) as sampler:
            torch.manual_seed(16)
            loss = new.OPAAGGAPatchEDMLoss()(net, images, 64, 101, labels)
        sampler.assert_called_once()
        self.assertEqual(len(net.calls), 1)
        noisy, sigma, pos, doubled_labels = net.calls[0]
        self.assertEqual(noisy.shape, (6, 2, 64, 64))
        self.assertEqual(loss.shape, clean.shape)
        torch.testing.assert_close(sigma[:3], sigma[3:], rtol=0, atol=0)
        torch.testing.assert_close(pos[:3], pos[3:], rtol=0, atol=0)
        torch.testing.assert_close(pos[:3], _position_maps(top, left, 3, 64, 101, images.device))
        torch.testing.assert_close(doubled_labels, torch.cat([labels, labels]))
        e1, e2 = (noisy[:3] - clean) / sigma[:3], (noisy[3:] - clean) / sigma[3:]
        self.assertFalse(torch.equal(e1, e2))
        self.assertLess(abs(torch.corrcoef(torch.stack([e1.flatten(), e2.flatten()]))[0, 1].item()), 0.03)
        self.assertAlmostEqual(e1.std().item(), 1, delta=0.03)
        self.assertAlmostEqual(e2.std().item(), 1, delta=0.03)

    def test_loss_and_parameter_gradients_match_current_opa(self):
        images = make_images(batch=2)
        top, left = torch.tensor([0, 31]), torch.tensor([37, 4])
        geometry = (top, left, top, left, torch.tensor([True, False]), 32)
        for weight in [0.0, 0.3, 1.0]:
            for seed in [123, 456]:
                with self.subTest(weight=weight, seed=seed):
                    a, b = TinyDenoiser(), TinyDenoiser()
                    original = old.IndependentNoiseOverlapPatch_EDMLoss(
                        lambda_overlap=weight, pair_geometry='identical',
                        spatial_weight_mode='uniform', alignment_mode='symmetric')
                    current = new.OPAAGGAPatchEDMLoss(opa_weight=weight)
                    with patch.object(old, '_sample_overlap_geometry', return_value=geometry):
                        torch.manual_seed(seed)
                        loss_a = original(a, images, 64, 101)
                    with patch.object(new, 'sample_single_locations', return_value=(top, left)):
                        torch.manual_seed(seed)
                        loss_b = current(b, images, 64, 101)
                    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
                    loss_a.sum().backward()
                    loss_b.sum().backward()
                    torch.testing.assert_close(a.scale.grad, b.scale.grad, rtol=0, atol=0)
                    for x, y in zip(a.calls[0][:3], b.calls[0][:3]):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_agga_is_used_at_all_scales_and_small_scales_keep_single_view(self):
        images = make_images(batch=2)
        for size in [16, 32, 64]:
            with self.subTest(patch_size=size):
                top, left = torch.tensor([0, 3]), torch.tensor([5, 1])
                clean = _extract_patches(images, top, left, size)
                expected_pos = _position_maps(top, left, 2, size, 101, images.device)
                net = TinyDenoiser()
                with patch.object(new, 'sample_single_locations', return_value=(top, left)) as sampler:
                    torch.manual_seed(123)
                    result = new.OPAAGGAPatchEDMLoss(rho=0.65, gamma=1.0)(
                        net, images, size, 101)
                sampler.assert_called_once_with(images, size, 0.65, 1.0, score_mode='multiply')
                self.assertEqual(result.shape, clean.shape)
                self.assertEqual(len(net.calls), 1)
                actual_noisy, actual_sigma, actual_pos, _ = net.calls[0]
                torch.testing.assert_close(actual_pos[:2], expected_pos)
                self.assertEqual(actual_noisy.shape[0], 4 if size == 64 else 2)
                self.assertEqual(actual_sigma.shape[0], 4 if size == 64 else 2)
                result.sum().backward()
                self.assertTrue(torch.isfinite(net.scale.grad))
                if size != 64:
                    # With an identical predetermined crop, preserve baseline EDM exactly.
                    reference = TinyDenoiser()
                    original = Patch_EDMLoss()
                    with patch.object(original, 'pachify', return_value=(clean, expected_pos)):
                        torch.manual_seed(123)
                        baseline = original(reference, images, size, 101)
                    torch.testing.assert_close(result, baseline, rtol=0, atol=0)
                    baseline.sum().backward()
                    torch.testing.assert_close(net.scale.grad, reference.scale.grad, rtol=0, atol=0)

    def test_rho_only_changes_sampling_probabilities(self):
        images = torch.zeros(1, 2, 80, 83)
        images[:, :, 30:60, 25:55] = 1
        probabilities = []
        actual = torch.multinomial
        for rho in [0.0, 0.65]:
            with patch.object(agg_sampler.torch, 'multinomial',
                              side_effect=lambda p, n: (probabilities.append(p.clone()), actual(p, n))[1]):
                agg_sampler.sample_single_locations(images, 64, rho, 1.0)
        p0, p1 = probabilities
        self.assertEqual(p0.numel(), 17 * 20)
        torch.testing.assert_close(p0, torch.full_like(p0, 1 / p0.numel()))
        support, grad = agg_sampler._structure_maps(images[0])
        score = agg_sampler._region_score(support, grad, (64, 64), 1.0).flatten()
        torch.testing.assert_close(p1, 0.35 * p0 + 0.65 * score / score.sum())
        self.assertFalse(torch.equal(p0, p1))
        self.assertTrue(bool((p1 >= 0.35 / p0.numel()).all()))
        # With equal positions and random draws, rho has no effect on loss or RNG consumption.
        top, left = torch.tensor([16]), torch.tensor([19])
        outputs = []
        for rho in [0.0, 0.65]:
            net = TinyDenoiser()
            with patch.object(new, 'sample_single_locations', return_value=(top, left)):
                torch.manual_seed(123)
                loss = new.OPAAGGAPatchEDMLoss(rho=rho)(net, images, 64, 80)
                loss.sum().backward()
                outputs.append((loss.detach(), net.scale.grad, torch.get_rng_state()))
        for a, b in zip(*outputs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_complete_single_patch_domain_and_corners(self):
        images = torch.ones(1, 2, 70, 73)
        for rho in [0.0, 0.65]:
            for index, expected in [(0, (0, 0)), (9, (0, 9)), (60, (6, 0)), (69, (6, 9))]:
                def choose(probabilities, count):
                    self.assertEqual(probabilities.numel(), 7 * 10)
                    self.assertTrue(bool((probabilities > 0).all()))
                    return torch.tensor([index])
                with patch.object(agg_sampler.torch, 'multinomial', side_effect=choose):
                    top, left = agg_sampler.sample_single_locations(images, 64, rho, 1.0)
                self.assertEqual((top.item(), left.item()), expected)
        # 64x64 cannot support the old half-patch geometry, but is valid here.
        loss = new.OPAAGGAPatchEDMLoss()(TinyDenoiser(), torch.ones(1, 2, 64, 64), 64, 64)
        self.assertEqual(loss.shape, (1, 2, 64, 64))
        for rho in [0.0, 0.65]:
            top, left = agg_sampler.sample_single_locations(torch.zeros(2, 2, 64, 64), 64, rho, 1.0)
            self.assertTrue(bool((top == 0).all() and (left == 0).all()))

    def test_checkpoint_roundtrip_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'snapshot.pkl'
            with checkpoint.open('wb') as f:
                pickle.dump(dict(ema=torch.nn.Identity(),
                                 loss_fn=new.OPAAGGAPatchEDMLoss(opa_weight=0.7, rho=0.2, gamma=0.4),
                                 augment_pipe=None, dataset_kwargs={}), f)
            code = """
import pickle, sys, torch
torch.set_num_threads(1)
with open(sys.argv[1], 'rb') as f: data = pickle.load(f)
loss = data['loss_fn']
assert (loss.opa_weight, loss.rho, loss.gamma) == (0.7, 0.2, 0.4)
def net(x, sigma, **kw): return 0.5 * x
for size in (16, 32, 64):
    value = loss(net, torch.ones(2, 2, 70, 70), size, 70)
    assert value.shape == (2, 2, size, size) and torch.isfinite(value).all()
print('fresh-process checkpoint callable')
"""
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'train/padis-mri'), str(ROOT)]))
            result = subprocess.run([sys.executable, '-c', code, str(checkpoint)], env=env,
                                    cwd=tmp, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_dryrun_without_site_packages_or_data_or_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / 'must_not_exist'
            result = subprocess.run([sys.executable, '-S', str(ROOT / 'train/padis-mri/train_opa_agga.py'),
                                     '--mode', 'dryrun', '--data', '/unavailable/no-data',
                                     '--outdir', str(outdir)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.JSONDecoder().raw_decode(result.stdout)[0]
            self.assertEqual(config['total_kimg'], 5000)
            self.assertEqual(config['patch_list'], [16, 32, 64])
            self.assertFalse(outdir.exists())
            self.assertNotIn('torch', sys.modules.get('opa_agga_entry', types.SimpleNamespace()).__dict__)

    def test_cli_persisted_config_and_loss_constructor(self):
        # Exercise the real entry and JSON writer; replace only GPU startup and training loop.
        import dnnlib
        from torch_utils import distributed as dist
        captured = {}
        fake_loop = types.ModuleType('training.training_loop')
        def receive(**config):
            captured.update(config)
            loss = dnnlib.util.construct_class_by_name(**config['loss_kwargs'])
            self.assertEqual((loss.opa_weight, loss.rho, loss.gamma, loss.score_mode), (0.4, 0.2, 0.8, 'add'))
        fake_loop.training_loop = receive
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'data'
            data.mkdir()
            (data / 'noisy.pt').touch()  # Never loaded by this mocked launch test.
            args = ['--mode', 'train', '--data', str(data), '--outdir', str(Path(tmp) / 'runs'),
                    '--opa-weight', '0.4', '--rho', '0.2', '--gamma', '0.8', '--score-mode', 'add']
            with patch.dict(sys.modules, {'training.training_loop': fake_loop}), \
                 patch('training.training_loop', fake_loop, create=True), \
                 patch.object(dist, 'init') as init, \
                 patch.object(torch.multiprocessing, 'set_start_method'), \
                 patch.object(dnnlib.util, 'Logger'), contextlib.redirect_stdout(io.StringIO()):
                entry.main(args)
            init.assert_called_once()
            saved = json.loads((Path(captured['run_dir']) / 'training_options.json').read_text())
            self.assertEqual(saved['loss_kwargs'], captured['loss_kwargs'])
            self.assertEqual(saved['opa_agga']['reference_sha'], entry.REFERENCE_SHA)
            self.assertEqual(saved['total_kimg'], 5000)
            self.assertEqual((saved['batch_size'], saved['batch_gpu']), (4, 1))
            self.assertNotIn('resume_pkl', saved)
            self.assertEqual(saved['optimizer_kwargs']['lr'], 1e-4)

    def test_matched_configs_and_original_counter(self):
        a = entry.build_config(entry.parser().parse_args(['--data', '/unused', '--rho', '0']))
        b = entry.build_config(entry.parser().parse_args(['--data', '/unused', '--rho', '0.65']))
        a['loss_kwargs']['rho'] = b['loss_kwargs']['rho']
        self.assertEqual(a, b)
        # Original loop is used directly; dual-view batch size never enters its counter.
        source = (ROOT / 'train/padis-mri/training/training_loop.py').read_text()
        self.assertIn('cur_nimg += int(batch_size * batch_mul_avg)', source)
        self.assertIn('loss.sum().mul(loss_scaling / batch_gpu_total / batch_mul).backward()', source)
        self.assertIn('ema_beta = 0.5 ** (batch_size * batch_mul_avg', source)

    def test_resume_transfer_and_invalid_configuration(self):
        p = entry.parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p.parse_args(['--data', '/unused', '--resume', 'x', '--transfer', 'y'])
        c = entry.build_config(p.parse_args(['--data', '/unused', '--resume', '/tmp/training-state-001000.pt']))
        self.assertEqual(c['resume_kimg'], 1000)
        self.assertEqual(c['total_kimg'], 5000)
        self.assertEqual(c['resume_pkl'], '/tmp/network-snapshot-001000.pkl')
        for option, value in [('--rho', '1'), ('--gamma', '-1'), ('--opa-weight', 'nan'), ('--total-kimg', '0')]:
            with self.subTest(option=option), self.assertRaises(ValueError):
                entry.build_config(p.parse_args(['--data', '/unused', option, value]))

    def test_shell_modes_and_explicit_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / 'capture_python'
            runner.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            runner.chmod(0o755)
            env = dict(os.environ, PYTHON=str(runner), GPU='2', DATA_DIR='/data/custom',
                       OUTDIR=str(Path(tmp) / 'runs'), OPA_WEIGHT='0.6', RHO='0', GAMMA='0.9',
                       SCORE_MODE='add', SEED='321', TOTAL_KIMG='5000')
            for mode in ['dryrun', 'smoke', 'train']:
                result = subprocess.run(['bash', str(ROOT / 'bash/train_opa_agga.sh'), mode],
                                        cwd=tmp, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(result.stdout)
                for flag, value in [('--mode', mode), ('--gpu', '2'), ('--data', '/data/custom'),
                                    ('--opa-weight', '0.6'), ('--rho', '0'), ('--gamma', '0.9'),
                                    ('--score-mode', 'add'), ('--seed', '321'), ('--total-kimg', '5000')]:
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertEqual('torch.distributed.run' in args, mode != 'dryrun')
            self.assertFalse((Path(tmp) / 'runs').exists())


if __name__ == '__main__':
    unittest.main()

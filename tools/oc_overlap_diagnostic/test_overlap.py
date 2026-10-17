#!/usr/bin/env python3
"""Tests use synthetic arrays only; no MRI experimental results are created."""
import ast
from copy import deepcopy
import importlib.util
from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import analyze_overlap as d

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def basic_config():
    return dict(dataset_size=7, sample_index=0, patch_sizes=[16, 32, 64],
                patch_probs=[.2, .3, .5], batch_size=4, batch_gpu=1,
                seed=123, padded_size=[96, 96], image_size=[64, 64], pad_width=16,
                modes=["uniform_e0", "agga"], overlap_threshold=.5,
                kimg_checkpoints=[.5, 1, 5, 10], snapshot_interval=100,
                configuration_source="SYNTHETIC UNIT TEST ONLY")


def uniform_cache(config):
    h, w = config["padded_size"]
    cache = {}
    for s in config["patch_sizes"]:
        p, cdf = d.cdf_from_probabilities(np.ones((h-s+1)*(w-s+1)))
        cache[s] = (p, cdf, w-s+1)
    return {m: cache for m in config["modes"]}


class GeometryTests(unittest.TestCase):
    def test_exact_half_is_included(self):
        index = d.OverlapIndex(160, 160, 64)
        self.assertEqual(index.observe(0, 0, 0), -1)
        self.assertEqual(index.observe(0, 32, 1), 0)

    def test_below_half_is_excluded(self):
        index = d.OverlapIndex(160, 160, 64)
        index.observe(0, 0, 0)
        self.assertEqual(index.observe(0, 33, 1), -1)

    def test_duplicate_is_overlap(self):
        index = d.OverlapIndex(128, 128, 64)
        index.observe(64, 64, 3)
        self.assertEqual(index.observe(64, 64, 4), 3)

    def test_cross_scale_not_compared(self):
        a, b = d.OverlapIndex(128, 128, 32), d.OverlapIndex(128, 128, 64)
        a.observe(0, 0, 0)
        self.assertEqual(b.observe(0, 0, 1), -1)

    def test_multiple_witnesses_only_one_count(self):
        index = d.OverlapIndex(128, 128, 64)
        index.observe(0, 0, 0)
        index.observe(0, 1, 1)
        before = index.overlap
        self.assertGreaterEqual(index.observe(1, 0, 2), 0)
        self.assertEqual(index.overlap - before, 1)

    def test_no_combining_intersections(self):
        index = d.OverlapIndex(200, 200, 64)
        # Each contributes 31*64 to the middle rectangle, their union is 62*64.
        index.observe(50, 0, 0)
        index.observe(50, 66, 1)
        self.assertEqual(index.observe(50, 33, 2), -1)

    def test_diagonal_area(self):
        self.assertEqual(d.intersection_area((0, 0), (16, 16), 64), 48*48)
        index = d.OverlapIndex(128, 128, 64)
        index.observe(0, 0, 0)
        self.assertGreaterEqual(index.observe(16, 16, 1), 0)
        index = d.OverlapIndex(128, 128, 64)
        index.observe(0, 0, 0)
        self.assertEqual(index.observe(20, 20, 1), -1)

    def test_all_boundaries_and_outside(self):
        for point in [(0, 0), (0, 16), (16, 0), (16, 16)]:
            with self.subTest(point=point):
                index = d.OverlapIndex(80, 80, 64)
                self.assertEqual(index.observe(*point, 0), -1)
                self.assertEqual(index.observe(*point, 1), 0)
        for point in [(-1, 0), (0, -1), (17, 0), (0, 17)]:
            with self.assertRaises(ValueError):
                d.OverlapIndex(80, 80, 64).observe(*point, 0)

    def test_exact_index_against_naive(self):
        rng = np.random.default_rng(42)
        for s in [16, 32, 64]:
            for threshold in [.1, .5, .75, 1.0]:
                with self.subTest(size=s, threshold=threshold):
                    index = d.OverlapIndex(130, 145, s, threshold)
                    points = rng.integers([0, 0], [131-s, 146-s], (300, 2))
                    history = []
                    for i, point in enumerate(points):
                        matching = [j for j, previous in enumerate(history)
                                    if d.intersection_area(point, previous, s) >= index.required]
                        witness = index.observe(*point, i)
                        self.assertEqual(witness >= 0, bool(matching))
                        if witness >= 0:
                            self.assertIn(witness, matching)
                        history.append(point)

    def test_all_query_sites_after_single_rectangle(self):
        for s in [3, 4, 16]:
            index = d.OverlapIndex(32, 35, s)
            index.observe(8, 9, 0)
            for y in range(33-s):
                for x in range(36-s):
                    self.assertEqual(index.witness[y, x] >= 0,
                                     2*d.intersection_area((8, 9), (y, x), s) >= s*s)

    def test_zero_denominators(self):
        row = d.summarize({s: d.OverlapIndex(96, 96, s) for s in [16, 32, 64]})
        self.assertTrue(all(row[k] is None for k in d.METRICS))

    def test_invalid_geometry(self):
        for s, t in [(0, .5), (100, .5), (16, 0), (16, 1.1)]:
            with self.assertRaises(ValueError):
                d.OverlapIndex(96, 96, s, t)


class ScheduleTests(unittest.TestCase):
    def test_indices_equal_actual_infinite_sampler(self):
        tree = ast.parse((d.ROOT / "torch_utils/misc.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InfiniteSampler")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__iter__")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_misc_iter", "exec"), namespace)
        for size in [1, 2, 7, 200]:
            obj = SimpleNamespace(dataset=range(size), shuffle=True, seed=123,
                                  window_size=.5, rank=0, num_replicas=1)
            expected = list(islice(namespace["__iter__"](obj), 2000))
            self.assertEqual(list(islice(d.infinite_indices(size, 123), 2000)), expected)

    def test_real_default_counter_and_events(self):
        steps = list(d.schedule(200, 0, [16, 32, 64], [.2, .3, .5], 4, 1, 123, 1000, [512, 512]))
        self.assertEqual(len(steps), 9)
        self.assertEqual([s["cur_nimg"] for s in steps], [121*i for i in range(1, 10)])
        self.assertEqual(steps[-1]["cur_nimg"], 1089)
        # Independent schedule draw and actual InfiniteSampler stream.
        rng = np.random.RandomState(123)
        stream = d.infinite_indices(200, 123)
        total = 0
        expected = []
        for i in range(9):
            for r in range(4):
                s = int(rng.choice([16, 32, 64], p=[.2, .3, .5]))
                for offset in range(d.BATCH_MUL[s]):
                    total += 1
                    if next(stream) == 0:
                        expected.append((i+1, r, offset, total, s))
        actual = [(e["iteration"], e["accumulation_round"], e["round_offset"], e["global_patch_event"], e["patch_size"])
                  for step in steps for e in step["events"]]
        self.assertEqual(actual, expected)
        self.assertEqual(total, steps[-1]["total_patch_events"])

    def test_multiple_occurrences_one_round(self):
        steps = list(d.schedule(1, 0, [16], [1], 4, 1, 123, 1, [96, 96]))
        events = steps[0]["events"]
        self.assertEqual(len(events), 256)
        self.assertEqual([e["round_offset"] for e in events[:64]], list(range(64)))

    def test_shared_schedule_and_identical_counts(self):
        c = basic_config()
        result = d.simulate(c, uniform_cache(c))
        a, b = result["rows"]["uniform_e0"], result["rows"]["agga"]
        for x, y in zip(a, b):
            for key in ["cur_nimg", "total_patch_events", "schedule_sha256", "sample0_total_patches",
                        "sample0_n16", "sample0_n32", "sample0_n64", *d.METRICS]:
                self.assertEqual(x[key], y[key])

    def test_mode_order_and_standalone_dont_change_results(self):
        c = basic_config()
        combined = d.simulate(c, uniform_cache(c))
        for modes in [["agga"], ["uniform_e0"], ["agga", "uniform_e0"]]:
            separate = deepcopy(c)
            separate["modes"] = modes
            result = d.simulate(separate, uniform_cache(separate))
            for mode in modes:
                self.assertEqual(combined["events"][mode], result["events"][mode])
                self.assertEqual(combined["rows"][mode], result["rows"][mode])

    def test_continuous_prefix_and_checkpoint_crossing(self):
        c = basic_config()
        result = d.simulate(c, uniform_cache(c))
        for rows in result["rows"].values():
            for row in rows:
                self.assertGreaterEqual(row["actual_kimg"], row["target_kimg"])
                self.assertLess(row["actual_kimg"]-row["target_kimg"], .121 + 1e-12)
            for before, after in zip(rows, rows[1:]):
                for key in ["sample0_total_patches", "sample0_n16", "sample0_n32", "sample0_n64",
                            "sample0_overlap16", "sample0_overlap32", "sample0_overlap64"]:
                    self.assertGreaterEqual(after[key], before[key])
        short = deepcopy(c)
        short["kimg_checkpoints"] = [.5, 1]
        prefix = d.simulate(short, uniform_cache(short))
        for mode in c["modes"]:
            self.assertEqual(prefix["rows"][mode], result["rows"][mode][:2])
            self.assertEqual(prefix["events"][mode], result["events"][mode][:len(prefix["events"][mode])])

    def test_repeated_simulations_identical(self):
        c = basic_config()
        self.assertEqual(d.simulate(c, uniform_cache(c)), d.simulate(c, uniform_cache(c)))

    def test_metric_identities(self):
        c = basic_config()
        for rows in d.simulate(c, uniform_cache(c))["rows"].values():
            for r in rows:
                n = sum(r[f"sample0_n{s}"] for s in c["patch_sizes"])
                o = sum(r[f"sample0_overlap{s}"] for s in c["patch_sizes"])
                self.assertEqual(n, r["sample0_total_patches"])
                self.assertAlmostEqual(r["Rall"], 100*o/n)
                if r["sample0_n64"]:
                    self.assertAlmostEqual(r["R64_total"], r["R64"] * r["sample0_n64"]/n)


class DistributionTests(unittest.TestCase):
    def test_inverse_cdf_order_and_endpoints(self):
        p, cdf = d.cdf_from_probabilities([1, 1, 1, 1, 1, 1])
        u = (np.arange(6)+.5)/6
        self.assertEqual(d.draw_locations(cdf, 3, u).tolist(), [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]])
        self.assertEqual(d.draw_locations(cdf, 3, [0, np.nextafter(1., 0.)]).tolist(), [[0,0],[1,2]])

    def test_unequal_empirical_distribution(self):
        p, cdf = d.cdf_from_probabilities(np.arange(1, 1001))
        self.assertTrue(d.empirical_check(p, cdf, 123)["passed"])

    def test_invalid_probabilities(self):
        for p in [[0,0], [-1,2], [np.nan,1], [np.inf,1]]:
            with self.assertRaises(ValueError):
                d.cdf_from_probabilities(p)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable; these checks must pass before a real run")
class OriginalSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.torch = torch
        y, x = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")
        magnitude = torch.exp(-((x-48)**2+(y-48)**2)/300) * (1+.2*torch.sin(x.float()/3))
        cls.image = torch.stack([magnitude, magnitude*.3])

    def test_original_cache_exact_all_scales(self):
        cache, report = d.build_cache(self.image, [16,32,64], .65, 1., 123)
        for size in [16,32,64]:
            self.assertEqual(report[str(size)]["cache_source_max_abs_error"], 0)
            self.assertTrue(report[str(size)]["original_call_intercepted"])
            self.assertTrue(report[str(size)]["empirical_check"]["passed"])
            self.assertEqual(cache[size][0].size, (97-size)**2)
            self.assertAlmostEqual(cache[size][0].sum(), 1., places=13)
            self.assertGreater(float(cache[size][0].max()), float(cache[size][0].min()))

    def test_e0_original_probabilities_uniform(self):
        cache, _ = d.build_cache(self.image, [16,32,64], 0., 0., 123)
        for p, cdf, width in cache.values():
            np.testing.assert_allclose(p, np.full(p.size, 1/p.size), rtol=0, atol=1e-15)

    def test_zero_and_nonfinite_uniform_fallback(self):
        for image in [self.torch.zeros_like(self.image), self.torch.full_like(self.image, float("nan"))]:
            cache, report = d.build_cache(image, [64], .65, 1., 123)
            self.assertTrue(report["64"]["uniform_fallback"])
            p = cache[64][0]
            np.testing.assert_allclose(p, np.full(p.size, 1/p.size), rtol=0, atol=1e-15)

    def test_original_flat_index_mapping(self):
        module = d.load_agg_module()
        for size in [16,32,64]:
            width = 97-size
            def last(probabilities, *args, **kwargs):
                return self.torch.tensor([probabilities.numel()-1])
            with patch.object(self.torch, "multinomial", last):
                top, left = module.sample_single_locations(self.image[None], size, .65, 1.)
            self.assertEqual((int(top[0]), int(left[0])), (width-1, width-1))

    def test_real_sampler_empirical_small_canvas(self):
        module = d.load_agg_module()
        image = self.image[:, :8, :9]
        cache, _ = d.build_cache(image, [4], .65, 1., 123)
        self.torch.manual_seed(123)
        top, left = module.sample_single_locations(image[None].repeat(4000,1,1,1), 4, .65, 1.)
        observed = np.bincount((top*6+left).numpy(), minlength=30)/4000
        p = cache[4][0]
        self.assertTrue(np.all(np.abs(observed-p) <= 7*np.sqrt(p*(1-p)/4000)+7/4000))

    def test_dataset_preprocessing_matches_original_operations(self):
        raw = self.torch.complex(self.image[0][None], self.image[1][None]).repeat(3,1,1)
        with TemporaryDirectory(prefix="oc_SYNTHETIC_preprocessing_") as temp:
            path = Path(temp)/"synthetic.pt"
            self.torch.save({"x_est_gt":raw}, path)
            image, metadata = d.load_image(path, 0, 5)
            reference = self.torch.view_as_real(raw).numpy().astype(np.float32).transpose(0,3,1,2)
            reference = np.pad(reference, ((0,0),(0,0),(5,5),(5,5)))
            np.testing.assert_array_equal(image.numpy(), reference[0])
            self.assertEqual(metadata["dataset_size"], 3)
            self.assertEqual(metadata["padded_size"], [106,106])

    def test_synthetic_cli_outputs_and_provenance(self):
        with TemporaryDirectory(prefix="oc_SYNTHETIC_cli_") as temp:
            temp = Path(temp)
            raw = self.torch.complex(self.image[0][None], self.image[1][None]).repeat(3,1,1)
            path = temp/"synthetic.pt"
            self.torch.save({"x_est_gt":raw}, path)
            output = d.main(["--data",str(path),"--pad_width","0","--output_dir",str(temp/"results"),
                             "--kimg_checkpoints",".1",".2",".5","1",
                             "--configuration_source","SYNTHETIC AUTOMATED TEST ONLY"])
            for name in ["config.json","probability_checks.json","execution.json","REPORT.md",
                         "uniform_e0/checkpoints.csv","agga/checkpoints.csv","comparison/comparison.csv",
                         "comparison/R64_comparison.svg","comparison/R64_comparison.png",
                         "agga/sample0_events.csv","agga/actual_overlap_pair.svg","agga/coverage64.npy"]:
                self.assertTrue((output/name).is_file(), name)
            execution = json.loads((output/"execution.json").read_text())
            self.assertEqual(execution["source_before"]["source_sha256"], execution["source_after"]["source_sha256"])
            rows = json.loads((output/"agga/statistics.json").read_text())
            self.assertEqual(sum(r["is_formal_checkpoint"] for r in rows), 4)


class CLITests(unittest.TestCase):
    def test_defaults(self):
        args = d.parse_args([])
        self.assertEqual(args.kimg_checkpoints, [500,1000,5000,10000])
        self.assertEqual(args.modes, ["uniform_e0","agga"])

    def test_missing_data_produces_no_results(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)/"results"
            with self.assertRaises(FileNotFoundError):
                d.main(["--data",str(Path(temp)/"missing.pt"),"--output_dir",str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""CPU-only contract tests; no MRI files, checkpoints, CUDA or external services."""

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from eval_pula import recon_pula
from eval_pula.pa_balance import (PABalanceStats, VARIANTS, balanced_coefficients,
                                  edm_coefficients, edm_sigma_data, interior_window,
                                  region_expand, region_mean)
from eval_pula.recon_pula import PULASystem, reconstruct, whiten_measurement
from eval_pula.run import random_shifts
from eval_pula.trajectories import TrajectorySystems

spec = importlib.util.spec_from_file_location("legacy_recon", ROOT / "tests/fixtures/legacy_recon_pula.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


class CoilOperator:
    def __init__(self, mask, maps):
        self.mask, self.maps = mask, maps

    def forward(self, channels):
        image = torch.complex(channels[:, :1], channels[:, 1:])
        return self.mask * torch.fft.fft2(self.maps * image, norm="ortho")

    def adjoint(self, data):
        return (self.maps.conj() * torch.fft.ifft2(self.mask * data, norm="ortho")).sum(1, keepdim=True)


def fixture(dtype=torch.complex128, height=4, width=4):
    rng = torch.Generator().manual_seed(109)
    maps = torch.randn((1, 2, height, width), dtype=dtype, generator=rng)
    mask = torch.ones((1, 1, height, width), dtype=maps.real.dtype)
    mask[..., 1::2] = 0
    operator = CoilOperator(mask, maps)
    image = torch.randn((1, 1, height, width), dtype=dtype, generator=rng)
    measurement = mask * torch.randn(maps.shape, dtype=dtype, generator=rng)
    return operator, measurement, image


class Net(torch.nn.Module):
    sigma_data = .5

    def forward(self, state, sigma, shifts):
        cs, co, ci = edm_coefficients(sigma, self.sigma_data)
        z = ci * state
        rows = torch.arange(state.shape[-2], device=state.device, dtype=state.real.dtype)[None, None, :, None]
        cols = torch.arange(state.shape[-1], device=state.device, dtype=state.real.dtype)[None, None, None, :]
        shifts = state.real.new_tensor(shifts)
        offset = (.03 * shifts[:, 0] + .02 * shifts[:, 1])[:, None, None, None]
        residual = .2 * torch.complex(z.real.tanh(), z.imag.tanh()) + offset * (1 + .1 * rows + .2 * cols)
        return cs * state + co * residual


class Oracle:
    def __init__(self, height=4, pad=2):
        self.net = Net()
        self.pad = self.psize = pad
        self.image_size = height
        self.seen_shifts = []

    def denoise(self, state, sigma, shift):
        shifts = [shift] if isinstance(shift[0], int) else shift
        self.seen_shifts.append(shifts)
        return self.net(state, sigma, shifts)

    denoise_padded = denoise


def make_system(module=recon_pula, convention="padis_real", seed=123, height=4, dtype=torch.complex128):
    op, y, image = fixture(dtype=dtype, height=height, width=height)
    op, y = module.whiten_measurement(op, y, 2.0, convention)
    return module.PULASystem(op, y, torch.Generator().manual_seed(seed), 30, 1e-12, convention), image


def configure(system, variant, sigma=.7, gamma=.5, variance=None):
    # Initialization is identical for every variant; the shape cache is reused.
    system.initialization_rhs(1.)
    system.configure_update(variant, sigma, gamma, .5, variance)


class PABalancedTests(unittest.TestCase):
    def close(self, actual, expected, atol=1e-11, rtol=1e-10):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_01_baseline_operator_rhs_warm_rng_and_legacy_hybrid(self):
        for convention in ("padis_real", "paper_complex"):
            old, image = make_system(legacy, convention)
            new, _ = make_system(convention=convention)
            old_pair, new_pair = old.initialization_rhs(1.), new.initialization_rhs(1.)
            for a, b in zip(old_pair, new_pair): self.close(a, b, atol=0, rtol=0)
            self.close(new.normal(image, .7), old.adjoint(old.forward(image)) + image / old.system_sigma(.7)**2, 0, 0)
            score = (.1-image) / new.system_sigma(.7)**2
            old_pair, new_pair = old.update_rhs(image, score, .7, .5), new.update_rhs(image, score, .7, .5)
            for a, b in zip(old_pair, new_pair): self.close(a, b, 0, 0)
            self.close(old.solve(*old_pair, .7)[0], new.solve(*new_pair, .7)[0], 0, 0)
            self.assertTrue(torch.equal(old.generator.get_state(), new.generator.get_state()))
            for partition_mode in ("fixed", "cycle_k", "random_step"):
                shifts = ([(0, 0)] * 4 if partition_mode == "fixed" else
                          [(0, 0), (0, 1), (1, 0), (1, 1)])
                levels = ([random_shifts(4, 2, f"{partition_mode}:{i}") for i in range(3)]
                          if partition_mode == "random_step" else [shifts] * 3)
                args = dict(dps_operator=fixture()[0], dps_measurement=fixture()[1], dps_sigmas=(.08,.03),
                            dps_shifts=random_shifts(4,2,19), dps_inner_loops=2, report=lambda _: None,
                            pula_shifts_by_level=levels)
                old, _ = make_system(legacy, convention)
                new, _ = make_system(convention=convention)
                old_result = legacy.reconstruct(old, Oracle(), (1., .7,.4,.2), shifts, .5, **args)
                new_result = reconstruct(new, Oracle(), (1., .7,.4,.2), shifts, .5, **args)
                self.close(new_result[0], old_result[0], 0, 0)
                self.assertEqual(new_result[2]["initialization_cg"], old_result[2]["initialization_cg"])
                self.assertEqual(new_result[2]["update_cg"], old_result[2]["update_cg"])
                self.assertTrue(torch.equal(old.generator.get_state(), new.generator.get_state()))

    def test_02_hermitian_positive_and_explicit_matrix_formula(self):
        for convention in ("padis_real", "paper_complex"):
            system, image = make_system(convention=convention)
            variance = torch.linspace(0, 8, image.numel(), dtype=torch.float64).reshape_as(image)
            for variant in ("balanced_k0", "balanced_full"):
                configure(system, variant, variance=variance)
                u = image.flip(-1) + .3j
                jv, ju = system.normal(image,.7), system.normal(u,.7)
                self.close(torch.vdot(u.flatten(), jv.flatten()), torch.vdot(ju.flatten(), image.flatten()))
                self.assertGreater(torch.vdot(image.flatten(), jv.flatten()).real.item(), 0)
                n = image.numel()
                columns = []
                for j in range(n):
                    basis = torch.eye(n,dtype=image.dtype)[j].reshape_as(image)
                    columns.append(system.adjoint(system.forward(basis)).flatten())
                b = torch.stack(columns,1)
                kappa = 1 if convention == "padis_real" else 2
                diagonal = 1/(kappa*.7**2) + .5/(2*kappa*(.7**2+.5**2))
                vdiag = variance.flatten() * .5/(2*kappa) if variant == "balanced_full" else torch.zeros(n,dtype=torch.float64)
                expected = 1.25*b + torch.diag(diagonal + vdiag)
                self.close(jv.flatten(), expected@image.flatten())
                before = (system.forward_calls, system.adjoint_calls)
                system.normal(image, .7)
                self.assertEqual((system.forward_calls-before[0],system.adjoint_calls-before[1]),(1,1))

    def test_03_zero_variance_p3_equals_p2_and_warm_start(self):
        a, image = make_system(); b, _ = make_system()
        configure(a,"balanced_k0"); configure(b,"balanced_full",variance=torch.zeros_like(image.real))
        score = (.2-image)/.7**2
        ar, aw = a.update_rhs(image,score,.7,.5)
        br, bw = b.update_rhs(image,score,.7,.5)
        self.close(ar,br,0,0); self.close(aw,bw,0,0)
        self.close(a.solve(ar,aw,.7)[0],b.solve(br,bw,.7)[0],0,0)
        self.close(a.normal_diagonal.expand_as(b.normal_diagonal),b.normal_diagonal,0,0)

    def test_04_real_complex_p0_p2_p3_physical_equivalence(self):
        for dtype,atol,rtol in ((torch.complex128,1e-10,1e-9),(torch.complex64,2e-5,2e-5)):
            for variant in ("baseline","balanced_k0","balanced_full"):
                results=[]
                for convention in ("padis_real","paper_complex"):
                    system,_=make_system(convention=convention,dtype=dtype)
                    oracle=Oracle()
                    sequence=[random_shifts(4,2,f"paired:{i}") for i in range(3)]
                    out=reconstruct(system,oracle,(1.,.7,.4,.2),sequence[0],.5,
                                    pula_shifts_by_level=sequence,pula_variant=variant,pa_block_size=2,report=lambda _:None)
                    results.append((out,system.generator.get_state()))
                self.close(results[0][0][0],results[1][0][0],atol,rtol)
                self.assertTrue(torch.equal(results[0][1],results[1][1]))
                for left,right in zip(results[0][0][2]["pa_balance"],results[1][0][2]["pa_balance"]):
                    self.assertAlmostEqual(left["per_trajectory"][0]["v_mean"],right["per_trajectory"][0]["v_mean"],places=5 if dtype==torch.complex64 else 9)

    def test_05_statistics_freeze_weighted_ema_transport_and_partial_regions(self):
        x=torch.zeros((1,1,9,10),dtype=torch.complex128)
        stats=PABalanceStats(.5,2,2,8)
        frozen=stats.begin_level(1.,x)
        self.assertEqual(frozen.shape,x.shape); self.assertEqual(frozen.max().item(),0)
        cs,co,ci=edm_coefficients(1.,.5)
        stats.observe(x,cs*x+co*torch.ones_like(x)*2,(0,0))
        stats.observe(x,cs*x+co*torch.ones_like(x)*4,(1,1))
        self.close(stats.m,torch.ones_like(stats.m)*.5)
        self.close(stats.weight,torch.ones_like(stats.weight)*.5)
        self.assertEqual(frozen.max().item(),0)  # no look-ahead or in-layer mutation
        row=stats.end_level(.5); self.assertEqual(row["innovation_pairs"],1)
        old_m,old_w=stats.m.clone(),stats.weight.clone()
        second=stats.begin_level(.5,x)
        self.close(stats.m,old_m*.5); self.close(stats.weight,old_w*.5)
        expected=(edm_coefficients(.5,.5)[1]/.5**2)**2*(old_m*.5)/(old_w*.5+stats.epsilon)
        self.close(second,region_expand(expected,x.shape,8))
        self.assertIsNone(stats.previous_f); self.assertIsNone(stats.previous_z)
        self.assertTrue((second>=0).all()); self.assertTrue(torch.isfinite(second).all())
        stats.observe(x,torch.zeros_like(x),(1,0))
        self.assertEqual(stats.pairs,0)  # no cross-sigma difference
        value=torch.arange(90,dtype=torch.float64).reshape(1,1,9,10)
        pooled=region_mean(value,8)
        self.close(pooled[0,0,1,1],value[0,0,8:,8:].mean())
        self.close(interior_window(x,(0,0),2,2),interior_window(x,(2,2),2,2))

    def test_06_slope_fit_is_previous_layer_only(self):
        x=torch.ones((1,1,4,4),dtype=torch.complex128)
        stats=PABalanceStats(.5,2,2,2)
        stats.begin_level(.7,x); cs,co,ci=edm_coefficients(.7,.5)
        for scale in (1.,2.,3.):
            state=scale*x
            stats.observe(state,cs*state+co*(2*ci*state),(0,1))
            self.assertEqual(stats.slope,0)
        row=stats.end_level(.5)
        self.assertAlmostEqual(row["next_state_slope"],1.,places=10)
        stats.begin_level(.4,x)
        self.assertAlmostEqual(stats.slope,1.,places=10)

    def test_07_noise08_only_scales_active_noise_and_matches_warm(self):
        baseline,image=make_system(); noise08,_=make_system()
        for a,b in zip(baseline.initialization_rhs(1.),noise08.initialization_rhs(1.)): self.close(a,b,0,0)
        noise08.configure_update("noise08",.7,.5)
        score=(.3-image)/.7**2
        rng=baseline.generator.get_state().clone()
        likelihood=baseline.adjoint(baseline.y-baseline.forward(image))
        noise,nx=baseline.noise_rhs(image,.7)
        baseline.generator.set_state(rng)
        rhs,warm=noise08.update_rhs(image,score,.7,.5)
        self.close(rhs,.5*(likelihood+score)+.8*noise)
        self.close(warm,.5*.7**2*score+.7*(.8*nx))
        baseline.update_rhs(image,score,.7,.5)
        self.assertTrue(torch.equal(baseline.generator.get_state(),noise08.generator.get_state()))
        self.close(baseline.normal(image,.7),noise08.normal(image,.7),0,0)

    def test_08_all_variants_same_raw_rng_partitions_and_332_budget(self):
        sequence=[random_shifts(4,2,f"pula:{level}") for level in range(23)]
        dps_shifts=random_shifts(240,2,123)
        records=[]
        for variant in VARIANTS:
            system,_=make_system(height=2)
            system.cg_iters,system.cg_rtol=10,1e-6
            draws=[]
            original=system.draw_noise
            def capture(template,generator):
                draw=original(template,generator); draws.append(draw.clone()); return draw
            system.draw_noise=capture
            oracle=Oracle(2)
            raw_draws=[]
            native=recon_pula.native_complex_noise
            def capture_native(template,generator):
                value=native(template,generator); raw_draws.append(value.clone()); return value
            with patch.object(recon_pula,"native_complex_noise",side_effect=capture_native):
                result=reconstruct(system,oracle,[1.]+[.7*(.3/.7)**(i/22) for i in range(23)],sequence[0],.5,
                               pula_shifts_by_level=sequence,pula_variant=variant,pa_block_size=2,
                               dps_operator=fixture(height=2,width=2)[0],dps_measurement=fixture(height=2,width=2)[1],
                               dps_sigmas=[.1*(.03/.1)**(i/23) for i in range(24)],dps_shifts=dps_shifts,
                               dps_inner_loops=10,report=lambda _:None)
            timing=result[2]
            self.assertEqual(timing["logical_nfe"],332); self.assertEqual(timing["cg_solves"],93)
            self.assertEqual(timing["network_backprops"],240); self.assertEqual(timing["network_forwards"],332)
            self.assertEqual(len(draws),186)
            if variant=="balanced_full": self.assertGreater(timing["pa_balance"][1]["per_trajectory"][0]["v_mean"],0)
            self.assertEqual(len(raw_draws),656)
            records.append((draws,oracle.seen_shifts,system.generator.get_state(),timing["initialization_cg"],raw_draws))
        for record in records[1:]:
            for left,right in zip(records[0][0],record[0]): self.close(left,right,0,0)
            for left,right in zip(records[0][4],record[4]): self.close(left,right,0,0)
            self.assertEqual(records[0][1],record[1]); self.assertEqual(records[0][3],record[3])
            self.assertTrue(torch.equal(records[0][2],record[2]))

    def test_09_batched_trajectories_independent_against_serial(self):
        for variant in VARIANTS:
            sequences=[[random_shifts(4,2,f"{seed}:{i}") for i in range(3)] for seed in (123,124)]
            systems=[make_system(seed=seed)[0] for seed in (123,124)]
            shifts=[[sequences[j][i][k] for j in range(2)] for i in range(3) for k in range(4)]
            levels=[shifts[i:i+4] for i in range(0,12,4)]
            batch=reconstruct(TrajectorySystems(systems),Oracle(),(1.,.7,.4,.2),levels[0],.5,
                              pula_shifts_by_level=levels,pula_variant=variant,pa_block_size=2,report=lambda _:None)
            for j,seed in enumerate((123,124)):
                serial_system,_=make_system(seed=seed)
                serial=reconstruct(serial_system,Oracle(),(1.,.7,.4,.2),sequences[j][0],.5,
                                   pula_shifts_by_level=sequences[j],pula_variant=variant,pa_block_size=2,report=lambda _:None)
                self.close(batch[0][j:j+1],serial[0],0,0)
                self.assertTrue(torch.equal(systems[j].generator.get_state(),serial_system.generator.get_state()))
                for l in range(3):
                    self.assertEqual(batch[2]["pa_balance"][l]["per_trajectory"][j],serial[2]["pa_balance"][l]["per_trajectory"][0])

    def test_10_invalid_inputs_tiny_sigma_and_zero_denominators(self):
        x=torch.zeros((1,1,1,3),dtype=torch.complex64)
        stats=PABalanceStats(.5,2,2)
        stats.begin_level(1e-10,x)
        stats.observe(x,x,(0,0)); stats.observe(x,x,(1,1)); row=stats.end_level(.5)
        self.assertEqual(row["innovation_mean"],0); self.assertEqual(row["state_slope"],0)
        self.assertTrue(torch.isfinite(stats.begin_level(1e-11,x)).all())
        for invalid in (0.,float("nan"),float("inf")):
            with self.assertRaises(ValueError): edm_coefficients(invalid,.5)
        for variance in (-torch.ones_like(x.real),torch.full_like(x.real,float("nan")),torch.full_like(x.real,float("inf"))):
            with self.assertRaises(ValueError): balanced_coefficients(x,.7,.5,.5,"padis_real",variance)
        with self.assertRaises(ValueError): balanced_coefficients(x,1e-30,.5,.5,"padis_real")
        with self.assertRaises(FloatingPointError): balanced_coefficients(x,1e30,.5,.5,"padis_real")
        with self.assertRaises(FloatingPointError): stats.observe(x,torch.full_like(x,complex(float("nan"),0)),(0,0))
        self.assertEqual(edm_sigma_data(Net()),.5)
        self.assertEqual(edm_sigma_data(torch.nn.DataParallel(Net())),.5)

    def test_11_discrete_gaussian_balance_noncommuting_matrices(self):
        rng=torch.Generator().manual_seed(121)
        def spd():
            a=torch.randn((5,5),dtype=torch.float64,generator=rng); return a.T@a+torch.eye(5)
        h,k,v=spd(),spd(),spd(); gamma=.5
        j=h+gamma/2*(k+v); m=torch.linalg.inv(j); c=torch.linalg.inv(k)
        t=torch.eye(5)-gamma*m@k
        q=m@(2*gamma*h+gamma**2*v)@m
        self.close(t@c@t.T+q,c,1e-14,1e-13)

    def test_12_cli_bash_dry_run_and_invalid_variant(self):
        for variant in VARIANTS:
            command=["bash","bash/eval_pa_balanced_pula_R7_pilot.sh","--dry-run"]
            env={**os.environ,"VARIANT":variant,"GPU":"3","RUN_ID":"cpu_dry_run"}
            result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            # Bash emits one command line followed by one JSON plan per contrast.
            plans=[]
            text=result.stdout
            while "\n{" in text:
                text=text[text.index("\n{")+1:]
                plan,end=json.JSONDecoder().raw_decode(text); plans.append(plan); text=text[end:]
            self.assertEqual(len(plans),2)
            for plan in plans:
                self.assertEqual(plan["logical_updates"],332); self.assertEqual(plan["cg_solves_per_trajectory"],93)
                self.assertEqual(plan["sample_indices"],[0,26]); self.assertEqual(plan["partition_mode"],"random_step")
                self.assertEqual(plan["pula_variant"],variant); self.assertFalse(plan["gpu_executed"])
        bad=subprocess.run([sys.executable,"eval_pula/run.py","--model_path","x","--val_dir","x",
                            "--out_root","x","--pula_variant","balanced_full","--dry_run"],
                           cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0); self.assertIn("random_step",bad.stderr)

    def test_13_balanced_rhs_is_original_noise_and_diagonal_warm(self):
        image=fixture()[2]
        variance=torch.linspace(0,3,image.numel(),dtype=image.real.dtype).reshape_as(image)
        score=(.3-image)/.7**2
        for convention in ("padis_real","paper_complex"):
            systems=[]
            for variant in ("baseline","balanced_k0","balanced_full"):
                system,_=make_system(convention=convention)
                configure(system,variant,variance=variance)
                clone=torch.Generator().set_state(system.generator.get_state())
                system.draw_noise(system.y,clone)
                nx=system.draw_noise(image,clone)
                rhs,warm=system.update_rhs(image,score,.7,.5)
                if variant!="baseline":
                    self.close(warm,(.5*score+nx/system.system_sigma(.7))/system.normal_diagonal,0,0)
                systems.append((rhs,system.generator.get_state()))
            for rhs,rng in systems[1:]:
                self.close(rhs,systems[0][0],0,0)
                self.assertTrue(torch.equal(rng,systems[0][1]))

    def test_14_window_offset_small_fov_and_spatial_map_shape(self):
        x=torch.zeros((1,1,9,10),dtype=torch.complex64)
        window=interior_window(x,(3,5),64,64)
        expected=math.sin(math.pi*((64-3+.5)%64)/64)**2*math.sin(math.pi*((64-5+.5)%64)/64)**2
        self.assertAlmostEqual(window[0,0].item(),expected,places=14)
        self.assertFalse(torch.equal(window,interior_window(x,(5,3),64,64)))
        scalar,full=balanced_coefficients(x,.7,.5,.5,"padis_real",torch.zeros((9,10)))
        self.assertEqual(scalar,1.25); self.assertEqual(full.shape,x.shape)
        for height,width in ((1,1),(3,7),(8,8),(9,10),(17,19)):
            state=torch.zeros((1,1,height,width),dtype=torch.complex64)
            tracker=PABalanceStats(.5,2,2,8)
            self.assertEqual(tracker.begin_level(.7,state).shape,state.shape)
        with self.assertRaises(ValueError):
            balanced_coefficients(x,.7,.5,.5,"padis_real",torch.zeros((8,10)))
        with self.assertRaises(ValueError):
            PABalanceStats(.5,2,2).begin_level(1e-200,x)

    def test_15_original_noise_and_dps_loop_source_contract(self):
        import ast
        old=ast.parse((ROOT/"tests/fixtures/legacy_recon_pula.py").read_text())
        new=ast.parse((ROOT/"eval_pula/recon_pula.py").read_text())
        for name in ("real_unit_noise","native_complex_noise","whiten_measurement"):
            left=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            right=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(left),ast.dump(right))
        def dps_loop(tree):
            return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
                        and ast.unparse(n.target)=="(outer, sigma)")
        self.assertEqual(ast.dump(dps_loop(old)),ast.dump(dps_loop(new)))


if __name__=="__main__":
    unittest.main(verbosity=2)

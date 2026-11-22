"""Evaluator adapter that replaces only the PaDIS reconstruction entrypoint."""

from evaluator import DPSHyperEvaluator

from .recon_lgfc import dps2_lgfc


class LgfcDPSHyperEvaluator(DPSHyperEvaluator):
    def __init__(
        self,
        *args,
        lgfc_config,
        sigma_min=0.003,
        sigma_max=10.0,
        rho=7.0,
        save_diagnostics=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lgfc_config = lgfc_config
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.save_diagnostics = save_diagnostics

    def dps2_wrapper(
        self,
        inverse_op,
        measurement,
        clean,
        zeta,
        pad,
        psize,
        num_steps,
        inner_loops,
        save_dir=None,
        tag=None,
        **_unused_evaluator_options,
    ):
        return dps2_lgfc(
            net=self.model,
            latents=self.latents,
            latents_pos=self.latents_pos,
            inverseop=inverse_op,
            measurement=measurement,
            lgfc_config=self.lgfc_config,
            num_steps=num_steps,
            inner_loops=inner_loops,
            zeta=zeta,
            pad=pad,
            patch_size=psize,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            save_dir=save_dir,
            tag=tag,
            save_diagnostics=self.save_diagnostics,
        )

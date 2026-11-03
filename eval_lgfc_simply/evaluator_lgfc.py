"""Evaluator adapter that replaces only the PaDIS reconstruction entrypoint."""

from evaluator import DPSHyperEvaluator

from .recon_lgfc import dps2_lgfc


class LgfcDPSHyperEvaluator(DPSHyperEvaluator):
    def __init__(self, *args, lgfc_config, save_diagnostics=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lgfc_config = lgfc_config
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
            save_dir=save_dir,
            tag=tag,
            save_diagnostics=self.save_diagnostics,
        )

"""Opt-in scrambled Sobol sequence over actual operator calls."""
from eval_pula.flex_grids import build_grid_plan


def quality_grid_plan(rows, pad, seed, legacy_mode, pula_mode, dps_mode):
    sobol_p = pula_mode == 'sobol'
    sobol_d = dps_mode == 'sobol'
    shifts = build_grid_plan(rows, pad, seed, legacy_mode,
                             'cycle_k' if sobol_p else pula_mode,
                             'random_step' if sobol_d else dps_mode)
    if not (sobol_p or sobol_d):
        return shifts
    from torch.quasirandom import SobolEngine
    for method, enabled, offset in [('P', sobol_p, 0), ('D', sobol_d, 1)]:
        if not enabled:
            continue
        slots = [(i,j) for i,row in enumerate(rows) for j,m in enumerate(row) if m==method]
        if not slots:
            continue
        # Independent streams; no extra denoising, no change to state RNG.
        points = SobolEngine(2, scramble=True, seed=seed+offset).draw(len(slots))
        points = (points*pad).long().clamp(0,pad-1).tolist()
        for (i,j), point in zip(slots,points):
            shifts[i][j] = tuple(point)
    return shifts

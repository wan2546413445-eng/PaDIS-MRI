#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
export MPLBACKEND=Agg
export PYTHONPATH="$PWD:$PWD/train/padis-mri:$PWD/eval${PYTHONPATH:+:$PYTHONPATH}"

RUNS=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB
RUN=overnight_overlap_independent_center_symmetric_lam2p0_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform
MODEL="${MODEL:-$RUNS/$RUN/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam2p0-p64-align-symmetric-geom-identical-spatial-uniform/network-snapshot-005000.pkl}"
VAL="${VAL:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}"
OUT_ROOT="${OUT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/eval_fast_OPA_lam2p0_T1FLAIR_R7_s78x10_cg39_dps39_fh_v3}"

case "${GROUP:-both}" in
    both) groups=(isotropic fh) ;;
    isotropic|fh) groups=("$GROUP") ;;
    *) echo "GROUP must be both, isotropic, or fh." >&2; exit 2 ;;
esac
samples=({0..31})
if [[ -n "${SAMPLE:-}" ]]; then
    [[ "$SAMPLE" =~ ^([0-9]|[12][0-9]|3[01])$ ]] || {
        echo "SAMPLE must be between 0 and 31." >&2; exit 2;
    }
    samples=("$SAMPLE")
fi

[[ -f "$MODEL" && -d "$VAL" ]] || {
    echo "Check MODEL and VAL: $MODEL | $VAL" >&2; exit 1;
}
if [[ "${GROUP:-both}" != isotropic ]]; then
    : "${FH_COV_PATH:?Set FH_COV_PATH to the v3 training DCT and noise cache}"
    [[ -s "$FH_COV_PATH" ]] || { echo "Missing cache: $FH_COV_PATH" >&2; exit 1; }
    fh_cache_id="train-cache-v3:$(sha256sum < "$FH_COV_PATH")"
fi

profile=()
if [[ "${PROFILE_FH:-0}" == 1 ]]; then
    profile=(--profile_fh)
fi
diagnostics=()
if [[ "${FH_DIAGNOSTICS:-0}" == 1 ]]; then
    : "${SAMPLE:?Set SAMPLE to a single index for FH diagnostics}"
    diagnostics=(--diagnose_fh)
    OUT_ROOT="${OUT_ROOT}_diag"
fi
mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/.run.lock"
flock -n 9 || { echo "Already running: $OUT_ROOT" >&2; exit 1; }

has_outputs() {
    local eval_dir="$1/evaluate"
    [[ -s "$eval_dir/recons/recon_patch_$2.npy" &&
       -s "$eval_dir/config.json" &&
       -s "$eval_dir/timing.json" &&
       -s "$eval_dir/comp_plots/results.json" &&
       -s "$eval_dir/comp_plots/comparison_sample_$2.png" &&
       -s "$1/run.log" ]] &&
    { [[ "${#profile[@]}" == 0 ]] || [[ -s "$eval_dir/fh_profile.json" ]]; } &&
    { [[ "${#diagnostics[@]}" == 0 ]] ||
      [[ -s "$eval_dir/fh_diagnostics_$2.json" && -s "$eval_dir/fh_transition_$2.npz" ]]; }
}

run_sample() {
    local group="$1" idx="$2"
    local out="$OUT_ROOT/$group/sample_$idx" archived
    local -a fh=()
    if [[ "$group" == fh ]]; then
        fh=(--fh_cov_path "$FH_COV_PATH" --fh_max_rank "${FH_MAX_RANK:-16}")
    fi
    if [[ -f "$out/.done" ]] && has_outputs "$out" "$idx" &&
       { [[ "$group" == isotropic ]] || grep -Fxq -- "$fh_cache_id" "$out/.done"; }; then
        echo "Skip $group/sample_$idx (complete)"
        return
    fi
    if [[ -d "$out" ]]; then
        mkdir -p "$OUT_ROOT/.incomplete"
        archived=$(mktemp -d "$OUT_ROOT/.incomplete/${group}_sample_${idx}.XXXXXX")
        mv -- "$out" "$archived/"
    fi
    mkdir -p "$out"
    echo "Run $group/sample_$idx"
    python -u eval_fast/run.py \
        --mode hybrid --cg_covariance "$group" "${fh[@]}" "${profile[@]}" "${diagnostics[@]}" \
        --model_path "$MODEL" --val_dir "$VAL" --sample_indices "$idx" --mask_select 7 \
        --image_size 384 --pad 64 --psize 64 --seed 123 \
        --steps 78 --inner_loops 10 --zeta 3.0 \
        --cg_lambda 0.1 --cg_iters 5 --cg_gain 0.1 --dps_tail_steps 39 \
        --save_dir "$out" 2>&1 | tee "$out/run.log"
    if ! has_outputs "$out" "$idx"; then
        echo "Missing evaluation output: $out" >&2
        exit 1
    fi
    if [[ "$group" == fh ]]; then
        printf '%s\n' "$fh_cache_id" > "$out/.done"
    else
        touch "$out/.done"
    fi
}

for idx in "${samples[@]}"; do
    for group in "${groups[@]}"; do
        run_sample "$group" "$idx"
    done
done

if [[ "${#diagnostics[@]}" != 0 && "${GROUP:-both}" == both ]]; then
    python -m eval_fast.fh_diagnostics "$OUT_ROOT" "$SAMPLE"
fi

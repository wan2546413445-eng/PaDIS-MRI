#!/usr/bin/env python3
"""
Generate non-overwriting context-center training files.

Default source:
  train/padis-mri/train.py

Generated:
  train/padis-mri/training/training_loop_context_center.py
  train/padis-mri/train_context_center.py

Required:
  train/padis-mri/training/patch_loss_context_center.py

This version preserves original EDM defaults. It does not add or pass:
  --sigma-data
  --p-mean
  --p-std
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_training_loop(text: str) -> str:
    text = text.replace(
        '"""Main training loop."""',
        '"""Main training loop. Modified as context-center training loop."""',
        1,
    )

    if "context_margin" not in text[text.find("def training_loop("): text.find("def training_loop(") + 1600]:
        text = text.replace(
            "pad_width           = 0,       # Width of zero padding on each side\n"
            "    device              = torch.device('cuda'),",
            "pad_width           = 0,       # Width of zero padding on each side\n"
            "    context_margin      = 0,       # Extra context pixels around the supervised patch\n"
            "    device              = torch.device('cuda'),",
            1,
        )

    helper = """
    def get_batch_mul_for_effective_patch(effective_patch_size):
        \"""
        Estimate memory using actual network input size:
            effective_patch_size = target_patch_size + 2 * context_margin
        \"""
        effective_patch_size = int(effective_patch_size)
        if effective_patch_size in batch_mul_dict:
            return batch_mul_dict[effective_patch_size]

        valid_sizes = sorted(batch_mul_dict.keys())
        larger_or_equal = [s for s in valid_sizes if s >= effective_patch_size]
        if len(larger_or_equal) > 0:
            return batch_mul_dict[larger_or_equal[0]]

        return 1
"""

    if "get_batch_mul_for_effective_patch" not in text:
        text = text.replace(
            "batch_mul_dict = {512: 1, 256: 1, 128: 4, 96: 8, 64: 16, 56: 16, 48: 16, 32: 32, 24: 32, 16: 64}",
            "batch_mul_dict = {512: 1, 256: 1, 128: 4, 96: 8, 64: 16, 56: 16, 48: 16, 32: 32, 24: 32, 16: 64}" + helper,
            1,
        )

    text = text.replace(
        "batch_mul_avg = batch_mul_dict[patch_size] // batch_mul_dict[img_resolution]",
        "batch_mul_avg = get_batch_mul_for_effective_patch(int(patch_size) + 2 * int(context_margin))",
    )

    text = text.replace(
        "batch_mul_avg = float(np.sum([p * batch_mul_dict.get(int(ps), 1) for ps, p in zip(patch_list, p_list)]))",
        "batch_mul_avg = float(np.sum([p * get_batch_mul_for_effective_patch(int(ps) + 2 * int(context_margin)) for ps, p in zip(patch_list, p_list)]))",
    )

    text = text.replace(
        "batch_mul_avg = np.sum(np.array(p_list) * np.array([4, 2, 1]))  # 2",
        "batch_mul_avg = float(np.sum([p * get_batch_mul_for_effective_patch(int(ps) + 2 * int(context_margin)) for ps, p in zip(patch_list, p_list)]))",
    )

    text = text.replace(
        "batch_mul = batch_mul_dict[patch_size] #// batch_mul_dict[img_resolution]",
        "effective_patch_size = int(patch_size) + 2 * int(context_margin)\n"
        "                batch_mul = get_batch_mul_for_effective_patch(effective_patch_size)",
    )

    old = (
        "loss = loss_fn(net=ddp, images=images, patch_size=patch_size, resolution=img_resolution,\n"
        "                               labels=labels, augment_pipe=augment_pipe)"
    )
    new = (
        "loss = loss_fn(net=ddp, images=images, patch_size=patch_size, resolution=img_resolution,\n"
        "                               labels=labels, augment_pipe=augment_pipe, context_margin=context_margin)"
    )
    text = text.replace(old, new, 1)

    return text


def patch_train_entry(text: str) -> str:
    text = text.replace(
        "from training import training_loop",
        "from training import training_loop_context_center as training_loop",
        1,
    )

    if "--context-margin" not in text:
        marker = "@click.option('--pad_width'"
        idx = text.find(marker)
        if idx < 0:
            raise RuntimeError("Could not find @click.option('--pad_width'...) to insert --context-margin.")
        line_end = text.find("\n", idx)
        option = (
            "@click.option('--context-margin', help='Extra context pixels around each supervised patch', "
            "metavar='INT', type=int, default=0, show_default=True)\n"
        )
        text = text[:line_end + 1] + option + text[line_end + 1:]

    if "c.context_margin = opts.context_margin" not in text:
        text = text.replace(
            "c.pad_width = opts.pad_width",
            "c.pad_width = opts.pad_width\n    c.context_margin = opts.context_margin",
            1,
        )

    text = text.replace(
        "c.loss_kwargs.class_name = 'training.patch_loss.Patch_EDMLoss'",
        "c.loss_kwargs.class_name = 'training.patch_loss_context_center.Patch_ContextCenter_EDMLoss'",
    )
    text = text.replace(
        'c.loss_kwargs.class_name = "training.patch_loss.Patch_EDMLoss"',
        'c.loss_kwargs.class_name = "training.patch_loss_context_center.Patch_ContextCenter_EDMLoss"',
    )

    if "c.loss_kwargs.context_margin = opts.context_margin" not in text:
        marker = "# Network options."
        if marker not in text:
            marker = "    if opts.augment:"
        block = (
            "    if opts.precond == 'pedm':\n"
            "        c.loss_kwargs.context_margin = opts.context_margin\n\n"
        )
        text = text.replace(marker, block + marker, 1)

    banner = (
        "# ---------------------------------------------------------------\n"
        "# Auto-generated context-center training entrypoint.\n"
        "# Source file is preserved. Generated by tools/make_context_center_train_default.py.\n"
        "# Original EDM sigma defaults are preserved.\n"
        "# ---------------------------------------------------------------\n\n"
    )
    if "Auto-generated context-center training entrypoint" not in text:
        text = banner + text

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-src", default="train/padis-mri/train.py")
    parser.add_argument("--train-dst", default="train/padis-mri/train_context_center.py")
    parser.add_argument("--loop-src", default="train/padis-mri/training/training_loop.py")
    parser.add_argument("--loop-dst", default="train/padis-mri/training/training_loop_context_center.py")
    args = parser.parse_args()

    train_src = Path(args.train_src)
    loop_src = Path(args.loop_src)
    train_dst = Path(args.train_dst)
    loop_dst = Path(args.loop_dst)

    if not train_src.exists():
        raise FileNotFoundError(f"Source training file not found: {train_src}")
    if not loop_src.exists():
        raise FileNotFoundError(f"Source training loop not found: {loop_src}")

    loop_text = patch_training_loop(loop_src.read_text(encoding="utf-8"))
    train_text = patch_train_entry(train_src.read_text(encoding="utf-8"))

    loop_dst.parent.mkdir(parents=True, exist_ok=True)
    train_dst.parent.mkdir(parents=True, exist_ok=True)

    loop_dst.write_text(loop_text, encoding="utf-8")
    train_dst.write_text(train_text, encoding="utf-8")

    print(f"[OK] Wrote {loop_dst}")
    print(f"[OK] Wrote {train_dst}")
    print("[Check] grep -n \"context_margin\\|get_batch_mul_for_effective_patch\" train/padis-mri/training/training_loop_context_center.py")
    print("[Check] grep -n \"context-margin\\|patch_loss_context_center\\|training_loop_context_center\" train/padis-mri/train_context_center.py")


if __name__ == "__main__":
    main()

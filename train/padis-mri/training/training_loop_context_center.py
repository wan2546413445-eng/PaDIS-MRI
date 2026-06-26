# ---------------------------------------------------------------
# Context-center training loop entrypoint for PaDIS-MRI.
#
# File path:
#   train/padis-mri/training/training_loop_context_center.py
#
# This file keeps the original training_loop.py untouched. It reads the current
# training_loop.py source, applies the context-center modifications, and exposes
# a patched training_loop() function.
#
# Main changes:
#   1. Add context_margin to training_loop().
#   2. Estimate batch_mul using:
#        effective_patch_size = patch_size + 2 * context_margin
#   3. Pass context_margin into Patch_ContextCenter_EDMLoss.
# ---------------------------------------------------------------

from __future__ import annotations

from pathlib import Path


def _patch_training_loop_source(text: str) -> str:
    text = text.replace(
        '"""Main training loop."""',
        '"""Main training loop. Modified as context-center training loop."""',
        1,
    )

    # Add context_margin to training_loop() arguments.
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

    # Add batch multiplier helper after batch_mul_dict.
    if "get_batch_mul_for_effective_patch" not in text:
        old = "batch_mul_dict = {512: 1, 256: 1, 128: 4, 96: 8, 64: 16, 56: 16, 48: 16, 32: 32, 24: 32, 16: 64}"
        if old not in text:
            raise RuntimeError("Could not find batch_mul_dict line in training_loop.py.")
        text = text.replace(old, old + helper, 1)

    # Progressive path.
    text = text.replace(
        "batch_mul_avg = batch_mul_dict[patch_size] // batch_mul_dict[img_resolution]",
        "batch_mul_avg = get_batch_mul_for_effective_patch(int(patch_size) + 2 * int(context_margin))",
    )

    # User-supplied patch list path.
    text = text.replace(
        "batch_mul_avg = float(np.sum([p * batch_mul_dict.get(int(ps), 1) for ps, p in zip(patch_list, p_list)]))",
        "batch_mul_avg = float(np.sum([p * get_batch_mul_for_effective_patch(int(ps) + 2 * int(context_margin)) for ps, p in zip(patch_list, p_list)]))",
    )

    # Default patch list path.
    text = text.replace(
        "batch_mul_avg = np.sum(np.array(p_list) * np.array([4, 2, 1]))  # 2",
        "batch_mul_avg = float(np.sum([p * get_batch_mul_for_effective_patch(int(ps) + 2 * int(context_margin)) for ps, p in zip(patch_list, p_list)]))",
    )

    # Per-step batch multiplier.
    text = text.replace(
        "batch_mul = batch_mul_dict[patch_size] #// batch_mul_dict[img_resolution]",
        "effective_patch_size = int(patch_size) + 2 * int(context_margin)\n"
        "                batch_mul = get_batch_mul_for_effective_patch(effective_patch_size)",
    )

    # Pass context_margin to Patch_ContextCenter_EDMLoss.
    old_call = (
        "loss = loss_fn(net=ddp, images=images, patch_size=patch_size, resolution=img_resolution,\n"
        "                               labels=labels, augment_pipe=augment_pipe)"
    )
    new_call = (
        "loss = loss_fn(net=ddp, images=images, patch_size=patch_size, resolution=img_resolution,\n"
        "                               labels=labels, augment_pipe=augment_pipe, context_margin=context_margin)"
    )
    if old_call in text:
        text = text.replace(old_call, new_call, 1)

    # Basic safety checks on the patched source.
    required = [
        "context_margin      = 0",
        "get_batch_mul_for_effective_patch",
        "effective_patch_size = int(patch_size) + 2 * int(context_margin)",
        "context_margin=context_margin",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError("Context-center training_loop patch incomplete. Missing: " + ", ".join(missing))

    return text


_source_path = Path(__file__).with_name("training_loop.py")
_source = _source_path.read_text(encoding="utf-8")
_patched = _patch_training_loop_source(_source)

# Execute the patched original source in this module namespace.
# This exposes training_loop() with the context_margin argument.
exec(compile(_patched, str(_source_path), "exec"), globals(), globals())

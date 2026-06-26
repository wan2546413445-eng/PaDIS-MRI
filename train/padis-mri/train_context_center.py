# ---------------------------------------------------------------
# Context-center training entrypoint for PaDIS-MRI.
#
# File path:
#   train/padis-mri/train_context_center.py
#
# This file keeps the original train.py untouched. It reads train.py, applies
# the context-center modifications, and executes the patched entrypoint.
#
# Main changes:
#   1. Import training_loop_context_center instead of training_loop.
#   2. Add CLI option: --context-margin.
#   3. Use training.patch_loss_context_center.Patch_ContextCenter_EDMLoss.
#   4. Pass context_margin to loss_kwargs and training_loop().
#
# No sigma route is introduced here:
#   No --sigma-data
#   No --p-mean
#   No --p-std
# ---------------------------------------------------------------

from __future__ import annotations

from pathlib import Path


def _patch_train_source(text: str) -> str:
    # Use the context-center training loop.
    if "from training import training_loop_context_center as training_loop" not in text:
        old_import = "from training import training_loop"
        if old_import not in text:
            raise RuntimeError("Could not find 'from training import training_loop' in train.py.")
        text = text.replace(
            old_import,
            "from training import training_loop_context_center as training_loop",
            1,
        )

    # Add --context-margin next to --pad_width.
    if "--context-margin" not in text:
        marker = "@click.option('--pad_width'"
        idx = text.find(marker)
        if idx < 0:
            raise RuntimeError("Could not find @click.option('--pad_width'...) in train.py.")
        line_end = text.find("\n", idx)
        option = (
            "@click.option('--context-margin', help='Extra context pixels around each supervised patch', "
            "metavar='INT', type=int, default=0, show_default=True)\n"
        )
        text = text[:line_end + 1] + option + text[line_end + 1:]

    # Store context_margin in the config so training_loop receives it via **c.
    if "c.context_margin = opts.context_margin" not in text:
        old = "c.pad_width = opts.pad_width"
        if old not in text:
            raise RuntimeError("Could not find 'c.pad_width = opts.pad_width' in train.py.")
        text = text.replace(
            old,
            "c.pad_width = opts.pad_width\n    c.context_margin = opts.context_margin",
            1,
        )

    # Use the context-center loss for pedm.
    text = text.replace(
        "c.loss_kwargs.class_name = 'training.patch_loss.Patch_EDMLoss'",
        "c.loss_kwargs.class_name = 'training.patch_loss_context_center.Patch_ContextCenter_EDMLoss'",
    )
    text = text.replace(
        'c.loss_kwargs.class_name = "training.patch_loss.Patch_EDMLoss"',
        'c.loss_kwargs.class_name = "training.patch_loss_context_center.Patch_ContextCenter_EDMLoss"',
    )

    # Pass context_margin to the loss constructor. Original EDM sigma defaults remain unchanged.
    if "c.loss_kwargs.context_margin = opts.context_margin" not in text:
        marker = "# Network options."
        if marker not in text:
            marker = "    if opts.augment:"
        if marker not in text:
            raise RuntimeError("Could not find insertion point for loss_kwargs.context_margin.")
        block = (
            "    if opts.precond == 'pedm':\n"
            "        c.loss_kwargs.context_margin = opts.context_margin\n\n"
        )
        text = text.replace(marker, block + marker, 1)

    # Safety checks.
    required = [
        "training_loop_context_center as training_loop",
        "--context-margin",
        "c.context_margin = opts.context_margin",
        "training.patch_loss_context_center.Patch_ContextCenter_EDMLoss",
        "c.loss_kwargs.context_margin = opts.context_margin",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError("Context-center train.py patch incomplete. Missing: " + ", ".join(missing))

    forbidden = ["--sigma-data", "--p-mean", "--p-std"]
    found_forbidden = [item for item in forbidden if item in text]
    if found_forbidden:
        raise RuntimeError("Unexpected sigma CLI option found in context-center entrypoint: " + ", ".join(found_forbidden))

    return text


_source_path = Path(__file__).with_name("train.py")
_source = _source_path.read_text(encoding="utf-8")
_patched = _patch_train_source(_source)

# Execute patched train.py. If this file is called as a script, the patched
# train.py sees __name__ == "__main__" and will run main().
exec(compile(_patched, str(_source_path), "exec"), globals(), globals())

#!/usr/bin/env python3
"""
Idempotently patch tokamind's warmstart config inheritance to HONOR the requested
data.split instead of force-overriding it with the source model's split.

Why: scripts_mast/mast_utils/config/inheritance.py replaces the requested
data.split with the pretrained source model's split ("random" for
tokamind-base-v2), logging only a warning. For the Move-2 experiment that is
fatal twice over:

  1. The DCT3D embeddings tuning then selects shots from the (unfiltered) RANDOM
     split CSV with shuffle=True and seed 54 — and shot 14305 is position 0 of
     that selection. This is the exact, deterministic cause of the recurring
     `FileNotFoundError: .../14305.zarr` across every pod since the start.
  2. Even without the crash, the finetune (and, by inheritance, the eval) would
     silently run on the random split, invalidating the temporal OOD design.

This is a deliberate scientific choice: we WANT a temporal-split finetune from a
random-split-pretrained base; the model adapts during finetuning and conformal
calibration happens in the temporal domain. Upstream's warning log line is kept.
"""

import os
import sys

WORK = os.environ.get("WORK", "/data_local/work")
TARGET = os.path.join(WORK, "tokamind", "scripts_mast", "mast_utils", "config", "inheritance.py")

OLD = '''        merged["data"]["split"] = source_split
        merged["model_source"]["data_split"] = source_split'''

NEW = '''        # PATCHED (fusion_ai Move-2): honor the requested split instead of forcing
        # the source model's split. See fusion_ai/DIAGNOSIS_14305.md.
        merged["data"]["split"] = requested_split
        merged["model_source"]["data_split"] = requested_split'''

src = open(TARGET).read()

if "PATCHED (fusion_ai Move-2)" in src:
    print(f"already patched: {TARGET}")
    sys.exit(0)

if OLD not in src:
    sys.exit(
        f"ERROR: expected code block not found in {TARGET} — upstream changed. "
        f"Inspect the finetune branch of inherit_from_source_model() manually."
    )

open(TARGET, "w").write(src.replace(OLD, NEW, 1))

# Hard verification
patched = open(TARGET).read()
assert 'merged["data"]["split"] = requested_split' in patched
assert patched.count("PATCHED (fusion_ai Move-2)") == 1
compile(patched, TARGET, "exec")
print(f"patched: {TARGET} (warmstart finetune now honors requested data.split)")

#!/usr/bin/env python3
"""
Hard gate for the Move-2 run: verifies the invariant that caused the recurring
`FileNotFoundError: .../14305.zarr` when broken.

The invariant: the CSV the *running code actually reads* (i.e. the one resolved by
the installed tokamark package), the first-N shot selection, and the contents of
the local data directory must agree.

Exits non-zero with a plain-language explanation on any failure.

Env knobs: WORK (default /data_local/work), DATA (default /data_local/mast_data),
TRAIN_N (300), EVAL_N (100).
"""

import csv
import os
import sys

WORK = os.environ.get("WORK", "/data_local/work")
DATA = os.environ.get("DATA", "/data_local/mast_data")
TRAIN_N = int(os.environ.get("TRAIN_N", "300"))
EVAL_N = int(os.environ.get("EVAL_N", "100"))

failures = []


def check(ok, msg_ok, msg_fail):
    if ok:
        print(f"  OK   {msg_ok}")
    else:
        print(f"  FAIL {msg_fail}")
        failures.append(msg_fail)
    return ok


print("== 1. Which tokamark does Python import? ==")
import tokamark  # noqa: E402
import tokamark.tools.path as tkpath  # noqa: E402

pkg_file = os.path.realpath(tokamark.__file__)
print(f"  tokamark.__file__ = {pkg_file}")
check(
    pkg_file.startswith(os.path.realpath(WORK)),
    f"import resolves inside {WORK}",
    f"tokamark is imported from OUTSIDE {WORK} — this is the 14305 bug. "
    f"Fix: pip install -e {WORK}/tokamark --force-reinstall --no-deps",
)

print("== 2. Which temporal CSV does that package resolve, and is it the filtered one? ==")
csv_path = tkpath.TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE
print(f"  resolved CSV: {csv_path}")
rows = list(csv.DictReader(open(csv_path)))
check(
    len(rows) == TRAIN_N + 2 * EVAL_N,
    f"CSV has {len(rows)} rows (expected {TRAIN_N + 2 * EVAL_N})",
    f"CSV has {len(rows)} rows, expected {TRAIN_N + 2 * EVAL_N}. The package is serving an "
    f"UNFILTERED CSV. Fix: copy data/TokaMark_temporal_data_splits_filtered500.csv over it "
    f"(setup_pod.sh does this).",
)

print("== 3. Does the first-N selection match the shots on disk? ==")
train = [r["shot_id"] for r in rows if r["train"] == "True"][:TRAIN_N]
val = [r["shot_id"] for r in rows if r["val"] == "True"][:EVAL_N]
test = [r["shot_id"] for r in rows if r["test"] == "True"][:EVAL_N]
selection = train + val + test
on_disk = {d[:-5] for d in os.listdir(DATA) if d.endswith(".zarr")} if os.path.isdir(DATA) else set()
missing = [s for s in selection if s not in on_disk]
check(
    not missing,
    f"all {len(selection)} selected shots present in {DATA} ({len(on_disk)} on disk)",
    f"{len(missing)} selected shots MISSING from {DATA} (first few: {missing[:5]}). "
    f"Run pod/download_shots.py.",
)
check(
    "14305" not in selection,
    "canary: 14305 is not in the selection",
    "canary FAILED: 14305 is in the selection — the CSV is not the expected filtered one.",
)

print("== 3b. Random-split CSV (used by DCT3D tuning via warmstart inheritance) ==")
rcsv_path = tkpath.RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE
rrows = list(csv.DictReader(open(rcsv_path)))
r_ids = {r["shot_id"] for r in rrows}
check(
    len(rrows) <= TRAIN_N + 2 * EVAL_N,
    f"random CSV has {len(rrows)} rows (filtered)",
    f"random CSV at {rcsv_path} has {len(rrows)} rows — UNFILTERED. The seed-54 embeddings-tuning "
    f"shuffle selects shot 14305 from it (position 0). Fix: setup_pod.sh step 4 installs "
    f"data/TokaMark_data_splits_filtered500.csv over it.",
)
r_missing = sorted(r_ids - on_disk) if os.path.isdir(DATA) else sorted(r_ids)
check(
    not r_missing,
    "every shot in the random CSV is on disk",
    f"{len(r_missing)} random-CSV shots not on disk (first few: {r_missing[:5]}).",
)
check(
    "14305" not in r_ids,
    "canary: 14305 is not in the random CSV",
    "canary FAILED: 14305 present in the random CSV — it is not the filtered version.",
)

print("== 3c. Warmstart inheritance patch (honor requested data.split) ==")
inh_path = os.path.join(WORK, "tokamind", "scripts_mast", "mast_utils", "config", "inheritance.py")
inh_src = open(inh_path).read() if os.path.isfile(inh_path) else ""
check(
    "PATCHED (fusion_ai Move-2)" in inh_src,
    "inheritance.py is patched — finetune will honor data.split: temporal",
    f"inheritance.py NOT patched — warmstart would force data.split back to the source model's "
    f"'random', silently breaking the temporal design. Run pod/patch_inheritance.py.",
)

print("== 4. Subset-of-shots guard ==")
# Even if configs regress to subset_of_shots: null, a filtered CSV caps exposure at 500 shots.
n_train_total = sum(1 for r in rows if r["train"] == "True")
check(
    n_train_total <= TRAIN_N,
    f"CSV train rows ({n_train_total}) <= TRAIN_N — a null subset_of_shots cannot request unknown shots",
    f"CSV has {n_train_total} train rows > TRAIN_N={TRAIN_N}; a stale subset_of_shots: null "
    f"would request shots that are not on disk.",
)

print("== 5. Pretrained weights ==")
weights_dir = os.path.join(WORK, "tokamind", "runs", "tokamind-base-v2")
has_files = os.path.isdir(weights_dir) and any(os.scandir(weights_dir))
check(
    has_files,
    f"{weights_dir} exists and is non-empty",
    f"{weights_dir} missing/empty. Fetch from HF UKAEA-IBM-STFC/tokamind-base-v2 (setup_pod.sh).",
)

print("== 6. The mmt.eval.forward import bug is patched ==")
try:
    import mmt.eval.forward  # noqa: F401

    check(True, "import mmt.eval.forward succeeds", "")
except Exception as e:  # noqa: BLE001
    check(False, "", f"import mmt.eval.forward failed ({type(e).__name__}: {e}). Run pod/patch_forward.py.")

print("== 7. GPU ==")
try:
    import torch

    if torch.cuda.is_available():
        print(f"  OK   torch {torch.__version__}, CUDA available: {torch.cuda.get_device_name(0)}")
    elif os.environ.get("ALLOW_CPU") == "1":
        print("  WARN CUDA not available, but ALLOW_CPU=1 — proceeding CPU-only (very slow).")
    else:
        check(
            False,
            "",
            f"torch {torch.__version__} cannot use the GPU (often a torch build newer than the "
            f"host driver's CUDA). setup_pod.sh step 3b fixes this; or set ALLOW_CPU=1 to "
            f"deliberately run on CPU.",
        )
except Exception as e:  # noqa: BLE001
    check(False, "", f"torch import failed: {e}")

print()
if failures:
    print(f"VERIFY FAILED — {len(failures)} problem(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("VERIFY PASSED — the setup invariant holds.")

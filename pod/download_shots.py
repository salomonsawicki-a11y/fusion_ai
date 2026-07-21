#!/usr/bin/env python3
"""
Download MAST shot zarr stores for the Move-2 selection from the STFC anonymous S3.

Fetches exactly the first-TRAIN_N/EVAL_N/EVAL_N (train/val/test) selection from the
CSV that the *installed* tokamark package resolves — the same CSV the pipeline will
read — so the download and the run can never disagree.

Env knobs: DATA (default /data_local/mast_data), TRAIN_N (300), EVAL_N (100).
"""

import csv
import os
import shutil
import sys

import s3fs

ENDPOINT = "https://s3.echo.stfc.ac.uk"
PREFIX = "mast/tokamark/v1"

DATA = os.environ.get("DATA", "/data_local/mast_data")
TRAIN_N = int(os.environ.get("TRAIN_N", "300"))
EVAL_N = int(os.environ.get("EVAL_N", "100"))


def main() -> int:
    import tokamark.tools.path as tkpath

    csv_path = tkpath.TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE
    print(f"selection CSV (as resolved by installed tokamark): {csv_path}")
    rows = list(csv.DictReader(open(csv_path)))
    train = [r["shot_id"] for r in rows if r["train"] == "True"][:TRAIN_N]
    val = [r["shot_id"] for r in rows if r["val"] == "True"][:EVAL_N]
    test = [r["shot_id"] for r in rows if r["test"] == "True"][:EVAL_N]
    selection = train + val + test
    print(f"selection: {len(train)} train + {len(val)} val + {len(test)} test = {len(selection)} shots")

    os.makedirs(DATA, exist_ok=True)
    present = [s for s in selection if os.path.isdir(os.path.join(DATA, f"{s}.zarr"))]
    to_fetch = [s for s in selection if s not in set(present)]
    print(f"already present: {len(present)}, to fetch: {len(to_fetch)}")
    if not to_fetch:
        return 0

    fs = s3fs.S3FileSystem(anon=True, endpoint_url=ENDPOINT)
    n_err = 0
    for i, shot in enumerate(to_fetch, 1):
        src = f"{PREFIX}/{shot}.zarr"
        dst = os.path.join(DATA, f"{shot}.zarr")
        tmp = dst + ".partial"
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            fs.get(src, tmp, recursive=True)
            os.rename(tmp, dst)
            print(f"[{i}/{len(to_fetch)}] {shot} done")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"[{i}/{len(to_fetch)}] {shot} FAILED: {e}")

    if n_err:
        print(f"{n_err} shots failed — rerun this script to retry (it skips completed shots).")
        return 1
    print("all shots downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Swap / restore the `test` flags in the temporal split CSV for the "familiar" eval.

Eval only scores shots flagged `test`. The OOD eval uses the CSV as-is (test =
new campaigns M8/M9). For the familiar eval we point `test` at the old-campaign
val shots instead:

  swap:    test <- val   (val rows become the scored set; original test rows off)
  restore: put the original CSV back from the .bak backup

Operates on the CSV resolved by the *installed* tokamark package — the one the
pipeline actually reads. A .bak backup is created on swap; if a run died mid-swap,
`restore` puts the original back.
"""

import csv
import os
import shutil
import sys


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("swap", "restore", "status"):
        print("usage: swap_test_val.py {swap|restore|status}")
        return 2

    import tokamark.tools.path as tkpath

    path = tkpath.TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE
    bak = path + ".bak"
    mode = sys.argv[1]

    if mode == "status":
        print(f"CSV: {path}")
        print(f"backup exists (CSV currently swapped): {os.path.exists(bak)}")
        return 0

    if mode == "restore":
        if not os.path.exists(bak):
            print(f"nothing to restore ({bak} absent) — CSV is already the original.")
            return 0
        shutil.move(bak, path)
        print(f"restored original CSV from {bak}")
        return 0

    # swap
    if os.path.exists(bak):
        print(f"refusing to swap: {bak} already exists (CSV appears already swapped). "
              f"Run 'restore' first.")
        return 1
    shutil.copy2(path, bak)
    rows = list(csv.DictReader(open(path)))
    fieldnames = list(rows[0].keys())
    n_scored = 0
    for r in rows:
        was_val = r["val"] == "True"
        r["test"] = "True" if was_val else "False"
        if was_val:
            n_scored += 1
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"swapped: test <- val ({n_scored} shots now scored as 'test'); backup at {bak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

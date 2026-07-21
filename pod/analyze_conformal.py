#!/usr/bin/env python3
"""
Split-conformal coverage analysis for the Move-2 OOD experiment.

NOTE: this is a RECONSTRUCTION. The validated original was lost with the pod's
container disk. If you have a backup of the original analyze_conformal.py, prefer
it. Method here (documented so results are interpretable):

  1. Load eval traces ({shot}__{signal}.npz, keys true/pred) from --familiar
     (old-campaign val shots) and --ood (new-campaign M8/M9 test shots).
  2. Absolute residuals |true - pred| per point (finite values only).
  3. Familiar shots are split 50/50 deterministically (sorted shot id, alternating)
     into a CALIBRATION half and a FAMILIAR-COVERAGE half, so the familiar coverage
     number is out-of-sample rather than trivially ~0.90.
  4. q_hat = conformal quantile of calibration residuals at level 0.90 with the
     (n+1) finite-sample correction; intervals are pred +/- q_hat.
  5. Coverage = fraction of points with residual <= q_hat, reported for the
     familiar half and for all OOD shots, plus per-shot spread.

Reading the result: cov_familiar@90 ~ 0.90 is the sanity gate; cov_OOD@90 clearly
below it is the Move-2 finding (overconfidence out of distribution).
"""

import argparse
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np


def load_residuals_by_shot(traces_dir: str) -> dict[str, np.ndarray]:
    """Return {shot_id: 1-D array of |true-pred| residuals} for every npz trace."""
    by_shot: dict[str, list[np.ndarray]] = defaultdict(list)
    files = sorted(glob.glob(os.path.join(traces_dir, "*.npz")))
    if not files:
        sys.exit(f"no .npz traces found in {traces_dir}")
    for f in files:
        shot = os.path.basename(f).split("__")[0]
        d = np.load(f)
        res = np.abs(d["true"].astype(np.float64) - d["pred"].astype(np.float64)).ravel()
        res = res[np.isfinite(res)]
        if res.size:
            by_shot[shot].append(res)
    return {s: np.concatenate(chunks) for s, chunks in by_shot.items()}


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample-corrected split-conformal quantile at level 1-alpha."""
    n = residuals.size
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, level, method="higher"))


def coverage(by_shot: dict[str, np.ndarray], q: float) -> tuple[float, list[float]]:
    all_res = np.concatenate(list(by_shot.values()))
    per_shot = [float(np.mean(r <= q)) for r in by_shot.values()]
    return float(np.mean(all_res <= q)), per_shot


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--familiar", required=True, help="traces dir for the familiar (old-campaign val) eval")
    p.add_argument("--ood", required=True, help="traces dir for the OOD (M8/M9 test) eval")
    p.add_argument("--alpha", type=float, default=0.1, help="miscoverage level (default 0.1 -> 90%% intervals)")
    args = p.parse_args()

    fam = load_residuals_by_shot(args.familiar)
    ood = load_residuals_by_shot(args.ood)

    fam_shots = sorted(fam.keys())
    calib_shots = fam_shots[0::2]
    eval_shots = fam_shots[1::2]
    if not calib_shots or not eval_shots:
        sys.exit("need at least 2 familiar shots to split into calibration/coverage halves")

    calib_res = np.concatenate([fam[s] for s in calib_shots])
    q = conformal_quantile(calib_res, alpha=args.alpha)

    fam_eval = {s: fam[s] for s in eval_shots}
    cov_fam, per_fam = coverage(fam_eval, q)
    cov_ood, per_ood = coverage(ood, q)

    pct = int(round((1 - args.alpha) * 100))
    print()
    print(f"=== Split-conformal coverage @ {pct}% (q_hat = {q:.6g}) ===")
    print(f"calibration: {len(calib_shots)} familiar shots, {calib_res.size} residuals")
    print()
    print(f"{'set':<22}{'shots':>7}{'points':>12}{'coverage':>10}{'per-shot p10':>14}{'p50':>8}{'p90':>8}")
    for name, cov, per, n_shots, n_pts in (
        (f"familiar (held-out)", cov_fam, per_fam, len(eval_shots), sum(v.size for v in fam_eval.values())),
        (f"OOD (M8/M9)", cov_ood, per_ood, len(ood), sum(v.size for v in ood.values())),
    ):
        p10, p50, p90 = np.percentile(per, [10, 50, 90])
        print(f"{name:<22}{n_shots:>7}{n_pts:>12}{cov:>10.3f}{p10:>14.3f}{p50:>8.3f}{p90:>8.3f}")
    print()
    gap = cov_fam - cov_ood
    gate = abs(cov_fam - (1 - args.alpha)) <= 0.03
    print(f"sanity gate (familiar within 0.03 of {1 - args.alpha:.2f}): {'PASS' if gate else 'FAIL'}")
    print(f"familiar - OOD coverage gap: {gap:+.3f}")
    if not gate:
        print("-> familiar coverage is off; debug calibration/training before interpreting the OOD number.")
    elif gap > 0.03:
        print("-> Move-2 finding: the model is measurably overconfident on new campaigns.")
    else:
        print("-> no clear OOD miscoverage at this sample size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

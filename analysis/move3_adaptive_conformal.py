#!/usr/bin/env python3
"""
Move 3 — OOD-aware conformal via ONLINE recalibration.

Move 2 showed that split-conformal 90% intervals, calibrated on familiar
(old-campaign) residuals, cover only ~0.35 of true values on new campaigns
(M8/M9). Root cause: split conformal assumes the calibration and test data are
exchangeable, which breaks under the M5/M6/M7 -> M8/M9 temporal shift.

Physical insight for the fix: this task forecasts plasma current (summary-ip)
100 ms ahead, so the TRUE value is observed ~100 ms after each forecast. A real
control loop therefore sees its own error almost immediately and has no reason to
keep trusting a stale old-campaign calibration. This script recalibrates the
interval width ONLINE as the new campaign streams past, using only information
available at prediction time (past observed errors) — never peeking at the point
being predicted.

Methods compared (all at target 90%):
  static      : Move-2 baseline. One q_hat from familiar calibration, frozen.
  trailing    : q_hat = quantile of the last W observed residuals ("calibrate on
                my recent errors"). Directly learns the new campaign's error scale.
  aci         : Adaptive Conformal Inference (Gibbs & Candes 2021). Keeps the
                familiar residual pool but moves the target level alpha_t up/down
                by gamma each step to chase realized coverage.
  oracle      : NOT usable in practice (peeks at OOD labels). The single static
                q_hat that would give 90% on OOD, shown only as the width target
                the online methods are trying to discover.

Streaming unit is one forecast WINDOW (all its horizon points revealed at once,
matching the physical feedback cadence); coverage is reported per point so the
numbers are directly comparable to Move 2. Shots are streamed in ascending shot-id
order (roughly chronological), windows within a shot in window-index order.

Usage:
  python move3_adaptive_conformal.py --familiar <dir> --ood <dir>
  # dirs are the untarred move2 traces:  tar xzf move2_traces.tgz  -> ood/ familiar/

No GPU required. Runs on the trace files alone.
"""

import argparse
import glob
import math
import os
import sys
from collections import deque

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected split-conformal quantile at level 1-alpha."""
    n = residuals.size
    if n == 0:
        return float("inf")
    alpha = min(max(alpha, 0.0), 1.0)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(1.0, level)
    return float(np.quantile(residuals, level, method="higher"))


def load_windows(traces_dir: str):
    """
    Return a list of per-window residual arrays, ordered by (shot_id, window_index).

    Each element is a 1-D array of |true - pred| for every finite point in that window.
    """
    files = sorted(glob.glob(os.path.join(traces_dir, "*.npz")))
    if not files:
        sys.exit(f"no .npz traces found in {traces_dir}")
    windows = []
    for f in sorted(files, key=lambda p: _shot_key(p)):
        d = np.load(f)
        true = np.asarray(d["true"], dtype=np.float64)
        pred = np.asarray(d["pred"], dtype=np.float64)
        # index the windows in temporal order if a window index is stored
        order = np.argsort(d["window_index"]) if "window_index" in d else np.arange(true.shape[0])
        for i in order:
            r = np.abs(true[i] - pred[i]).ravel()
            r = r[np.isfinite(r)]
            if r.size:
                windows.append(r)
    return windows


def _shot_key(path: str):
    base = os.path.basename(path).split("__")[0]
    try:
        return (0, int(base))
    except ValueError:
        return (1, base)


# ----------------------------------------------------------------------------------------------------------------------
def coverage_of(windows, q_series):
    """Per-point coverage given a per-window half-width series q_series."""
    hit = 0
    tot = 0
    for r, q in zip(windows, q_series):
        hit += int(np.count_nonzero(r <= q))
        tot += r.size
    return hit / tot if tot else float("nan"), tot


def run_static(windows, q_hat):
    return [q_hat] * len(windows)


def run_trailing(windows, calib_residuals, alpha, W):
    """
    q_hat for window t = (1-alpha)-quantile of the last W observed residuals,
    initialised from the familiar calibration pool so we START at the Move-2
    baseline and adapt away from it. Truth for window t is revealed only AFTER
    its interval is emitted (no look-ahead).
    """
    buf = deque(calib_residuals[-W:], maxlen=W)
    q_series = []
    for r in windows:
        q_series.append(conformal_quantile(np.fromiter(buf, dtype=np.float64), alpha))
        buf.extend(r.tolist())
    return q_series


def run_aci(windows, calib_residuals, alpha, gamma):
    """
    Adaptive Conformal Inference: fixed familiar residual pool, moving level alpha_t.
    err_t = fraction of window-t points outside the emitted interval (in [0,1]);
    alpha_{t+1} = clip(alpha_t + gamma*(alpha - err_t), 0, 1).
    """
    pool = np.asarray(calib_residuals, dtype=np.float64)
    a_t = alpha
    q_series = []
    for r in windows:
        q = conformal_quantile(pool, a_t)
        q_series.append(q)
        err = float(np.mean(r > q))  # realized miscoverage this window
        a_t = min(1.0, max(0.0, a_t + gamma * (alpha - err)))
    return q_series


def summarize(name, windows, q_series, warmup, note=""):
    cov, n = coverage_of(windows, q_series)
    widths = np.array([q for q in q_series], dtype=np.float64)
    cov_warm, _ = coverage_of(windows[warmup:], q_series[warmup:])
    finite_w = widths[np.isfinite(widths)]
    mean_w = float(np.mean(finite_w)) if finite_w.size else float("inf")
    print(f"{name:<12}{cov:>10.3f}{cov_warm:>16.3f}{mean_w:>16.0f}   {note}")
    return cov, cov_warm, mean_w


# ----------------------------------------------------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--familiar", required=True, help="familiar (old-campaign val) traces dir")
    p.add_argument("--ood", required=True, help="OOD (M8/M9 test) traces dir")
    p.add_argument("--alpha", type=float, default=0.1, help="miscoverage level (0.1 -> 90%% intervals)")
    p.add_argument("--window", type=int, default=20000, help="trailing-window size in points (default 20000)")
    p.add_argument("--gamma", type=float, default=0.01, help="ACI learning rate (default 0.01)")
    p.add_argument("--warmup", type=int, default=20, help="online warm-up windows excluded from 'post-warmup' coverage")
    args = p.parse_args()

    print("loading traces...")
    fam = load_windows(args.familiar)
    ood = load_windows(args.ood)
    print(f"familiar: {len(fam)} windows | OOD: {len(ood)} windows")

    # Familiar is split 50/50 (alternating in stream order) into calibration and a
    # held-out familiar stream, so the in-distribution control is out-of-sample.
    calib = fam[0::2]
    fam_eval = fam[1::2]
    calib_res = np.concatenate(calib) if calib else np.array([])
    q_static = conformal_quantile(calib_res, args.alpha)

    # Oracle (uses OOD labels — reference only)
    q_oracle = conformal_quantile(np.concatenate(ood), args.alpha)

    print(f"\nstatic q_hat (familiar calibration) = {q_static:.0f}")
    print(f"oracle q_hat (needs OOD labels)      = {q_oracle:.0f}   [reference target width]")

    hdr = f"\n{'method':<12}{'coverage':>10}{'post-warmup':>16}{'mean half-width':>16}"
    print("\n=== OOD (M8/M9) stream, target %.0f%% ===" % ((1 - args.alpha) * 100))
    print(hdr)
    summarize("static", ood, run_static(ood, q_static), args.warmup, "Move-2 baseline (frozen)")
    summarize("trailing", ood, run_trailing(ood, calib_res, args.alpha, args.window), args.warmup,
              f"W={args.window} recent points")
    summarize("aci", ood, run_aci(ood, calib_res, args.alpha, args.gamma), args.warmup, f"gamma={args.gamma}")
    summarize("oracle", ood, run_static(ood, q_oracle), args.warmup, "not deployable; width target")

    print("\n=== FAMILIAR held-out stream (in-distribution control; should stay ~%.0f%%) ===" % ((1 - args.alpha) * 100))
    print(hdr)
    summarize("static", fam_eval, run_static(fam_eval, q_static), args.warmup, "")
    summarize("trailing", fam_eval, run_trailing(fam_eval, calib_res, args.alpha, args.window), args.warmup, "")
    summarize("aci", fam_eval, run_aci(fam_eval, calib_res, args.alpha, args.gamma), args.warmup, "")

    print(
        "\nRead: 'coverage' is over the whole OOD stream (includes cold-start); 'post-warmup'\n"
        "excludes the first %d windows the online methods need to adapt. A good Move-3 result:\n"
        "trailing/aci post-warmup coverage near %.2f (vs static ~0.35), at a mean half-width\n"
        "between the static and oracle values — i.e. honest intervals learned online, no OOD\n"
        "labels used, while the in-distribution control stays ~%.2f." % (
            args.warmup, 1 - args.alpha, 1 - args.alpha)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

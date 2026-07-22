# Move-2 result — 2026-07-22

Finetune of tokamind-base-v2 on task_4-4 (forecast summary-ip 100 ms ahead), temporal
split: train/val = campaigns M5/M6/M7, test = M8/M9. 300/100/100 shots, batch 32,
best val loss 0.1998 (ft_full epoch 8). Split-conformal 90% intervals calibrated on
50 held-out-from-calibration familiar (old-campaign val) shots. Full log: move2.log;
traces: move2_traces.tgz (both archived on the user's machine).

```
=== Split-conformal coverage @ 90% (q_hat = 173595) ===
calibration: 50 familiar shots, 4739600 residuals

set                     shots       points   coverage   per-shot p10    p50    p90
familiar (held-out)        50      4635600      0.908          0.795  0.911  0.998
OOD (M8/M9)               100    119938800      0.352          0.169  0.393  0.417

sanity gate (familiar within 0.03 of 0.90): PASS
familiar - OOD coverage gap: +0.556
```

## Interpretation

- Sanity gate passes (0.908 vs 0.90 nominal), matching Move 1's in-distribution 0.909 —
  the conformal machinery is working.
- On new campaigns the nominally-90% intervals cover only 35.2% of true values, despite
  being +/-174 kA wide (MAST Ip is ~400-900 kA). The miscoverage is systematic: even the
  90th-percentile OOD shot only reaches 0.417 coverage.
- Conclusion (Move-2 thesis confirmed): the model is severely overconfident
  out-of-distribution; split-conformal calibration transported across campaigns fails.

## Suggested tightening analyses (all runnable from the traces tarball, no GPU)

- M8 vs M9 coverage separately (does miscoverage grow with campaign distance?)
- Per-shot coverage distributions / identification of any covered OOD shots
- Interval width vs |error| scatter, familiar vs OOD
- Coverage as a function of forecast horizon position within the 100 ms window

## Task-level eval metrics (familiar eval, from log)

NRMSE_mean 0.439 | NMAE_mean 0.259 | RMSE_mean 132.5 kA | MAE_mean 78.0 kA | 23438 windows

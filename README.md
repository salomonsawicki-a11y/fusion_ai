# fusion_ai — Move-2 OOD experiment kit

Everything needed to rebuild a fresh GPU pod and run the Move-2 experiment:
finetune TokaMind (task_4-4: forecast plasma current `summary-ip` 100 ms ahead) on the
**temporal** TokaMark split (train/val = old campaigns M5/M6/M7, test = new campaigns
M8/M9), then compare split-conformal 90% coverage on familiar vs out-of-distribution
shots. Thesis: the model is overconfident OOD (familiar coverage ≈ 0.90 is the sanity
gate; OOD clearly below it is the finding).

This kit exists because the previous pod's container disk — holding the only copies of
the run scripts — was wiped on pod stop. Everything here is committed so that can never
hurt again. `DIAGNOSIS_14305.md` explains the recurring `14305.zarr FileNotFoundError`
that blocked the last attempts and how this kit makes it structurally impossible.

## Quickstart on a fresh pod (RunPod, 1× L4, ≥30 GB container disk)

**When deploying, filter hosts by CUDA version >= 13.0** (Filter button next to the
GPU search box). tokamind requires torch>=2.10, whose default builds need CUDA 13;
on an older-driver host (e.g. CUDA 12.8) torch silently loses the GPU. setup_pod.sh
step 3b can usually repair that by installing an older-CUDA torch build, but picking
a CUDA >= 13.0 host avoids the issue and a multi-GB reinstall entirely. No network
volume needed; persistent storage 0 GB.

```bash
# on the pod, inside tmux:
git clone https://github.com/salomonsawicki-a11y/fusion_ai.git /data_local/fusion_ai   # private repo: clone via token, or scp the checkout
cd /data_local/fusion_ai
bash pod/setup_pod.sh                                    # clone repos, install, weights, 6.8 GB data, verify (~20-40 min)
bash pod/run_move2.sh 2>&1 | tee /data_local/work/move2.log   # 1-3 h on L4
```

The run ends by printing the conformal coverage table. Before stopping the pod, copy
`/data_local/work/move2.log` and `/data_local/work/move2_traces.tgz` to your Mac (scp
commands are printed at the end of the run). **Do not** rely on `/workspace` (network
volume) — it returns phantom errors; treat it as cold storage at best.

## Files

| File | Purpose |
|---|---|
| `pod/setup_pod.sh` | Full pod rebuild: repos, editable installs, weights, data, verification |
| `pod/run_move2.sh` | The experiment: verify → configure → finetune → eval OOD → eval familiar → conformal table |
| `pod/verify_setup.py` | Hard gate: the CSV the *running code* reads, the selection, and the disk must agree |
| `pod/edit_configs.py` | Writes config YAMLs and re-reads/asserts every value (no sed) |
| `pod/download_shots.py` | Anonymous S3 download of the exact 500-shot selection |
| `pod/patch_forward.py` | Idempotent fix for the upstream `mmt.eval.forward` import bug |
| `pod/swap_test_val.py` | `test <- val` CSV swap/restore for the familiar eval |
| `pod/analyze_conformal.py` | Split-conformal coverage table (reconstruction — see header note) |
| `data/TokaMark_temporal_data_splits_filtered500.csv` | The exact 300/100/100 shot selection, committed |
| `DIAGNOSIS_14305.md` | Root-cause analysis of the recurring FileNotFoundError |
| `docs/HANDOFF_CLAUDE_CODE.md` | Original handoff (historical context; machine state described there is gone) |

## Knobs (env vars)

`TRAIN_N` (300), `EVAL_N` (100), `BATCH` (32 — drop to 16/8 on CUDA OOM),
`NUM_WORKERS` (8), `WORK` (/data_local/work), `DATA` (/data_local/mast_data).

## Reading the result

- `familiar` coverage far from 0.90 → calibration/training problem; debug before
  interpreting anything else.
- `familiar` ≈ 0.90 and `OOD` clearly lower → the Move-2 finding is in hand. Next
  analyses: per-shot coverage spread (printed), M8 vs M9 separately, width-vs-error.

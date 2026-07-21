# CLAUDE CODE HANDOFF — Fusion OOD Move-2 Run (diagnose, then execute)

You are Claude Code running ON the RunPod pod. Your job: get `run_move2.sh` to run to
completion and print the final conformal coverage table. State as of Mon Jul 13, 2026,
~9 PM ET. The user is beginner-intermediate in ML and has burned many hours on
infrastructure failures — be direct, verify everything yourself, do not ask them to
run commands you can run.

## FIRST TASK — diagnose the recurring FileNotFoundError (unresolved)

A `FileNotFoundError: /data_local/mast_data/14305.zarr does not exist` keeps recurring,
INCLUDING after a verified-consistent setup. Known facts:

- The split CSV at `/data_local/work/tokamark/src/tokamark/metadata/TokaMark_temporal_data_splits.csv`
  was filtered to 500 rows. Verified minutes before the last failure: 500 rows, 500 shots
  on disk in `/data_local/mast_data`, downloader reports "already present: 500, to fetch: 0",
  and **shot 14305 is NOT in the first-300/100/100 selection from that CSV** (checked
  programmatically, `14305 in need` → False).
- Yet something still requests 14305.zarr. **The user never pasted the full traceback of the
  last failure — only the final line.** Do not assume which component raised it.

Hypotheses to check, in order:
1. **Stale config state.** `run_move2.sh` edits configs in
   `tokamind/scripts_mast/configs/common/` in place (finetune_warmstart.yaml, eval.yaml).
   A previous partial run may have left `data.local_path`, `split`, or `subset_of_shots`
   pointing somewhere stale. READ both YAMLs before anything.
2. **A second copy of the CSV.** The pipeline may resolve the metadata CSV from the
   *installed* tokamark package path, not the git checkout the user edited. Check
   `pip show tokamark` (is it `-e` editable, and pointing at `/data_local/work/tokamark`?).
   If pip install ran against a different checkout (e.g. an old `/workspace/tokamark`),
   the pipeline reads a DIFFERENT, unfiltered CSV whose first-300 selection includes 14305.
   **This is the leading suspect** — the user previously had repos at /workspace and
   reinstalled multiple times across pod restarts.
3. **Shot-selection logic mismatch.** Verify in tokamark source how
   `subset_of_shots` selects shots (`tokamark.data_split.get_train_test_val_shots`,
   documented as first-N in CSV row order, shuffle=False) — confirm against the actual
   installed source, not documentation.
4. Only if 1–3 are clean: rerun and capture the FULL traceback from
   `/data_local/work/move2.log` to see which module and which split triggers it.

Fix accordingly. The clean invariant you are aiming for: the CSV the *running code
actually reads*, the shot-selection logic, and the contents of `/data_local/mast_data`
must agree. When in doubt, the robust move is: `pip install -e /data_local/work/tokamark
--force-reinstall --no-deps`, confirm `python -c "import tokamark; print(tokamark.__file__)"`
points into `/data_local/work`, and re-verify the CSV that file's package actually loads.

## What this project is (context)

Research on TokaMind foundation model + TokaMark benchmark (MAST tokamak plasma data,
GitHub org UKAEA-IBM-STFC-Fusion-FMs). Thesis: these models are overconfident
out-of-distribution. Move 1 (done, validated): split-conformal 90% intervals, in-dist
coverage 0.909. **Move 2 (this run):** finetune task_4-4 (forecast plasma current
"summary-ip" 100ms ahead) with `split: temporal` — train = old campaigns M5/M6/M7,
test = new campaigns M8/M9. Calibrate on old-campaign val residuals, compare coverage:
familiar ≈ 0.90 is the sanity gate; OOD clearly below it is the finding. Move 3 (later):
physics-informed OOD-aware conformal fix.

## Machine state

- RunPod pod, 1× L4, 30 GB container disk (`/`), network volume at `/workspace`.
- **`/workspace` (network volume) is UNRELIABLE** — intermittent phantom
  FileNotFoundError on files that exist. DO NOT read or write anything under /workspace.
  Everything lives on local disk under `/data_local`. (Master 81 GB dataset and old work
  sit on /workspace; treat as cold storage only.)
- `/data_local/work/` — fresh git clones of `tokamind` and `tokamark`; project files
  `run_move2.sh` (chmod +x), `analyze_conformal.py`, `download_shots.py`.
- `/data_local/work/tokamind/runs/tokamind-base-v2` — pretrained weights from HF
  `UKAEA-IBM-STFC/tokamind-base-v2`. Verify present and non-empty.
- `/data_local/mast_data/` — 500 shots ({shot_id}.zarr), matching the filtered CSV's
  first-300/100/100 (train/val/test) selection.
- pip: `tokamark` and `tokamind` installed editable; s3fs, huggingface_hub present.
  torch 2.13.0+cu130, CUDA verified True on L4. Python 3.12.
- Container disk is 30 GB and was once filled to 100% — keep caches purged
  (`rm -rf /root/.cache/pip /root/.cache/huggingface` after any installs).
- **Container disk is wiped on pod stop.** /data_local must be rebuilt after any restart
  (repos from GitHub, weights from HF, data via download_shots.py — never from /workspace).
- Work in tmux (`tmux attach` / `tmux new -s work`).

## Known repo facts (verified from source previously — trust but re-verify cheaply)

1. REPO BUG: TYPE_CHECKING-guarded names used in runtime annotations;
   `mmt/eval/forward.py` raises NameError: TorchDecoder on import.
   run_move2.sh step 0 patches idempotently (inserts `from __future__ import annotations`)
   and hard-verifies the import. Keep that step.
2. `data.split: temporal` → TokaMark_temporal_data_splits.csv (train+val = M5/M6/M7,
   test = M8/M9). Standardization stats are per-split.
3. `data.subset_of_shots: N` = FIRST N rows per split in CSV row order. Deterministic.
4. Eval only scores the "test" flag. run_move2.sh step 5 swaps test←val in the CSV for
   the familiar eval, then restores it. If a run died mid-step-5, CHECK the CSV wasn't
   left swapped (a `.bak` may exist alongside).
5. Data: anonymous S3, endpoint https://s3.echo.stfc.ac.uk, prefix mast/tokamark/v1,
   one {shot_id}.zarr per shot. download_shots.py fetches exactly the first-N selection
   from whatever CSV it's pointed at. Shots for this task average ~14 MB (500 ≈ 6.8 GB).
6. Model ~9M params — never GPU-bound; task_4-4 is data-heavy. Benign warnings:
   numba/numpy gripe, fsspec conflicts, leaked semaphore, torchvision/torchaudio
   version mismatch.

## The run

```bash
cd /data_local/work
WORK=/data_local/work DATA=/data_local/mast_data ./run_move2.sh 2>&1 | tee /data_local/work/move2.log
```

Env knobs: TRAIN_N (300), EVAL_N (100), BATCH (32; drop to 16/8 on CUDA OOM).
Expected wall-clock 1–3 hrs, dominated by the 15-epoch full-model stage.
Traces land at `runs/<run>/eval/traces/{shot}__{signal}.npz` (keys true/pred/window_index).
Final step runs `analyze_conformal.py --familiar /tmp/m2/familiar --ood /tmp/m2/ood`
(validated logic — reuse, never rewrite) and prints the table.

## Reading the result

- `cov_familiar@90` far from 0.90 → calibration/training problem; debug before
  interpreting anything.
- `cov_familiar@90` ≈ 0.90 and `cov_OOD@90` clearly lower → the move-2 finding is in hand.
- Report to the user: the table, plain-language interpretation, and whether the sanity
  gate passed. If the gap shows, suggest the tightening analyses (per-shot coverage
  spread, M8 vs M9 separately, width-vs-error scatter).

## After success — protect the results

Immediately copy off the container disk (which dies on pod stop). /workspace is
untrustworthy; prefer having the user scp to their Mac:
`scp -P <port> -i ~/.ssh/id_ed25519 root@<pod-ip>:/data_local/work/move2.log .`
plus the traces dir (tar it first: `tar czf /data_local/work/move2_traces.tgz -C /data_local/work/tokamind/runs <run>/eval` ).
As a secondary copy, attempt rsync to /workspace but treat success as a bonus.
Then the user stops the pod.

## Working norms

Plain language, no jargon. One step at a time, verify after each. Run commands yourself
rather than instructing the user where possible. Read actual source before touching
repo APIs — do not invent config keys. If something fails, capture the FULL traceback
before theorizing.

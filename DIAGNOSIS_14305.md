# Diagnosis: the recurring `FileNotFoundError: /data_local/mast_data/14305.zarr`

## THE ACTUAL ROOT CAUSE (proven 2026-07-22, after the error recurred past a passing verify)

The error survived a fully verified setup, which eliminated every hypothesis based on
the temporal CSV and exposed the real chain:

1. `config/inheritance.py` (tokamind): a **warmstart finetune force-overrides the
   requested `data.split` with the source model's split**, logging only a warning.
   `tokamind-base-v2` was pretrained on `split: random`, so the merged finetune config
   silently becomes `random` no matter what the YAMLs say.
2. The DCT3D embeddings tuning step (`tune_dct3d.py`) then selects
   `n_shots: 100` training shots from the **random** split CSV with
   `shuffle=True, seed=54` (the config seed).
3. Deterministic replication: in that shuffled selection, **shot 14305 is position 0**
   — the first zarr opened. 97 of the 100 selected shots aren't on disk; 14305 is
   simply first in line. Same seed + same CSV = same crash on every pod, forever.

The crash was also load-bearing: without it, the finetune (and the eval, which inherits
the finetuned run's split) would have **silently run on the random split**, invalidating
the temporal OOD design of Move 2.

Fixes (all in this kit, gated by verify):
- `pod/patch_inheritance.py` — warmstart finetune honors the requested split
  (deliberate scientific choice: temporal finetune from a random-pretrained base;
  upstream's warning stays in the logs).
- `data/TokaMark_data_splits_filtered500.csv` — the random CSV is also replaced with a
  filtered 500-shot version (same on-disk shots, random-CSV schema), so ANY selection
  from either CSV — any seed, any shuffle, any subset — can only request on-disk shots.
- `pod/verify_setup.py` checks 3b/3c enforce both, with a 14305 canary on each CSV.

The analysis below (2026-07-21) predates this finding. Its two config hazards were real
and the structural fixes remain in place, but the deterministic 14305 selector was the
seed-54 shuffle over the random CSV described above.

---

## Earlier analysis (2026-07-21) — superseded but kept for the record

Investigated from upstream source (repos cloned at HEAD; CSV checked across
its full git history). Conclusion first, evidence below.

## Conclusion

Two config failures had to hold **simultaneously**, and by elimination both did:

1. **The run never read the filtered 500-row CSV.** The pipeline resolves the split CSV
   relative to the *installed* `tokamark` package (`tokamark/tools/path.py` builds paths
   from `Path(__file__)`). If pip's `tokamark` is not the editable install of the edited
   checkout, the run reads a pristine 11,188-row CSV. Had the filtered CSV been read, at
   most 500 shots could ever be requested — all of them on disk — so no missing-file
   error is possible, regardless of any other setting.
2. **`subset_of_shots` was effectively `null`.** Shot 14305 is train-flagged in both
   upstream CSVs (temporal position 1218/6525, random 3474/8963) and never appears in any
   first-300/100/100 selection (subsetted, shuffled seed 42, or any historical CSV
   version — the temporal CSV has exactly one content version in git history). The only
   shot lists containing 14305 are **unsubsetted train lists**, and `null` means "use all
   shots" (`data_split.py` line 76 skips the cap). `null` is the shipped default in
   `configs/common/finetune_warmstart.yaml` and `eval.yaml`, so any fresh re-clone or
   failed in-place edit regresses to it.

Why the "verified-consistent" setup still failed: verification checked the git checkout's
CSV and the data directory — but not **which CSV the imported package resolves** nor the
**effective config values at run time**. Those are exactly what `pod/verify_setup.py` and
`pod/edit_configs.py` now check.

Why always shot 14305 and not the first in-order missing shot: with parallel data-loading
workers, which missing shot's exception surfaces first is a race with fixed sharding —
consistent for a given setup, unrelated to CSV order. Immaterial to the fix.

## Supporting facts (all verified against source, not docs)

- `get_train_test_val_shots`: first-N per split in CSV row order; the tokamind call site
  (`entry_helpers.py:165`) passes no `shuffle` → False. `max_index=None` → **no limit**.
- Eval builds its shot list from `model_source["data_split"]`, not `data.split`
  (`entry_helpers.py:153`) — and **nothing upstream ever sets that key**; the run scripts
  must inject `model_source: {data_split: temporal}` into `eval.yaml` (edit_configs.py does).
- Eval only scores `test`-flagged shots; the familiar eval works by swapping `test <- val`
  in the CSV (swap_test_val.py, with backup/restore).
- `mmt/eval/forward.py` NameError-on-import bug (TYPE_CHECKING-guarded `TorchDecoder` in
  runtime annotations) confirmed present at tokamind HEAD; `pod/patch_forward.py` fixes it
  idempotently.
- Eval trace defaults cap traces at `n_max: 2` shots — must be raised for conformal
  analysis (edit_configs.py sets it effectively unlimited).

## The structural fix (all baked into this kit)

1. `pip install -e $WORK/tokamark` and **assert** `tokamark.__file__` is under `$WORK`.
2. Replace the package-resolved temporal CSV with the committed filtered 500-row CSV
   (`data/TokaMark_temporal_data_splits_filtered500.csv`) — then even a regressed
   `subset_of_shots: null` can only request the 500 on-disk shots. The bug becomes
   structurally impossible.
3. Write configs with `edit_configs.py` (re-reads and asserts every value; no sed).
4. Gate every run on `verify_setup.py` (run_move2.sh step 0), which includes a canary:
   14305 must not be in the selection.

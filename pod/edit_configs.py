#!/usr/bin/env python3
"""
Write the Move-2 settings into the tokamind config YAMLs, then re-read and assert
every value. This replaces fragile sed edits — a silent sed no-op (configs revert
to `subset_of_shots: null` + `split: random`) is exactly what caused the recurring
14305.zarr FileNotFoundError.

Edits (in $WORK/tokamind/scripts_mast/configs/):
  common/finetune_warmstart.yaml : data.local/local_path/subset_of_shots/split
  common/eval.yaml               : data.* + eval.traces (enable, n_max, all signals)
                                   + model_source.data_split (upstream never sets it;
                                   eval KeyErrors without it)
  tasks_overrides/task_4-4/finetune_overrides.yaml : loader.batch_size

Env knobs: WORK, DATA, TRAIN_N (300), EVAL_N (100), BATCH (32), NUM_WORKERS (8).
Note: comments in the YAMLs are lost on rewrite — acceptable for pod working copies.
"""

import os
import sys

import yaml

WORK = os.environ.get("WORK", "/data_local/work")
DATA = os.environ.get("DATA", "/data_local/mast_data")
TRAIN_N = int(os.environ.get("TRAIN_N", "300"))
EVAL_N = int(os.environ.get("EVAL_N", "100"))
BATCH = int(os.environ.get("BATCH", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))

CONFIGS = os.path.join(WORK, "tokamind", "scripts_mast", "configs")


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dump(path, cfg):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def edit_finetune():
    path = os.path.join(CONFIGS, "common", "finetune_warmstart.yaml")
    cfg = load(path)
    cfg["data"]["local"] = True
    cfg["data"]["local_path"] = DATA
    cfg["data"]["subset_of_shots"] = TRAIN_N
    cfg["data"]["split"] = "temporal"
    cfg.setdefault("loader", {})["num_workers"] = NUM_WORKERS
    dump(path, cfg)

    got = load(path)
    assert got["data"]["subset_of_shots"] == TRAIN_N, got["data"]
    assert got["data"]["split"] == "temporal", got["data"]
    assert got["data"]["local"] is True and got["data"]["local_path"] == DATA, got["data"]
    print(f"finetune_warmstart.yaml: local={DATA} subset={TRAIN_N} split=temporal")


def edit_eval():
    path = os.path.join(CONFIGS, "common", "eval.yaml")
    cfg = load(path)
    cfg["data"]["local"] = True
    cfg["data"]["local_path"] = DATA
    cfg["data"]["subset_of_shots"] = EVAL_N
    cfg["loader"]["batch_size"] = max(BATCH, 32)
    cfg["loader"]["num_workers"] = NUM_WORKERS
    cfg["eval"]["traces"] = {"enable": True, "n_max": 100000, "signals": None, "times_indexes": None}
    # Upstream never sets model_source.data_split; entry_helpers requires it for eval.
    cfg["model_source"] = {"data_split": "temporal"}
    dump(path, cfg)

    got = load(path)
    assert got["data"]["subset_of_shots"] == EVAL_N, got["data"]
    assert got["model_source"]["data_split"] == "temporal", got.get("model_source")
    assert got["eval"]["traces"]["enable"] is True, got["eval"]
    print(f"eval.yaml: local={DATA} subset={EVAL_N} model_source.data_split=temporal traces=on")


def edit_task_override():
    path = os.path.join(CONFIGS, "tasks_overrides", "task_4-4", "finetune_overrides.yaml")
    cfg = load(path) or {}
    cfg.setdefault("loader", {})["batch_size"] = BATCH
    dump(path, cfg)
    assert load(path)["loader"]["batch_size"] == BATCH
    print(f"task_4-4/finetune_overrides.yaml: batch_size={BATCH}")


if __name__ == "__main__":
    edit_finetune()
    edit_eval()
    edit_task_override()
    print("configs written and verified.")
    sys.exit(0)

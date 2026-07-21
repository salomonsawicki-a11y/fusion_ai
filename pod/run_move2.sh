#!/usr/bin/env bash
# Move-2 OOD experiment: finetune task_4-4 (forecast summary-ip 100 ms ahead) on the
# temporal split (train/val = old campaigns M5/M6/M7, test = new campaigns M8/M9),
# then eval twice (OOD = M8/M9 test; familiar = old-campaign val via test<-val CSV
# swap) and print the split-conformal coverage table.
#
# Usage:  bash pod/run_move2.sh 2>&1 | tee /data_local/work/move2.log
# Knobs:  TRAIN_N (300), EVAL_N (100), BATCH (32; drop to 16/8 on CUDA OOM),
#         WORK, DATA, NUM_WORKERS (8)
# Expect 1-3 h on an L4, dominated by the 15-epoch full-model stage.

set -euo pipefail

export WORK="${WORK:-/data_local/work}"
export DATA="${DATA:-/data_local/mast_data}"
export TRAIN_N="${TRAIN_N:-300}"
export EVAL_N="${EVAL_N:-100}"
export BATCH="${BATCH:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$WORK/tokamind/scripts_mast"
RUNS="$WORK/tokamind/runs"
M2=/tmp/m2

echo "== [0/6] Restore CSV if a previous run died mid-swap, patch + verify =="
python "$KIT/pod/swap_test_val.py" restore
python "$KIT/pod/patch_forward.py"
python "$KIT/pod/verify_setup.py"

echo "== [1/6] Write configs (values re-read and asserted, not sed-and-hope) =="
python "$KIT/pod/edit_configs.py"

echo "== [2/6] Finetune task_4-4 warmstart from tokamind-base-v2 =="
cd "$SCRIPTS"
python run_finetune.py --task task_4-4 --init warmstart --model tokamind-base-v2 --tag move2

# Resolve the finetune run dir (newest ft-* run mentioning the task), don't guess the id format.
RUN_DIR=$(ls -dt "$RUNS"/ft-*task_4-4*/ 2>/dev/null | head -1)
[ -n "$RUN_DIR" ] || { echo "ERROR: no finetune run dir found under $RUNS"; exit 1; }
RUN_ID=$(basename "$RUN_DIR")
echo "finetune run: $RUN_ID"

echo "== [3/6] OOD eval (CSV as-is: test = new campaigns M8/M9) =="
rm -rf "$RUN_DIR/eval/traces" "$M2"
mkdir -p "$M2"
python run_eval.py --task task_4-4 --model "$RUN_ID"
[ -d "$RUN_DIR/eval/traces" ] || { echo "ERROR: no traces at $RUN_DIR/eval/traces"; exit 1; }
mv "$RUN_DIR/eval/traces" "$M2/ood"
echo "OOD traces: $(ls "$M2/ood" | wc -l) files"

echo "== [4/6] Familiar eval (swap test <- val, eval, restore) =="
python "$KIT/pod/swap_test_val.py" swap
trap 'python "$KIT/pod/swap_test_val.py" restore' EXIT   # restore even on failure
python run_eval.py --task task_4-4 --model "$RUN_ID"
python "$KIT/pod/swap_test_val.py" restore
trap - EXIT
[ -d "$RUN_DIR/eval/traces" ] || { echo "ERROR: no traces from familiar eval"; exit 1; }
mv "$RUN_DIR/eval/traces" "$M2/familiar"
echo "familiar traces: $(ls "$M2/familiar" | wc -l) files"

echo "== [5/6] Preserve traces on local disk (pod stop wipes /tmp too) =="
tar czf "$WORK/move2_traces.tgz" -C "$M2" ood familiar
echo "traces archived: $WORK/move2_traces.tgz"

echo "== [6/6] Conformal coverage table =="
python "$KIT/pod/analyze_conformal.py" --familiar "$M2/familiar" --ood "$M2/ood"

echo
echo "DONE. Before stopping the pod, copy results to your Mac:"
echo "  scp -P <port> -i ~/.ssh/id_ed25519 root@<pod-ip>:$WORK/move2.log ."
echo "  scp -P <port> -i ~/.ssh/id_ed25519 root@<pod-ip>:$WORK/move2_traces.tgz ."

#!/usr/bin/env bash
# Rebuild /data_local from scratch on a fresh RunPod pod (container disk is wiped
# on every pod stop). Idempotent: safe to rerun after a partial failure.
#
# Usage (from the fusion_ai checkout):  bash pod/setup_pod.sh
# Env knobs: WORK (/data_local/work), DATA (/data_local/mast_data),
#            TRAIN_N (300), EVAL_N (100)
#
# NEVER read or write /workspace (network volume) — it returns phantom
# FileNotFoundError on files that exist. Everything lives on local disk.

set -euo pipefail

WORK="${WORK:-/data_local/work}"
DATA="${DATA:-/data_local/mast_data}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # fusion_ai checkout root

echo "== [1/7] Directories =="
mkdir -p "$WORK" "$DATA"

echo "== [2/7] Clone tokamind + tokamark (fresh from GitHub, never from /workspace) =="
for repo in tokamind tokamark; do
  if [ ! -d "$WORK/$repo/.git" ]; then
    git clone "https://github.com/UKAEA-IBM-STFC-Fusion-FMs/$repo.git" "$WORK/$repo"
  else
    echo "$repo already cloned"
  fi
done

echo "== [3/7] Install both packages editable against $WORK (the 14305 fix) =="
pip install -e "$WORK/tokamark" -e "$WORK/tokamind"
pip install s3fs huggingface_hub
# Prove the import resolves into $WORK — the exact failure mode behind 14305.zarr:
python -c "
import tokamark, sys
p = tokamark.__file__
assert p.startswith('$WORK'), f'tokamark imports from {p}, not $WORK — editable install failed'
print('tokamark imports from:', p)
"

echo "== [3b] Torch <-> GPU driver compatibility =="
# pip may have upgraded torch to a build compiled for a newer CUDA than the host
# driver supports, leaving torch.cuda.is_available() False -> silent CPU-only runs.
# Reinstall torch/vision from the wheel index matching the driver's CUDA version.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    DRIVER_NUM=$(echo "$CUDA_VER" | tr -d '.')
    echo "torch cannot use the GPU; driver supports CUDA $CUDA_VER"
    # Try PyTorch wheel indexes for CUDA versions the driver can run, newest first.
    # IMPORTANT: no PyPI fallback index here — pip would prefer PyPI's newest
    # (driver-incompatible) build, which is exactly the failure being fixed.
    FIXED=0
    for TAG in cu130 cu129 cu128 cu126 cu124 cu121 cu118; do
      NUM=${TAG#cu}
      [ "$NUM" -le "$DRIVER_NUM" ] || continue
      echo "-> trying torch>=2.10 from https://download.pytorch.org/whl/$TAG"
      if pip install --no-cache-dir --force-reinstall "torch>=2.10" torchvision \
           --index-url "https://download.pytorch.org/whl/$TAG"; then
        FIXED=1
        break
      fi
    done
    rm -rf /root/.cache/pip
    python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
    if [ "$FIXED" != "1" ] || ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      echo "ERROR: could not install a torch>=2.10 build compatible with driver CUDA $CUDA_VER."
      echo "Easiest fix: deploy a pod on a host with CUDA >= 13.0 (Filter -> CUDA version on the deploy page)."
      exit 1
    fi
  else
    echo "WARNING: nvidia-smi not found — no GPU on this machine?"
  fi
else
  python -c "import torch; print('torch', torch.__version__, '| CUDA available: True')"
fi

echo "== [4/7] Replace the package temporal CSV with the filtered 500-shot version =="
CSV=$(python -c "import tokamark.tools.path as p; print(p.TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE)")
if [ ! -f "$CSV.orig" ]; then cp "$CSV" "$CSV.orig"; fi
cp "$KIT/data/TokaMark_temporal_data_splits_filtered500.csv" "$CSV"
echo "filtered CSV installed at: $CSV ($(($(wc -l < "$CSV") - 1)) rows)"

echo "== [5/7] Pretrained weights from HF =="
WEIGHTS="$WORK/tokamind/runs/tokamind-base-v2"
if [ ! -d "$WEIGHTS" ] || [ -z "$(ls -A "$WEIGHTS" 2>/dev/null)" ]; then
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('UKAEA-IBM-STFC/tokamind-base-v2', local_dir='$WEIGHTS')
print('weights downloaded to $WEIGHTS')
"
else
  echo "weights already present"
fi

echo "== [6/7] Shot data (500 zarr stores, ~6.8 GB, anonymous S3) =="
python "$KIT/pod/download_shots.py"

echo "== [7/7] Purge caches (30 GB container disk fills fast) and verify =="
rm -rf /root/.cache/pip /root/.cache/huggingface
python "$KIT/pod/patch_forward.py"
python "$KIT/pod/verify_setup.py"

echo
echo "Setup complete. Run the experiment with:"
echo "  bash $KIT/pod/run_move2.sh 2>&1 | tee $WORK/move2.log"

#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${1:-$ROOT_DIR/.venvs/openvla}"
FLASH_ATTN_WHEEL="$ROOT_DIR/flash_attn-2.5.5+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "[openvla-runtime] creating venv at $RUNTIME_DIR"
uv venv --python 3.10 "$RUNTIME_DIR"

echo "[openvla-runtime] installing PyTorch cu118 stack"
uv pip install --python "$RUNTIME_DIR/bin/python" \
  --index-url "https://download.pytorch.org/whl/cu118" \
  "torch==2.2.0" \
  "torchvision==0.17.0" \
  "torchaudio==2.2.0"

echo "[openvla-runtime] installing minimal OpenVLA runtime dependencies"
uv pip install --python "$RUNTIME_DIR/bin/python" \
  "numpy<2" \
  "pillow>=10,<13" \
  "timm==0.9.10" \
  "tokenizers==0.19.1" \
  "transformers==4.40.1" \
  "huggingface_hub<1" \
  "accelerate>=0.25.0" \
  "sentencepiece==0.1.99" \
  "safetensors>=0.4.1" \
  "einops>=0.7" \
  "json-numpy>=2.1.1"

echo "[openvla-runtime] installing bitsandbytes without dependency upgrades"
uv pip install --python "$RUNTIME_DIR/bin/python" --no-deps \
  "bitsandbytes==0.43.1"

if [[ -f "$FLASH_ATTN_WHEEL" ]]; then
  echo "[openvla-runtime] installing local flash-attn wheel"
  uv pip install --python "$RUNTIME_DIR/bin/python" "$FLASH_ATTN_WHEEL"
else
  echo "[openvla-runtime] flash-attn wheel not found, leaving attn_implementation unset"
fi

cat <<EOF

[openvla-runtime] ready
  python: $RUNTIME_DIR/bin/python
  configure configs/vla_models.json to point runtime_python at this interpreter
EOF

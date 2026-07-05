#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is not installed. Install it with: pip install huggingface_hub" >&2
  exit 1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

read -r HF_REPO MODEL_DIR MODEL_SUBDIR <<EOF_PY
$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import yaml
from pathlib import Path
import os

config_path = Path(os.environ.get("CONFIG_PATH", "configs/default.yaml"))
if not config_path.exists():
    raise SystemExit(f"Config not found: {config_path}")

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

models = config.get("models", {})
print(models.get("hf_repo", ""), models.get("local_dir", "models"), models.get("local_subdir", ""))
PY
)
EOF_PY

if [[ -z "${HF_REPO}" || -z "${MODEL_SUBDIR}" ]]; then
  echo "Missing models.hf_repo or models.local_subdir in ${CONFIG_PATH}" >&2
  exit 1
fi

DEST_DIR="${MODEL_DIR%/}/${MODEL_SUBDIR}"
mkdir -p "${DEST_DIR}"

echo "Downloading ${HF_REPO} to ${DEST_DIR}"
huggingface-cli download "${HF_REPO}" --local-dir "${DEST_DIR}" --local-dir-use-symlinks False

echo "Done. If you change models, update ${CONFIG_PATH} and re-run this script."

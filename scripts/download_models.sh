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

read -r MODEL_DIR <<EOF_PY
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
print(models.get("local_dir", "models"))
PY
)
EOF_PY

mapfile -t MODEL_LINES <<EOF_PY
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
model_list = models.get("model_list") or []
for entry in model_list:
    model_id = entry.get("model_id")
    model_subdir = entry.get("model_subdir")
    hf_repo = entry.get("hf_repo") or model_id
    if not model_id or not model_subdir:
        continue
    print(f"{hf_repo}\t{model_subdir}")
PY
)
EOF_PY

if [[ "${#MODEL_LINES[@]}" -eq 0 ]]; then
  echo "No models.model_list entries found in ${CONFIG_PATH}" >&2
  exit 1
fi

for line in "${MODEL_LINES[@]}"; do
  HF_REPO=$(echo "${line}" | cut -f1)
  MODEL_SUBDIR=$(echo "${line}" | cut -f2)
  DEST_DIR="${MODEL_DIR%/}/${MODEL_SUBDIR}"
  mkdir -p "${DEST_DIR}"
  echo "Downloading ${HF_REPO} to ${DEST_DIR}"
  huggingface-cli download "${HF_REPO}" --local-dir "${DEST_DIR}" --local-dir-use-symlinks False
done

echo "Done. If you change models.model_list, re-run this script."

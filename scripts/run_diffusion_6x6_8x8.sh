#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PREFIX="${1:-$(date +%Y%m%d_%H%M%S)}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1/models}"

wait_for_vllm() {
  local retries=180
  local sleep_s=5
  for ((i=1; i<=retries; i++)); do
    if curl -fsS "${VLLM_URL}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${sleep_s}"
  done
  echo "vLLM did not become ready at ${VLLM_URL}" >&2
  return 1
}

echo "Starting vLLM if needed..."
docker compose up -d vllm
wait_for_vllm

echo "Running diffusion 6x6..."
docker compose run --rm swarm \
  python -m scripts.run_sweep \
  --config configs/diffusion_6x6.yaml \
  --run-id "${PREFIX}_diffusion_6x6"

echo "Running diffusion 8x8..."
docker compose run --rm swarm \
  python -m scripts.run_sweep \
  --config configs/diffusion_8x8.yaml \
  --run-id "${PREFIX}_diffusion_8x8"

echo "Completed."
echo "Run IDs:"
echo "  ${PREFIX}_diffusion_6x6"
echo "  ${PREFIX}_diffusion_8x8"

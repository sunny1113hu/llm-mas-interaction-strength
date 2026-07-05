#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PREFIX="${1:-$(date +%Y%m%d_%H%M%S)}"

echo "Running consensus RSP rule..."
python -m scripts.run_sweep \
  --config configs/consensus_mobility_scaled.yaml \
  --run-id "${PREFIX}_consensus_mobility_scaled"

echo "Running consensus MobilityRaw..."
python -m scripts.run_sweep \
  --config configs/consensus_mobility_raw.yaml \
  --run-id "${PREFIX}_consensus_mobility_raw"

echo "Running diffusion RSP rule..."
python -m scripts.run_sweep \
  --config configs/diffusion_mobility_scaled.yaml \
  --run-id "${PREFIX}_diffusion_mobility_scaled"

echo "Running diffusion MobilityRaw..."
python -m scripts.run_sweep \
  --config configs/diffusion_mobility_raw.yaml \
  --run-id "${PREFIX}_diffusion_mobility_raw"

echo "Completed mobility sweeps."
echo "Run IDs:"
echo "  ${PREFIX}_consensus_mobility_scaled"
echo "  ${PREFIX}_consensus_mobility_raw"
echo "  ${PREFIX}_diffusion_mobility_scaled"
echo "  ${PREFIX}_diffusion_mobility_raw"
